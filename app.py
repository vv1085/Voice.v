"""VOICE: adaptive P300 speller as a single-file Gradio app.

Everything is in this file: settings, P300 decoder, model building and the web UI.

    python app.py                       serve the app (builds the model files first if they are missing)
    python app.py --build               only build the model files (downloads BNCI2014-008, ~5-10 min)
    python app.py --build --source synthetic   offline test data, clearly labelled in the app
    python app.py --share               serve with a public gradio.live link (Colab)

Render:  Build Command  pip install -r requirements.txt && python app.py --build
         Start Command  python app.py
"""

from __future__ import annotations

import argparse
import html
import json
import os
import secrets
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

__version__ = "1.1.0"

# `config.X` refers to the constants defined in this module.
config = sys.modules[__name__]

# ==========================================================================
# CONFIG
# ==========================================================================
ROOT_DIR = Path(__file__).resolve().parent

# 6 x 6 speller grid (same layout as the notebook).
GRID: list[list[str]] = [
    list("ABCDEF"),
    list("GHIJKL"),
    list("MNOPQR"),
    list("STUVWX"),
    list("YZ0123"),
    list("456789"),
]
CHARACTERS: list[str] = [c for row in GRID for c in row]

# BNCI2014-008 "Flash stim" codes: 3-8 = rows, 9-14 = columns.
FLASH_CODES: list[int] = list(range(3, 15))
ROW_CODES: list[int] = list(range(3, 9))
COLUMN_CODES: list[int] = list(range(9, 15))

# Recording structure of BNCI2014-008.
SUBJECTS: list[int] = list(range(1, 9))
TRIALS_PER_SUBJECT = 35
REPETITIONS_PER_TRIAL = 10
EVENTS_PER_TRIAL = len(FLASH_CODES) * REPETITIONS_PER_TRIAL  # 120

# Preprocessing (matches notebook Module 2: MOABB P300 paradigm).
SAMPLING_RATE = 256.0
FILTER_LOW_HZ = 0.5
FILTER_HIGH_HZ = 10.0
EPOCH_START_S = -0.2
EPOCH_END_S = 0.8
DECIMATION = 4            # 256 Hz -> 64 Hz (safe after a 10 Hz low-pass)
SCALE_TO_MICROVOLTS = 1e6  # MOABB returns volts

# First N trials of each subject are calibration data; the rest are spelled.
CALIBRATION_TRIALS = 10

# Adaptive stopping (matches notebook Module 6 VOICE_CONFIG).
STOPPING = {
    "minimum_repetitions": 3,
    "maximum_repetitions": REPETITIONS_PER_TRIAL,
    "confidence_threshold": 0.75,
    "uncertainty_threshold": 0.45,
    "stability_window": 2,
}

MAX_ALTERNATIVES = 3  # next-best characters offered after a rejection

# Runtime locations (override with environment variables on Render).
ARTIFACT_DIR = Path(os.environ.get("VOICE_ARTIFACT_DIR", ROOT_DIR / "artifacts"))
SESSION_DIR = Path(os.environ.get("VOICE_SESSION_DIR", ROOT_DIR / "data" / "sessions"))


def code_to_grid_position(row_code: int, column_code: int) -> tuple[int, int]:
    return ROW_CODES.index(int(row_code)), COLUMN_CODES.index(int(column_code))


def pair_to_character(row_code: int, column_code: int) -> str:
    r, c = code_to_grid_position(row_code, column_code)
    return GRID[r][c]


# ==========================================================================
# DECODER
# ==========================================================================


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # numerically stable logistic


@dataclass
class LinearP300Model:
    """Binary target / non-target classifier: p = sigmoid(w . x + b)."""

    weights: np.ndarray            # (n_channels * n_times,)
    bias: float
    n_channels: int
    n_times: int

    @property
    def n_features(self) -> int:
        return self.n_channels * self.n_times

    def features(self, epochs: np.ndarray) -> np.ndarray:
        epochs = np.asarray(epochs, dtype=np.float64)
        if epochs.ndim == 2:
            epochs = epochs[None]
        if epochs.ndim != 3 or epochs.shape[1:] != (self.n_channels, self.n_times):
            raise ValueError(
                f"Expected epochs shaped (n, {self.n_channels}, {self.n_times}); "
                f"got {tuple(epochs.shape)}."
            )
        if not np.all(np.isfinite(epochs)):
            raise ValueError("EEG epochs contain NaN or infinite values.")
        return epochs.reshape(len(epochs), -1)

    def predict_proba(self, epochs: np.ndarray) -> np.ndarray:
        return _sigmoid(self.features(epochs) @ self.weights + self.bias)


class EvidenceAccumulator:
    """Accumulates log-odds evidence per flash code across repetitions."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.evidence = {code: 0.0 for code in config.FLASH_CODES}
        self.repetitions = 0
        self.previous_pair: tuple[int, int] | None = None
        self.stability = 0

    def update(self, flash_probabilities: dict[int, float]) -> dict:
        if set(flash_probabilities) != set(config.FLASH_CODES):
            raise ValueError("Each repetition must contain all 12 flash codes (3-14).")
        for code, p in flash_probabilities.items():
            p = float(np.clip(p, 1e-5, 1 - 1e-5))
            self.evidence[code] += float(np.log(p / (1 - p)))
        self.repetitions += 1
        return self.state()

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        e = np.exp(scores - scores.max())
        return e / e.sum()

    @staticmethod
    def _normalized_entropy(p: np.ndarray) -> float:
        h = -np.sum(p * np.log(p + 1e-12)) / np.log(len(p))
        return float(np.clip(h, 0.0, 1.0))

    def state(self) -> dict:
        row_p = self._softmax(np.array([self.evidence[c] for c in config.ROW_CODES]))
        col_p = self._softmax(np.array([self.evidence[c] for c in config.COLUMN_CODES]))
        r, c = int(np.argmax(row_p)), int(np.argmax(col_p))
        pair = (config.ROW_CODES[r], config.COLUMN_CODES[c])

        self.stability = self.stability + 1 if pair == self.previous_pair else 1
        self.previous_pair = pair

        return {
            "pair": list(pair),
            "character": config.GRID[r][c],
            "confidence": float(min(row_p.max(), col_p.max())),
            "uncertainty": (self._normalized_entropy(row_p) + self._normalized_entropy(col_p)) / 2,
            "row_probabilities": row_p.round(6).tolist(),
            "column_probabilities": col_p.round(6).tolist(),
            "repetitions": self.repetitions,
            "stability": self.stability,
        }


@dataclass
class AdaptiveStopping:
    minimum_repetitions: int = config.STOPPING["minimum_repetitions"]
    maximum_repetitions: int = config.STOPPING["maximum_repetitions"]
    confidence_threshold: float = config.STOPPING["confidence_threshold"]
    uncertainty_threshold: float = config.STOPPING["uncertainty_threshold"]
    stability_window: int = config.STOPPING["stability_window"]

    def should_stop(self, state: dict) -> tuple[bool, str]:
        reps = state["repetitions"]
        if reps >= self.maximum_repetitions:
            return True, "MAXIMUM_REPETITIONS"
        if reps < self.minimum_repetitions:
            return False, "MINIMUM_REPETITIONS_NOT_REACHED"
        if state["confidence"] < self.confidence_threshold:
            return False, "LOW_CONFIDENCE"
        if state["uncertainty"] > self.uncertainty_threshold:
            return False, "HIGH_UNCERTAINTY"
        if state["stability"] < self.stability_window:
            return False, "PAIR_NOT_STABLE"
        return True, "ADAPTIVE_CONFIDENCE_STOP"


def ranked_characters(state: dict, limit: int) -> list[tuple[str, float]]:
    """Characters ranked by joint row x column probability."""
    joint = np.outer(state["row_probabilities"], state["column_probabilities"])
    order = np.argsort(-joint, axis=None)[:limit]
    rows, cols = np.unravel_index(order, joint.shape)
    return [(config.GRID[r][c], float(joint[r, c])) for r, c in zip(rows, cols)]


@dataclass
class TrialDecoder:
    """Runs one trial repetition by repetition until the stopping rule fires."""

    model: LinearP300Model
    stopping: AdaptiveStopping = field(default_factory=AdaptiveStopping)

    def repetition_probabilities(self, epochs: np.ndarray, flash_codes) -> dict[int, float]:
        flash_codes = [int(c) for c in np.asarray(flash_codes).ravel()]
        if len(flash_codes) != 12 or sorted(flash_codes) != config.FLASH_CODES:
            raise ValueError("A repetition needs 12 epochs with flash codes 3-14 exactly once.")
        epochs = np.asarray(epochs)
        if len(epochs) != 12:
            raise ValueError("A repetition needs exactly 12 epochs.")
        probs = self.model.predict_proba(epochs)
        return {code: float(p) for code, p in zip(flash_codes, probs)}

    def decode(self, repetitions: list[tuple[np.ndarray, np.ndarray]]) -> dict:
        accumulator = EvidenceAccumulator()
        trace = []
        state, stop, reason = None, False, "NO_DATA"
        for epochs, codes in repetitions:
            probs = self.repetition_probabilities(epochs, codes)
            state = accumulator.update(probs)
            stop, reason = self.stopping.should_stop(state)
            trace.append({
                "flash_order": [int(c) for c in codes],
                "flash_probabilities": {str(k): round(v, 5) for k, v in probs.items()},
                "character": state["character"],
                "confidence": state["confidence"],
                "uncertainty": state["uncertainty"],
                "row_probabilities": state["row_probabilities"],
                "column_probabilities": state["column_probabilities"],
                "stop": stop,
                "reason": reason,
            })
            if stop:
                break
        if state is None:
            raise ValueError("No repetitions supplied.")
        return {"state": state, "stopped": stop, "reason": reason, "trace": trace}


# ==========================================================================
# ARTIFACTS
# ==========================================================================


MANIFEST = "manifest.json"
MODELS = "models.npz"
FORMAT_VERSION = 1


def subject_file(subject: int) -> str:
    return f"subject_{int(subject)}.npz"


def artifacts_present(directory: Path) -> bool:
    directory = Path(directory)
    if not (directory / MANIFEST).is_file() or not (directory / MODELS).is_file():
        return False
    try:
        manifest = json.loads((directory / MANIFEST).read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("format_version") != FORMAT_VERSION:
        return False
    return all((directory / subject_file(s)).is_file() for s in manifest.get("subjects", {}))


@dataclass
class SubjectTrials:
    subject: int
    trial_numbers: np.ndarray     # (n_trials,) original trial index in the recording
    epochs: np.ndarray            # (n_trials, 120, n_channels, n_times) float16 µV
    flash_codes: np.ndarray       # (n_trials, 120)
    labels: np.ndarray            # (n_trials, 120) 1 = target flash

    def __len__(self) -> int:
        return len(self.trial_numbers)

    def repetitions(self, i: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Split trial i into 10 repetitions: the r-th flash of every code, in time order."""
        codes = self.flash_codes[i].astype(int)
        positions = {c: np.flatnonzero(codes == c) for c in config.FLASH_CODES}
        n_reps = min(len(p) for p in positions.values())
        reps = []
        for r in range(n_reps):
            local = np.sort([positions[c][r] for c in config.FLASH_CODES])
            reps.append((self.epochs[i, local].astype(np.float32), codes[local]))
        return reps

    def intended_character(self, i: int) -> str | None:
        targets = set(self.flash_codes[i][self.labels[i] == 1].astype(int).tolist())
        rows = [c for c in targets if c in config.ROW_CODES]
        cols = [c for c in targets if c in config.COLUMN_CODES]
        if len(rows) == 1 and len(cols) == 1:
            return config.pair_to_character(rows[0], cols[0])
        return None


class ArtifactStore:
    """Loads the manifest and models eagerly and subject trials on first use."""

    def __init__(self, directory: Path = config.ARTIFACT_DIR) -> None:
        self.directory = Path(directory)
        if not artifacts_present(self.directory):
            raise FileNotFoundError(
                f"No VOICE artifacts in '{self.directory}'. "
                "Run: python scripts/build_artifacts.py"
            )
        self.manifest = json.loads((self.directory / MANIFEST).read_text())
        with np.load(self.directory / MODELS) as data:
            self.models = {
                int(s): LinearP300Model(
                    weights=data[f"weights_{s}"].astype(np.float64),
                    bias=float(data[f"bias_{s}"]),
                    n_channels=int(self.manifest["n_channels"]),
                    n_times=int(self.manifest["n_times"]),
                )
                for s in self.manifest["subjects"]
            }
        self._trials: dict[int, SubjectTrials] = {}
        self._lock = threading.Lock()

    @property
    def subjects(self) -> list[int]:
        return sorted(self.models)

    @property
    def source(self) -> str:
        return self.manifest["source"]

    def model(self, subject: int) -> LinearP300Model:
        if int(subject) not in self.models:
            raise KeyError(f"Unknown subject {subject}.")
        return self.models[int(subject)]

    def trials(self, subject: int) -> SubjectTrials:
        subject = int(subject)
        self.model(subject)
        with self._lock:
            if subject not in self._trials:
                with np.load(self.directory / subject_file(subject)) as d:
                    self._trials[subject] = SubjectTrials(
                        subject=subject,
                        trial_numbers=d["trial_numbers"].astype(int),
                        epochs=d["epochs"],
                        flash_codes=d["flash_codes"].astype(np.int16),
                        labels=d["labels"].astype(np.int8),
                    )
            return self._trials[subject]


def write_artifacts(
    directory: Path,
    manifest: dict,
    models: dict[int, tuple[np.ndarray, float]],
    trials: dict[int, dict[str, np.ndarray]],
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / MODELS,
        **{f"weights_{s}": w.astype(np.float64) for s, (w, _) in models.items()},
        **{f"bias_{s}": np.float64(b) for s, (_, b) in models.items()},
    )
    for s, d in trials.items():
        np.savez_compressed(
            directory / subject_file(s),
            trial_numbers=d["trial_numbers"].astype(np.int16),
            epochs=d["epochs"].astype(np.float16),
            flash_codes=d["flash_codes"].astype(np.int8),
            labels=d["labels"].astype(np.int8),
        )
    manifest = {**manifest, "format_version": FORMAT_VERSION}
    # Manifest last: its presence marks a complete build.
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=2))


# ==========================================================================
# SESSION
# ==========================================================================


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MessageEngine:
    """Composed message with undo / redo."""

    def __init__(self) -> None:
        self.text = ""
        self._undo: list[str] = []
        self._redo: list[str] = []
        self.history: list[dict] = []

    def _set(self, text: str, action: str, character: str | None = None, source: str = "USER") -> None:
        if text == self.text:
            return
        self._undo.append(self.text)
        self._redo.clear()
        self.text = text
        self.history.append({"time": _now(), "action": action, "character": character,
                             "source": source, "message": text})

    def add(self, character: str, source: str) -> None:
        if character not in config.CHARACTERS:
            raise ValueError(f"'{character}' is not on the speller grid.")
        self._set(self.text + character, "ADD", character, source)

    def apply(self, action: str) -> None:
        if action == "space":
            self._set(self.text + " ", "SPACE", " ")
        elif action == "backspace":
            self._set(self.text[:-1], "BACKSPACE")
        elif action == "clear":
            self._set("", "CLEAR")
        elif action == "undo":
            if self._undo:
                self._redo.append(self.text)
                self.text = self._undo.pop()
                self.history.append({"time": _now(), "action": "UNDO", "message": self.text})
        elif action == "redo":
            if self._redo:
                self._undo.append(self.text)
                self.text = self._redo.pop()
                self.history.append({"time": _now(), "action": "REDO", "message": self.text})
        else:
            raise ValueError(f"Unknown edit action '{action}'.")

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)


class SpellerSession:
    """Replays a subject's held-out recorded trials through the adaptive decoder."""

    def __init__(self, store: ArtifactStore, subject: int) -> None:
        self.id = secrets.token_urlsafe(12)
        self.subject = int(subject)
        self.store = store
        self.trials = store.trials(self.subject)
        self.decoder = TrialDecoder(store.model(self.subject))
        self.message = MessageEngine()
        self.created = _now()
        self.finished_at: str | None = None
        self.lock = threading.Lock()

        self.position = 0                       # next held-out trial to decode
        self.candidate: str | None = None
        self.alternatives: list[tuple[str, float]] = []
        self.rejections = 0
        self.current: dict | None = None        # trial being confirmed
        self.explanation: dict | None = None    # brain response behind the last decoded letter
        self.trial_log: list[dict] = []
        self.events: list[dict] = [{"time": self.created, "event": "SESSION_STARTED"}]

    # -------------------------------------------------------------- state
    @property
    def finished(self) -> bool:
        return self.finished_at is not None

    @property
    def awaiting_confirmation(self) -> bool:
        return self.candidate is not None

    def _require_open(self) -> None:
        if self.finished:
            raise RuntimeError("This session is finished. Start a new one.")

    def _log(self, event: str, **details) -> None:
        self.events.append({"time": _now(), "event": event, **details})

    # ------------------------------------------------------------ actions
    def decode_next(self) -> dict:
        self._require_open()
        if self.awaiting_confirmation:
            raise RuntimeError("Accept or reject the current candidate first.")
        if self.position >= len(self.trials):
            raise RuntimeError("All held-out trials for this subject have been decoded.")

        i = self.position
        self.position += 1
        repetitions = self.trials.repetitions(i)
        result = self.decoder.decode(repetitions)
        state = result["state"]

        self.alternatives = ranked_characters(state, config.MAX_ALTERNATIVES + 1)
        self.candidate = state["character"]
        self.rejections = 0
        self.current = {
            "trial": int(self.trials.trial_numbers[i]),
            "intended": self.trials.intended_character(i),
            "decoded": state["character"],
            "repetitions": state["repetitions"],
            "stopping_reason": result["reason"],
            "confidence": round(state["confidence"], 4),
            "uncertainty": round(state["uncertainty"], 4),
            "selected": None,
            "outcome": None,
        }
        self.explanation = self._explain(repetitions[: state["repetitions"]], state, self.current)
        self.current["p300_uv"] = self.explanation["summary"]
        self._log("TRIAL_DECODED", trial=self.current["trial"], decoded=self.candidate,
                  repetitions=state["repetitions"], reason=result["reason"])
        return {"trial": self.current, "trace": result["trace"]}

    # Electrodes over the back/top of the head, where the P300 is strongest.
    P300_ELECTRODES = ("Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8")
    P300_WINDOW_MS = (250, 500)

    def _explain(self, used: list, state: dict, trial: dict) -> dict:
        """Average brain response per flash code over the rounds the decoder used."""
        channels = list(self.store.manifest.get("channels", []))
        picks = [channels.index(c) for c in self.P300_ELECTRODES if c in channels] or list(range(used[0][0].shape[1]))
        sums = {c: 0.0 for c in config.FLASH_CODES}
        for epochs, codes in used:
            for epoch, code in zip(epochs, codes):
                sums[int(code)] = sums[int(code)] + epoch[picks].astype(np.float64).mean(axis=0)
        per_code = {c: v / len(used) for c, v in sums.items()}

        n_times = len(next(iter(per_code.values())))
        start = int(round(config.EPOCH_START_S * config.SAMPLING_RATE))
        t_ms = (start + np.arange(n_times) * config.DECIMATION) / config.SAMPLING_RATE * 1000
        win = (t_ms >= self.P300_WINDOW_MS[0]) & (t_ms <= self.P300_WINDOW_MS[1])

        row, col = (int(c) for c in state["pair"])
        others = [c for c in config.FLASH_CODES if c not in (row, col)]
        chosen_wave = (per_code[row] + per_code[col]) / 2
        other_wave = np.mean([per_code[c] for c in others], axis=0)
        window = {c: float(per_code[c][win].mean()) for c in config.FLASH_CODES}

        intended_pair = None
        intended = trial.get("intended")
        if intended and intended != state["character"]:
            r = next(k for k, line in enumerate(config.GRID) if intended in line)
            intended_pair = (config.ROW_CODES[r], config.COLUMN_CODES[config.GRID[r].index(intended)])
        intended_wave = ((per_code[intended_pair[0]] + per_code[intended_pair[1]]) / 2
                         if intended_pair else None)
        peak_i = int(np.argmax(np.where(win, chosen_wave, -np.inf)))
        return {
            "t_ms": t_ms.round(1).tolist(),
            "chosen": chosen_wave.round(3).tolist(),
            "others": other_wave.round(3).tolist(),
            "intended": intended_wave.round(3).tolist() if intended_wave is not None else None,
            "pair": [row, col],
            "intended_pair": list(intended_pair) if intended_pair else None,
            "window_uv": {str(k): round(v, 3) for k, v in window.items()},
            "rounds": len(used),
            "electrodes": [channels[i] for i in picks] if channels else [],
            "summary": {
                "chosen_window_uv": round(float(chosen_wave[win].mean()), 3),
                "others_window_uv": round(float(other_wave[win].mean()), 3),
                "peak_ms": round(float(t_ms[peak_i]), 1),
            },
        }

    def accept(self) -> str:
        self._require_open()
        if not self.awaiting_confirmation:
            raise RuntimeError("There is no candidate to accept.")
        character = self.candidate
        self.message.add(character, source="EEG DECODER")
        self._close_trial(character, "ACCEPTED")
        return character

    def reject(self) -> str | None:
        self._require_open()
        if not self.awaiting_confirmation:
            raise RuntimeError("There is no candidate to reject.")
        self._log("CANDIDATE_REJECTED", character=self.candidate)
        self.rejections += 1
        if self.rejections < len(self.alternatives):
            self.candidate = self.alternatives[self.rejections][0]
            return self.candidate
        self._close_trial(None, "SKIPPED")
        return None

    def _close_trial(self, selected: str | None, outcome: str) -> None:
        self.current.update(selected=selected, outcome=outcome, rejections=self.rejections)
        self.trial_log.append(self.current)
        self._log("TRIAL_" + outcome, trial=self.current["trial"], selected=selected)
        self.current, self.candidate, self.alternatives, self.rejections = None, None, [], 0

    def edit(self, action: str, character: str | None = None) -> None:
        self._require_open()
        if action == "add":
            if not character:
                raise ValueError("'add' needs a character.")
            self.message.add(character, source="MANUAL")
        else:
            self.message.apply(action)

    def finish(self) -> dict:
        if self.awaiting_confirmation:
            self._close_trial(None, "UNCONFIRMED")
        if not self.finished:
            self.finished_at = _now()
            self._log("SESSION_FINISHED")
        return self.report()

    # ------------------------------------------------------------ outputs
    def summary(self) -> dict:
        decoded = [t for t in self.trial_log if t["intended"] is not None]
        accepted = [t for t in self.trial_log if t["outcome"] == "ACCEPTED" and t["intended"]]
        return {
            "trials_decoded": len(self.trial_log) + (1 if self.current else 0),
            "trials_remaining": len(self.trials) - self.position,
            "first_guess_accuracy": (
                round(sum(t["decoded"] == t["intended"] for t in decoded) / len(decoded), 4)
                if decoded else None
            ),
            "accepted_accuracy": (
                round(sum(t["selected"] == t["intended"] for t in accepted) / len(accepted), 4)
                if accepted else None
            ),
            "mean_repetitions": (
                round(float(np.mean([t["repetitions"] for t in self.trial_log])), 2)
                if self.trial_log else None
            ),
        }

    def state(self) -> dict:
        return {
            "session_id": self.id,
            "subject": self.subject,
            "finished": self.finished,
            "message": self.message.text,
            "can_undo": self.message.can_undo,
            "can_redo": self.message.can_redo,
            "candidate": self.candidate,
            "alternatives": [{"character": c, "probability": round(p, 4)} for c, p in self.alternatives],
            "rejections": self.rejections,
            "current_trial": self.current,
            "trial_log": self.trial_log,
            "summary": self.summary(),
        }

    def report(self) -> dict:
        return {
            "project": "PROJECT VOICE",
            "session_id": self.id,
            "subject": self.subject,
            "data_source": self.store.source,
            "created": self.created,
            "finished": self.finished_at,
            "message": self.message.text,
            "summary": self.summary(),
            "trials": self.trial_log,
            "events": self.events,
            "message_history": self.message.history,
            "stopping_rule": config.STOPPING,
        }

    def save(self, directory: Path = config.SESSION_DIR) -> Path | None:
        """Best effort: Render's free disk is temporary, so the client also downloads it."""
        try:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"VOICE_{self.id}.json"
            path.write_text(json.dumps(self.report(), indent=2))
            return path
        except OSError:
            return None


# ==========================================================================
# MODEL BUILDING (python app.py --build)
# ==========================================================================

CHANNELS = ["Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8"]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Data sources: both return {subject: (epochs_uv (4200, 8, 65) float32,
#                                      labels (4200,), flash_codes (4200,))}
# ---------------------------------------------------------------------------

def peak_memory_mb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:  # pragma: no cover - not available on Windows
        return float("nan")


DOWNLOAD_SOURCE = {"value": None}

BNCI_URLS = [u.strip().rstrip("/") + "/" for u in os.environ.get(
    "VOICE_BNCI_URLS",
    "https://lampx.tugraz.at/~bci/database/,http://bnci-horizon-2020.eu/database/data-sets/",
).split(",") if u.strip()]


def download_recording(subject: int, data_dir: Path) -> Path:
    """Download A0<s>.mat of BNCI 008-2014 (same files MOABB uses), with retries."""
    import shutil
    import urllib.request

    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / f"A{subject:02d}.mat"
    if target.is_file() and target.stat().st_size > 1_000_000:
        return target
    errors = []
    for base in BNCI_URLS:
        url = f"{base}008-2014/A{subject:02d}.mat"
        for attempt in range(1, 4):
            try:
                log(f"Subject {subject}: downloading {url} (attempt {attempt})")
                part = target.with_suffix(".part")
                request = urllib.request.Request(url, headers={"User-Agent": "VOICE/1.0"})
                with urllib.request.urlopen(request, timeout=120) as response, open(part, "wb") as f:
                    shutil.copyfileobj(response, f, length=1 << 20)
                part.replace(target)
                if BNCI_URLS[0] != base:          # try the working mirror first next time
                    BNCI_URLS.remove(base)
                    BNCI_URLS.insert(0, base)
                return target
            except Exception as error:  # network errors differ by platform
                errors.append(f"{url}: {error}")
                time.sleep(3 * attempt)
    sys.exit(f"Subject {subject}: could not download the recording.\n" + "\n".join(errors[-4:]))


MOABB_FETCH = r"""
import json, sys
import mne
mne.set_log_level("WARNING")
from moabb.datasets import BNCI2014_008
dataset = BNCI2014_008()
paths = {}
for subject in json.loads(sys.argv[1]):
    print(f"MOABB: fetching BNCI2014-008 subject {subject}", flush=True)
    paths[subject] = str(dataset.data_path(subject)[0])
print("VOICE_PATHS=" + json.dumps(paths), flush=True)
"""


def fetch_with_moabb(subjects: list[int], folder: Path) -> dict[int, Path]:
    """Download the recordings with MOABB (BNCI2014_008().data_path).

    Runs in a separate process: importing MOABB takes ~250 MB, which is freed
    before the recordings are processed, so the build stays within 512 MB.
    Returns {} if MOABB is not installed or the download fails.
    """
    folder.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MNE_DATA": str(folder.resolve()),
           "MNE_DATASETS_BNCI_PATH": str(folder.resolve())}
    log("Downloading BNCI2014-008 with MOABB...")
    try:
        result = subprocess.run([sys.executable, "-c", MOABB_FETCH, json.dumps(subjects)],
                                env=env, capture_output=True, text=True, timeout=3600)
    except (OSError, subprocess.TimeoutExpired) as error:
        log(f"MOABB download did not finish ({error}); using direct download instead.")
        return {}
    for line in result.stdout.splitlines():
        if line.startswith("MOABB:"):
            log(line)
    marker = [l for l in result.stdout.splitlines() if l.startswith("VOICE_PATHS=")]
    if result.returncode != 0 or not marker:
        reason = (result.stderr.strip().splitlines() or ["unknown error"])[-1]
        log(f"MOABB download failed ({reason}); using direct download instead.")
        return {}
    paths = {int(k): Path(v) for k, v in json.loads(marker[-1].split("=", 1)[1]).items()}
    missing = [s for s, p in paths.items() if not p.is_file()]
    if missing:
        log(f"MOABB returned missing files for subjects {missing}; downloading those directly.")
    log("MOABB download complete.")
    return {s: p for s, p in paths.items() if p.is_file()}


def run_to_epochs(run) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One BNCI 008-2014 recording struct -> (epochs uV (n, 8, 65), labels, flash codes 3-14).

    Same steps as MOABB's reader: EEG in microvolts at 256 Hz, target channel y,
    flash codes y_stim shifted +2. Then a 0.5-10 Hz zero-phase band-pass and
    -0.2..0.8 s epochs at each flash onset, decimated to 64 Hz.
    """
    from scipy.signal import butter, sosfiltfilt

    eeg_uv = np.asarray(run.X, dtype=np.float64).T              # (8, n_samples), microvolts
    target_channel = np.asarray(run.y).astype(int).ravel()
    flash_channel = np.asarray(run.y_stim).astype(int).ravel()
    classes = [str(c).strip().lower() for c in np.atleast_1d(getattr(run, "classes", ["nontarget", "target"]))]
    if eeg_uv.shape[0] != len(CHANNELS):
        raise ValueError(f"expected {len(CHANNELS)} EEG channels, found {eeg_uv.shape[0]}.")
    if "target" not in classes:
        raise ValueError(f"unexpected class names {classes}.")
    target_value = classes.index("target") + 1

    previous = np.r_[0, flash_channel[:-1]]
    onsets = np.flatnonzero((flash_channel != 0) & (previous == 0))
    if len(onsets) == 0 or len(onsets) % config.EVENTS_PER_TRIAL:
        raise ValueError(f"found {len(onsets)} flashes; expected a multiple of {config.EVENTS_PER_TRIAL}.")
    codes = flash_channel[onsets] + 2                            # 1-12 -> 3-14, as MOABB
    labels = (target_channel[onsets] == target_value).astype(np.int8)

    fs = config.SAMPLING_RATE
    sos = butter(4, [config.FILTER_LOW_HZ, config.FILTER_HIGH_HZ], btype="bandpass", fs=fs, output="sos")
    first = int(round(config.EPOCH_START_S * fs))
    last = int(round(config.EPOCH_END_S * fs))
    offsets = np.arange(first, last + 1)[:: config.DECIMATION]      # 65 samples
    pad = max(0, -first, last)
    index = onsets[:, None] + offsets[None, :] + pad
    epochs = np.empty((len(onsets), eeg_uv.shape[0], len(offsets)), np.float32)
    for ch in range(eeg_uv.shape[0]):           # one electrode at a time keeps memory low
        filtered = np.pad(sosfiltfilt(sos, eeg_uv[ch]), (pad, pad))
        epochs[:, ch, :] = filtered[index]
    del eeg_uv
    np.clip(epochs, -60000, 60000, out=epochs)
    return epochs, labels, codes.astype(np.int8)


def load_bnci(subjects: list[int], data_dir: str | None) -> dict[int, tuple]:
    """Load BNCI2014-008 with NumPy/SciPy only, one recording at a time.

    Files come from MOABB (BNCI2014_008().data_path), or a direct download of the
    same files if MOABB is unavailable. Reads them as MOABB does: EEG in microvolts at 256 Hz,
    target channel y, flash codes y_stim shifted +2 to 3-14), band-passes
    0.5-10 Hz (4th-order Butterworth, zero phase), and cuts -0.2..0.8 s epochs
    at each flash onset, decimated to 64 Hz. Peak memory stays far below 512 MB.
    """
    import gc
    from scipy.io import loadmat

    folder = Path(data_dir or os.environ.get("VOICE_DATA_DIR", Path.home() / "voice_data")).expanduser()
    expected = config.TRIALS_PER_SUBJECT * config.EVENTS_PER_TRIAL

    moabb_paths = {} if os.environ.get("VOICE_SKIP_MOABB") == "1" else fetch_with_moabb(subjects, folder)
    DOWNLOAD_SOURCE["value"] = "MOABB" if len(moabb_paths) == len(subjects) else (
        "MOABB + direct" if moabb_paths else "direct")

    out = {}
    for s in subjects:
        path = moabb_paths.get(s) or download_recording(s, folder)
        try:
            run = loadmat(path, struct_as_record=False, squeeze_me=True)["data"]
        except Exception as error:
            path.unlink(missing_ok=True)  # corrupt download: fetch again next time
            sys.exit(f"Subject {s}: could not read {path.name} ({error}). Run the build again.")

        try:
            epochs, labels, codes = run_to_epochs(run)
        except ValueError as error:
            sys.exit(f"Subject {s}: {error}")
        del run
        if len(codes) != expected:
            sys.exit(f"Subject {s}: expected {expected} flashes, found {len(codes)}.")
        out[s] = (epochs, labels, codes)
        gc.collect()
        log(f"Subject {s}: {expected} epochs ready | peak memory so far {peak_memory_mb():.0f} MB")
    return out


def preprocess(X_volts: np.ndarray) -> np.ndarray:
    x = X_volts[:, :, :: config.DECIMATION] * config.SCALE_TO_MICROVOLTS
    return np.clip(x, -60000, 60000).astype(np.float32)  # float16-safe range


def load_synthetic(subjects: list[int], seed: int = 7) -> dict[int, tuple]:
    """Simulated P300 data with the real recording's structure. For tests only."""
    rng = np.random.default_rng(seed)
    n_times = int(round((config.EPOCH_END_S - config.EPOCH_START_S) * config.SAMPLING_RATE)) + 1
    n_times = len(range(0, n_times, config.DECIMATION))
    t = np.linspace(config.EPOCH_START_S, config.EPOCH_END_S, n_times)
    erp = 2.5 * np.exp(-((t - 0.35) ** 2) / (2 * 0.06 ** 2))
    gain = np.array([0.6, 1.0, 1.2, 0.7, 0.9, 0.9, 0.8, 0.8])[:, None]
    out = {}
    for s in subjects:
        snr = rng.uniform(0.35, 0.8)
        codes, labels = [], []
        for _ in range(config.TRIALS_PER_SUBJECT):
            target = {int(rng.choice(config.ROW_CODES)), int(rng.choice(config.COLUMN_CODES))}
            for _ in range(config.REPETITIONS_PER_TRIAL):
                order = rng.permutation(config.FLASH_CODES)
                codes.extend(order)
                labels.extend(int(c in target) for c in order)
        codes, labels = np.array(codes, dtype=np.int8), np.array(labels, dtype=np.int8)
        x = rng.normal(0, 4.0, size=(len(codes), len(CHANNELS), n_times))
        x[labels == 1] += 0.6 * snr * gain * erp
        out[s] = (x.astype(np.float32), labels, codes)
    return out


# ---------------------------------------------------------------------------
# Validation, training, evaluation
# ---------------------------------------------------------------------------

def validate_structure(s: int, labels: np.ndarray, codes: np.ndarray) -> float:
    """Check 35 trials x 12 codes x 10 flashes; return share of trials whose two
    target codes are one row code and one column code."""
    n = config.EVENTS_PER_TRIAL
    mapped = 0
    for trial in range(config.TRIALS_PER_SUBJECT):
        c, y = codes[trial * n:(trial + 1) * n], labels[trial * n:(trial + 1) * n]
        counts = {k: int((c == k).sum()) for k in config.FLASH_CODES}
        if set(np.unique(c)) != set(config.FLASH_CODES) or set(counts.values()) != {config.REPETITIONS_PER_TRIAL}:
            sys.exit(f"Subject {s}, trial {trial}: unexpected flash structure {counts}.")
        targets = set(c[y == 1].tolist())
        if len(targets) != 2:
            sys.exit(f"Subject {s}, trial {trial}: expected 2 target codes, found {sorted(targets)}.")
        rows = targets & set(config.ROW_CODES)
        mapped += len(rows) == 1
    return mapped / config.TRIALS_PER_SUBJECT


def _class_shrunk_covariance(chunks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, int]:
    """Ledoit-Wolf shrunk covariance of one class, accumulated chunk by chunk.

    Same maths as scikit-learn's LDA(solver="lsqr", shrinkage="auto"): standardize
    the class, apply ledoit_wolf, scale back. Only 520 x 520 sums stay in memory.
    """
    n = sum(len(c) for c in chunks)
    total = sum(c.sum(axis=0, dtype=np.float64) for c in chunks)
    mean = total / n
    var = sum(((c.astype(np.float64) - mean) ** 2).sum(axis=0) for c in chunks) / n
    scale = np.sqrt(var)
    scale[scale == 0] = 1.0

    p = len(mean)
    gram = np.zeros((p, p))
    gram_sq = np.zeros((p, p))
    for c in chunks:
        z = (c.astype(np.float64) - mean) / scale
        gram += z.T @ z
        z2 = z * z
        gram_sq += z2.T @ z2

    emp_cov = gram / n
    trace = np.diag(emp_cov).sum()
    mu = trace / p
    delta_ = np.sum(gram ** 2) / n ** 2
    beta = (np.sum(gram_sq) / n - delta_) / (p * n)
    delta = (delta_ - 2.0 * mu * trace + p * mu ** 2) / p
    beta = min(beta, delta)
    shrinkage = 0.0 if beta == 0 else beta / delta
    shrunk = (1.0 - shrinkage) * emp_cov
    shrunk.flat[:: p + 1] += shrinkage * mu
    return mean, scale[:, None] * shrunk * scale[None, :], n


def fit_shrinkage_lda(chunks: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, float]:
    """Binary shrinkage LDA from (features, labels) chunks; returns weights, bias
    such that p(target) = sigmoid(features @ weights + bias)."""
    by_class = {k: [x[y == k] for x, y in chunks if np.any(y == k)] for k in (0, 1)}
    stats = {k: _class_shrunk_covariance(by_class[k]) for k in (0, 1)}
    n_total = stats[0][2] + stats[1][2]
    priors = {k: stats[k][2] / n_total for k in (0, 1)}
    covariance = priors[0] * stats[0][1] + priors[1] * stats[1][1]
    means = np.vstack([stats[0][0], stats[1][0]])
    coef = np.linalg.lstsq(covariance, means.T, rcond=None)[0].T
    intercept = -0.5 * np.einsum("ij,ij->i", means, coef) + np.log([priors[0], priors[1]])
    return coef[1] - coef[0], float(intercept[1] - intercept[0])


def check_against_sklearn(X: np.ndarray, y: np.ndarray) -> None:
    """The streaming LDA must reproduce scikit-learn on a sample."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    feats = X.reshape(len(X), -1).astype(np.float64)
    weights, bias = fit_shrinkage_lda([(feats[: len(feats) // 2], y[: len(y) // 2]),
                                       (feats[len(feats) // 2:], y[len(y) // 2:])])
    reference = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(feats, y)
    ours = LinearP300Model(weights, bias, X.shape[1], X.shape[2]).predict_proba(X)
    if not np.allclose(ours, reference.predict_proba(feats)[:, 1], atol=1e-6):
        raise RuntimeError("Streaming LDA does not match scikit-learn.")
    log("Streaming LDA matches scikit-learn on a 3,000-epoch sample.")


def evaluate(model: LinearP300Model, trials: SubjectTrials) -> dict:
    adaptive, fixed, reps = [], [], []
    always_ten = TrialDecoder(model)
    always_ten.stopping.minimum_repetitions = config.REPETITIONS_PER_TRIAL
    for i in range(len(trials)):
        intended = trials.intended_character(i)
        if intended is None:
            continue
        repetitions = trials.repetitions(i)
        a = TrialDecoder(model).decode(repetitions)["state"]
        f = always_ten.decode(repetitions)["state"]
        adaptive.append(a["character"] == intended)
        fixed.append(f["character"] == intended)
        reps.append(a["repetitions"])
    return {
        "scored_trials": len(adaptive),
        "adaptive_accuracy": round(float(np.mean(adaptive)), 4) if adaptive else None,
        "fixed_10_repetition_accuracy": round(float(np.mean(fixed)), 4) if fixed else None,
        "mean_repetitions": round(float(np.mean(reps)), 2) if reps else None,
    }


def build(source: str, out_dir: Path, subjects: list[int], calibration_trials: int,
          data_dir: str | None) -> dict:
    started = time.time()
    data = load_synthetic(subjects) if source == "synthetic" else load_bnci(subjects, data_dir)
    n = config.EVENTS_PER_TRIAL
    split = calibration_trials * n

    mapping = {s: validate_structure(s, d[1], d[2]) for s, d in data.items()}
    if min(mapping.values()) < 0.95:
        sys.exit("Target codes do not split into one row code (3-8) and one column code (9-14): "
                 f"{mapping}. Check ROW_CODES / COLUMN_CODES in voice/config.py.")

    if os.environ.get("VOICE_SKLEARN_CHECK") == "1":
        first = data[subjects[0]]
        check_against_sklearn(first[0][:3000], first[1][:3000])

    models, trial_files, evaluation = {}, {}, {}
    n_channels, n_times = next(iter(data.values()))[0].shape[1:]
    for s in subjects:
        others = [o for o in subjects if o != s]
        chunks = [(data[o][0].reshape(len(data[o][0]), -1), data[o][1]) for o in others]
        chunks.append((data[s][0][:split].reshape(split, -1), data[s][1][:split]))
        log(f"Subject {s}: training on {sum(len(c[1]) for c in chunks)} epochs "
            f"({len(others)} other subjects + {calibration_trials} calibration trials)")
        weights, bias = fit_shrinkage_lda(chunks)
        models[s] = (weights, bias)

        X_s, y_s, c_s = data[s]
        n_heldout = config.TRIALS_PER_SUBJECT - calibration_trials
        trial_files[s] = {
            "trial_numbers": np.arange(calibration_trials, config.TRIALS_PER_SUBJECT),
            "epochs": X_s[split:].reshape(n_heldout, n, n_channels, n_times),
            "flash_codes": c_s[split:].reshape(n_heldout, n),
            "labels": y_s[split:].reshape(n_heldout, n),
        }
        held_out = SubjectTrials(s, trial_files[s]["trial_numbers"],
                                 trial_files[s]["epochs"].astype(np.float16),  # evaluate what is served
                                 trial_files[s]["flash_codes"], trial_files[s]["labels"])
        evaluation[s] = evaluate(LinearP300Model(weights, bias, n_channels, n_times), held_out)
        log(f"Subject {s}: held-out accuracy {evaluation[s]['adaptive_accuracy']:.0%} "
            f"(adaptive, {evaluation[s]['mean_repetitions']} reps) | "
            f"{evaluation[s]['fixed_10_repetition_accuracy']:.0%} (10 reps)")

    manifest = {
        "source": "BNCI2014-008" if source == "bnci" else "synthetic",
        "synthetic": source == "synthetic",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "channels": CHANNELS,
        "n_channels": int(n_channels),
        "n_times": int(n_times),
        "units": "microvolts",
        "preprocessing": {
            "bandpass_hz": [config.FILTER_LOW_HZ, config.FILTER_HIGH_HZ],
            "epoch_s": [config.EPOCH_START_S, config.EPOCH_END_S],
            "sampling_rate_hz": config.SAMPLING_RATE / config.DECIMATION,
        },
        "download": DOWNLOAD_SOURCE["value"] if source == "bnci" else None,
        "classifier": "LDA with Ledoit-Wolf shrinkage (scikit-learn lsqr/auto equivalent), weights + bias",
        "calibration_trials": calibration_trials,
        "stopping": config.STOPPING,
        "subjects": {
            str(s): {"held_out_trials": int(len(trial_files[s]["trial_numbers"])),
                     "row_column_mapping_ok": mapping[s], **evaluation[s]}
            for s in subjects
        },
    }
    write_artifacts(out_dir, manifest, models, trial_files)
    log(f"Artifacts written to {out_dir} in {time.time() - started:.0f}s "
        f"(peak memory {peak_memory_mb():.0f} MB)")
    return manifest

def build_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="app.py --build")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--source", choices=["bnci", "synthetic"],
                        default=os.environ.get("VOICE_BUILD_SOURCE", "bnci"))
    parser.add_argument("--out", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--data-dir", default=os.environ.get("VOICE_DATA_DIR"),
                        help="Where the recordings are downloaded (default: ~/voice_data)")
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBJECTS)
    parser.add_argument("--calibration-trials", type=int, default=CALIBRATION_TRIALS)
    parser.add_argument("--force", action="store_true", help="Rebuild even if files exist")
    args, _ = parser.parse_known_args(argv)

    if artifacts_present(args.out) and not args.force:
        log(f"Model files already present in {args.out}; nothing to build (use --force to rebuild).")
        return
    if not 1 <= args.calibration_trials < TRIALS_PER_SUBJECT:
        sys.exit("--calibration-trials must be between 1 and 34.")
    if len(set(args.subjects)) < 2:
        sys.exit("Need at least 2 subjects (each decoder also trains on the others).")
    build(args.source, args.out, sorted(set(args.subjects)), args.calibration_trials, args.data_dir)


if __name__ == "__main__" and "--build" in sys.argv[1:]:
    build_main(sys.argv[1:])
    sys.exit(0)


# ==========================================================================
# WEB APP (Gradio)
# ==========================================================================

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402

STEP_DELAY_S = float(os.environ.get("VOICE_STEP_DELAY_S", "0.35"))  # pause between repetitions


class ModelFiles:
    """Loads the model files, building them in a background process if missing,
    so the web server can open its port straight away."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.store: ArtifactStore | None = None
        self.status = "Checking model files..."
        self.error: str | None = None
        self.log_path = Path(tempfile.gettempdir()) / "voice_build.log"
        self._lock = threading.Lock()

    def ensure(self) -> None:
        if artifacts_present(self.directory):
            self._load()
            return
        self.status = ("Preparing the model: downloading BNCI2014-008 and training 8 decoders. "
                       "This takes about 5-10 minutes the first time.")
        threading.Thread(target=self._build, daemon=True).start()

    def _build(self) -> None:
        with open(self.log_path, "w") as log_file:
            code = subprocess.call([sys.executable, os.path.abspath(__file__), "--build",
                                    "--out", str(self.directory)], stdout=log_file, stderr=subprocess.STDOUT)
        if code == 0 and artifacts_present(self.directory):
            self._load()
        else:
            tail = self.log_path.read_text()[-1500:] if self.log_path.exists() else ""
            self.error = f"Building the model files failed (exit {code}).\n\n{tail}"
            self.status = "Model build failed."

    def _load(self) -> None:
        with self._lock:
            self.store = ArtifactStore(self.directory)
            self.status = "ready"

    def progress(self) -> str:
        if not self.log_path.exists():
            return ""
        lines = [l for l in self.log_path.read_text().splitlines() if l.strip()]
        return lines[-1] if lines else ""


MODEL = ModelFiles(Path(os.environ.get("VOICE_ARTIFACT_DIR", ARTIFACT_DIR)))

CSS = """
.gradio-container { max-width: 1240px !important; margin: 0 auto; }
#voice-title h1 { letter-spacing: .14em; margin-bottom: 0; }
.voice-badge { display:inline-block; font-size:12px; font-weight:600; padding:4px 10px; border-radius:999px;
  background: var(--color-accent-soft); color: var(--color-accent); }
.voice-badge.synthetic { background:#fff1d6; color:#8a5300; }
.grid-btn button, button.grid-btn { font: 700 24px/1 ui-monospace, Menlo, monospace !important; min-height: 58px !important; }
.voice-candidate { display:flex; gap:18px; align-items:center; }
.voice-letter { width:92px; height:92px; flex:none; border-radius:16px; display:grid; place-items:center;
  font: 800 50px/1 ui-monospace, Menlo, monospace; background: var(--color-accent-soft); color: var(--color-accent); }
.voice-metrics { display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 6px 18px; flex:1; }
.voice-metrics dt { font-size:12px; opacity:.7; } .voice-metrics dd { margin:0; font-size:18px; font-weight:700; }
.voice-note { font-size:13px; opacity:.75; margin-top:8px; }
.voice-bars { display:grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.voice-bar { display:grid; grid-template-columns: 24px 1fr 40px; gap:6px; align-items:center; font-size:12px; margin:3px 0; }
.voice-track { height:10px; border-radius:5px; background: var(--neutral-200); overflow:hidden; }
.voice-fill { height:100%; background: var(--color-accent); border-radius:5px; }
.voice-fill.top { background: #1f8a4c; }
#voice-message textarea { font: 600 22px/1.3 ui-monospace, Menlo, monospace !important; letter-spacing:.06em; }
.erp-legend { display:flex; flex-wrap:wrap; gap: 6px 18px; font-size: 13px; margin: 4px 0 6px; }
.erp-key { display:inline-flex; align-items:center; gap:6px; }
.erp-swatch { width:18px; height:0; border-top:3px solid; display:inline-block; }
.erp-svg { width:100%; height:auto; max-width:760px; display:block; }
.erp-line { fill:none; stroke-width:2; stroke-linejoin:round; stroke-linecap:round; }
.erp-chosen { stroke:#2a78d6; border-color:#2a78d6; }
.erp-intended { stroke:#eb6834; border-color:#eb6834; stroke-dasharray:6 4; }
.erp-other { stroke:#8a8985; border-color:#8a8985; }
.dark .erp-chosen { stroke:#3987e5; border-color:#3987e5; }
.dark .erp-intended { stroke:#d95926; border-color:#d95926; }
.erp-peak { fill:#2a78d6; stroke: var(--background-fill-primary, #fff); stroke-width:2; }
.dark .erp-peak { fill:#3987e5; }
.erp-window { fill: currentColor; opacity:.06; }
.erp-grid { stroke: currentColor; opacity:.12; }
.erp-flash { stroke: currentColor; opacity:.45; stroke-dasharray:3 3; }
.erp-tick, .erp-note { fill: currentColor; opacity:.7; font-size:12px; }
.erp-label { fill: currentColor; font-size:12px; font-weight:600; }
.erp-hit { fill: transparent; }
.erp-hit:hover { fill: currentColor; opacity:.06; }
.erp p { margin: 8px 0; line-height:1.5; }
.erp-ok { font-weight:600; color:#1f8a4c; }
.erp-bad { font-weight:600; color:#c0392b; }
.erp-small, .erp-empty { font-size:13px; opacity:.75; }
"""

REASONS = {
    "ADAPTIVE_CONFIDENCE_STOP": "Stopped early: confident and stable.",
    "MAXIMUM_REPETITIONS": "Used all 10 repetitions.",
}


# ---------------------------------------------------------------- render
def pct(v) -> str:
    return "–" if v is None else f"{round(v * 100)}%"


def decoder_html(sess: SpellerSession | None, step: dict | None = None, note: str = "") -> str:
    letter = "–"
    t = sess.current if sess else None
    last = (sess.trial_log[-1] if sess and sess.trial_log else None)
    shown = t or last
    if step:
        letter = step["character"]
    elif sess and sess.candidate:
        letter = sess.candidate
    conf = step["confidence"] if step else (shown or {}).get("confidence")
    unc = step["uncertainty"] if step else (shown or {}).get("uncertainty")
    reps = (len(step.get("_so_far", [])) or None) if step else (shown or {}).get("repetitions")
    intended = (shown or {}).get("intended") if not step else (t or {}).get("intended")
    if not note and sess and sess.candidate and t:
        note = (f"Rejection {sess.rejections}: showing the next most likely character."
                if sess.rejections else f"Trial {t['trial']}: {REASONS.get(t['stopping_reason'], t['stopping_reason'])}")
    return f"""<div class="voice-candidate"><div class="voice-letter">{html.escape(letter)}</div>
      <dl class="voice-metrics">
        <div><dt>Confidence</dt><dd>{pct(conf)}</dd></div>
        <div><dt>Uncertainty</dt><dd>{pct(unc)}</dd></div>
        <div><dt>Repetitions</dt><dd>{reps if reps is not None else "–"}</dd></div>
        <div><dt>Intended</dt><dd>{html.escape(str(intended)) if intended else "–"}</dd></div>
      </dl></div><div class="voice-note">{html.escape(note)}</div>"""


def evidence_html(step: dict | None) -> str:
    def bars(probs, prefix):
        top = max(probs) if probs else None
        out = []
        for i in range(6):
            p = probs[i] if probs else 0.0
            cls = " top" if probs and p == top else ""
            out.append(f'<div class="voice-bar"><span>{prefix}{i + 1}</span><div class="voice-track">'
                       f'<div class="voice-fill{cls}" style="width:{p * 100:.1f}%"></div></div>'
                       f'<span>{pct(p) if probs else ""}</span></div>')
        return "".join(out)
    rows = step["row_probabilities"] if step else None
    cols = step["column_probabilities"] if step else None
    return (f'<div class="voice-bars"><div><b>Row evidence</b>{bars(rows, "R")}</div>'
            f'<div><b>Column evidence</b>{bars(cols, "C")}</div></div>')


def history_frame(sess: SpellerSession | None) -> pd.DataFrame:
    columns = ["Trial", "Intended", "Decoded", "Typed", "Reps", "Confidence"]
    if not sess or not sess.trial_log:
        return pd.DataFrame(columns=columns)
    rows = [[t["trial"], t["intended"] or "?", t["decoded"],
             t["selected"] or ("skipped" if t["outcome"] == "SKIPPED" else "–"),
             t["repetitions"], pct(t["confidence"])] for t in reversed(sess.trial_log)]
    return pd.DataFrame(rows, columns=columns)


def stats_md(sess: SpellerSession | None) -> str:
    if not sess:
        return "No session yet."
    s = sess.summary()
    return (f"**Trials:** {len(sess.trial_log)} · **First-guess accuracy:** {pct(s['first_guess_accuracy'])} · "
            f"**Mean repetitions:** {s['mean_repetitions'] or '–'} · **Trials left:** {s['trials_remaining']}")


def view(sess: SpellerSession | None, step: dict | None = None, note: str = "",
         busy: bool = False, report_path: str | None = None, reset: bool = False) -> list:
    active = bool(sess) and not sess.finished
    awaiting = active and sess.candidate is not None
    highlight = step["character"] if step else (sess.candidate if sess else None)
    can_type = active and not awaiting and not busy
    grid_updates = [
        gr.update(variant="primary" if ch == highlight else "secondary", interactive=can_type)
        for ch in config.CHARACTERS
    ]
    remaining = sess.summary()["trials_remaining"] if sess else 0
    return [
        sess,
        decoder_html(sess, step, note),
        evidence_html(step) if step or not sess or reset else gr.update(),
        sess.message.text if sess else "",
        history_frame(sess),
        stats_md(sess),
        gr.update(interactive=active and not awaiting and not busy and remaining > 0),
        gr.update(interactive=awaiting and not busy),
        gr.update(interactive=awaiting and not busy),
        gr.update(interactive=active and not busy),
        gr.update(value=report_path, visible=report_path is not None),
        explanation_html(sess) if not busy else gr.update(),
        *grid_updates,
    ]


# --------------------------------------------------------------- actions
def guard(fn):
    """Show problems as a toast instead of breaking the page."""
    def wrapped(sess, *args):
        try:
            return fn(sess, *args)
        except (RuntimeError, ValueError, KeyError) as error:
            gr.Warning(str(error))
            return view(sess)
    wrapped.__name__ = fn.__name__
    return wrapped


def start(_sess, subject):
    if MODEL.store is None:
        gr.Warning("The model is still being prepared. Try again in a minute.")
        return view(None)
    sess = SpellerSession(MODEL.store, int(subject))
    return view(sess, note="Session started. Click Decode next trial.", reset=True)


def decode(sess, animate):
    if sess is None:
        gr.Warning("Start a session first.")
        yield view(sess)
        return
    try:
        result = sess.decode_next()
    except RuntimeError as error:
        gr.Warning(str(error))
        yield view(sess)
        return
    so_far = []
    if animate:
        for i, step in enumerate(result["trace"]):
            so_far.append(step)
            shown = {**step, "_so_far": so_far}
            yield view(sess, shown, f"Repetition {i + 1}: current best {step['character']}", busy=True)
            time.sleep(STEP_DELAY_S)
    final = {**result["trace"][-1], "_so_far": result["trace"]}
    out = view(sess, final)
    out[1] = decoder_html(sess)  # settle on the confirmed candidate text
    yield out


@guard
def accept(sess):
    ch = sess.accept()
    return view(sess, note=f"Typed {ch}. Decode the next trial.")


@guard
def reject(sess):
    nxt = sess.reject()
    return view(sess, note="" if nxt else "No more alternatives; trial skipped.")


def editor(action):
    @guard
    def run(sess):
        if sess is None:
            raise RuntimeError("Start a session first.")
        sess.edit(action)
        return view(sess)
    return run


def typer(character):
    @guard
    def run(sess):
        if sess is None:
            raise RuntimeError("Start a session first.")
        sess.edit("add", character)
        return view(sess, note=f"Typed {character} manually.")
    return run


@guard
def finish(sess):
    report = sess.finish()
    sess.save()
    path = os.path.join(tempfile.mkdtemp(prefix="voice_"), f"VOICE_session_{sess.id}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return view(sess, note=f'Session finished. Message: "{report["message"] or "(empty)"}"', report_path=path)


def subject_note(subject) -> str:
    if MODEL.store is None:
        return ""
    meta = MODEL.store.manifest["subjects"].get(str(subject), {})
    if not meta:
        return ""
    return (f"{meta['held_out_trials']} held-out trials. Build-time accuracy: "
            f"**{pct(meta['adaptive_accuracy'])}** adaptive ({meta['mean_repetitions']} repetitions on average), "
            f"**{pct(meta['fixed_10_repetition_accuracy'])}** with all 10 repetitions.")



# ---------------------------------------------------------- EEG files
META_COLUMNS = ["letter_index", "round", "flash_code", "is_target"]


def feature_columns(channels: list[str], n_times: int) -> list[str]:
    return [f"{ch}_{i:02d}" for ch in channels for i in range(n_times)]


def group_letters(epochs: np.ndarray, codes: np.ndarray, labels: np.ndarray | None):
    """Flat flashes -> per-letter arrays. Each letter = 12 flash codes x N rounds."""
    per = len(config.FLASH_CODES)
    n = len(codes)
    if n == 0 or n % per:
        raise ValueError(f"The file has {n} flashes; it needs a multiple of {per} (12 flashes per round).")
    flashes = config.EVENTS_PER_TRIAL if n % config.EVENTS_PER_TRIAL == 0 else n
    L = n // flashes
    ep = epochs.reshape(L, flashes, *epochs.shape[1:])
    cd = codes.reshape(L, flashes)
    lb = labels.reshape(L, flashes) if labels is not None else None
    for i in range(L):
        counts = {c: int((cd[i] == c).sum()) for c in config.FLASH_CODES}
        if set(np.unique(cd[i]).tolist()) - set(config.FLASH_CODES) or len(set(counts.values())) != 1:
            raise ValueError(f"Letter {i + 1}: every flash code 3-14 must appear equally often, found {counts}.")
    return ep, cd, lb


def read_eeg_file(path: str) -> dict:
    """Read a VOICE EEG file (.npz / .csv) or an original BNCI 008-2014 recording (.mat)."""
    store = MODEL.store
    shape = (store.manifest["n_channels"], store.manifest["n_times"])
    suffix = Path(path).suffix.lower()
    meta: dict = {}
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as d:
            if "epochs" not in d or "flash_codes" not in d:
                raise ValueError("The .npz file needs 'epochs' and 'flash_codes' arrays.")
            epochs = d["epochs"].astype(np.float32).reshape(-1, *d["epochs"].shape[-2:])
            codes = d["flash_codes"].astype(int).ravel()
            labels = d["labels"].astype(np.int8).ravel() if "labels" in d else None
            if "meta_json" in d:
                meta = json.loads(str(d["meta_json"]))
        kind = "VOICE EEG file (.npz)"
    elif suffix == ".csv":
        frame = pd.read_csv(path)
        if "flash_code" not in frame.columns:
            raise ValueError("The CSV needs a 'flash_code' column plus 520 EEG value columns.")
        feats = [c for c in frame.columns if c not in META_COLUMNS]
        if len(feats) != shape[0] * shape[1]:
            raise ValueError(f"The CSV has {len(feats)} EEG value columns; expected {shape[0] * shape[1]} "
                             f"({shape[0]} electrodes x {shape[1]} time points).")
        epochs = frame[feats].to_numpy(np.float32).reshape(-1, *shape)
        codes = frame["flash_code"].to_numpy(int)
        labels = (frame["is_target"].to_numpy(np.int8)
                  if "is_target" in frame.columns and frame["is_target"].notna().all() else None)
        kind = "VOICE EEG file (.csv)"
    elif suffix == ".mat":
        from scipy.io import loadmat
        try:
            run = loadmat(path, struct_as_record=False, squeeze_me=True)["data"]
        except Exception as error:
            raise ValueError(f"Could not read this .mat file as a BNCI 008-2014 recording ({error}).") from None
        epochs, labels, codes = run_to_epochs(run)
        kind = "BNCI 008-2014 recording (.mat)"
    else:
        raise ValueError("Use a .npz or .csv VOICE EEG file, or a BNCI 008-2014 .mat recording.")

    if epochs.shape[1:] != shape:
        raise ValueError(f"Each flash must be {shape[0]} electrodes x {shape[1]} time points; got {epochs.shape[1:]}.")
    if not np.all(np.isfinite(epochs)):
        raise ValueError("The file contains missing or infinite EEG values.")
    ep, cd, lb = group_letters(epochs, codes, labels)
    return {"epochs": ep, "codes": cd, "labels": lb, "meta": meta, "kind": kind}


def write_eeg_files(stem: str, epochs: np.ndarray, codes: np.ndarray, labels: np.ndarray,
                    meta: dict) -> tuple[str, str]:
    folder = Path(tempfile.mkdtemp(prefix="voice_eeg_"))
    npz_path = folder / f"{stem}.npz"
    np.savez_compressed(npz_path, epochs=epochs.astype(np.float32), flash_codes=codes.astype(np.int8),
                        labels=labels.astype(np.int8), meta_json=np.array(json.dumps(meta)))
    L, F = codes.shape
    channels = MODEL.store.manifest["channels"]
    frame = pd.DataFrame(epochs.reshape(L * F, -1).round(3),
                         columns=feature_columns(channels, epochs.shape[-1]))
    frame.insert(0, "is_target", labels.reshape(-1))
    frame.insert(0, "flash_code", codes.reshape(-1))
    frame.insert(0, "round", np.tile(np.repeat(np.arange(1, F // 12 + 1), 12), L))
    frame.insert(0, "letter_index", np.repeat(np.arange(1, L + 1), F))
    csv_path = folder / f"{stem}.csv"
    frame.to_csv(csv_path, index=False)
    return str(npz_path), str(csv_path)


def decode_letters(subject: int, epochs: np.ndarray, codes: np.ndarray, labels: np.ndarray | None) -> list[dict]:
    model = MODEL.store.model(subject)
    trials = SubjectTrials(subject, np.arange(len(codes)), epochs, codes.astype(np.int16),
                           (labels if labels is not None else np.zeros_like(codes)).astype(np.int8))
    rows = []
    for i in range(len(trials)):
        result = TrialDecoder(model).decode(trials.repetitions(i))
        st = result["state"]
        rows.append({
            "letter": i + 1,
            "decoded": st["character"],
            "intended": trials.intended_character(i) if labels is not None else None,
            "rounds": st["repetitions"],
            "confidence": st["confidence"],
            "reason": result["reason"],
        })
    return rows


def letters_frame(rows: list[dict]) -> pd.DataFrame:
    has_truth = any(r["intended"] for r in rows)
    data = []
    for r in rows:
        item = {"#": r["letter"], "Decoded": r["decoded"]}
        if has_truth:
            item["Intended"] = r["intended"] or "?"
            item["Match"] = "✓" if r["decoded"] == r["intended"] else "✗"
        item.update({"Rounds": r["rounds"], "Confidence": pct(r["confidence"]),
                     "Stopped": "early (confident)" if r["reason"] == "ADAPTIVE_CONFIDENCE_STOP" else "after 10 rounds"})
        data.append(item)
    return pd.DataFrame(data)


def accuracy_line(rows: list[dict]) -> str:
    scored = [r for r in rows if r["intended"]]
    if not scored:
        return "No answer key in this file, so accuracy can't be checked."
    hits = sum(r["decoded"] == r["intended"] for r in scored)
    return f"**{hits} of {len(scored)} letters correct ({hits / len(scored):.0%})**."


def ui_decode_file(file_path, subject):
    empty = ("", "", pd.DataFrame())
    if MODEL.store is None:
        gr.Warning("The model is still being prepared. Try again in a minute.")
        return empty
    if not file_path:
        gr.Warning("Choose an EEG file first.")
        return empty
    try:
        data = read_eeg_file(file_path)
        if data["meta"].get("participant"):
            subject = int(data["meta"]["participant"])
        rows = decode_letters(int(subject), data["epochs"], data["codes"], data["labels"])
    except (ValueError, KeyError) as error:
        gr.Warning(str(error))
        return empty
    text = "".join(r["decoded"] for r in rows)
    rounds = np.mean([r["rounds"] for r in rows])
    notes = [f"**Read:** {html.escape(data['kind'])}, {len(rows)} letter(s), "
             f"{data['codes'].shape[1]} flashes each. **Decoder:** participant {subject}.",
             f"**Decoded text:** `{text}` · {accuracy_line(rows)} Average {rounds:.1f} rounds per letter."]
    if data["meta"].get("assembled"):
        notes.append("This file was assembled by VOICE from real recorded brain responses (Text → EEG file tab).")
    if data["kind"].startswith("BNCI"):
        notes.append(f"Note: in an original recording, the first {MODEL.store.manifest['calibration_trials']} "
                     "letters were used to train that participant's decoder, so accuracy on those is optimistic.")
    return "\n\n".join(notes), text, letters_frame(rows)


def erp_from_letter(epochs: np.ndarray, codes: np.ndarray, letter: str) -> dict:
    """Average response to the letter's row+column flashes vs the rest (for the preview chart)."""
    channels = MODEL.store.manifest["channels"]
    picks = [channels.index(c) for c in SpellerSession.P300_ELECTRODES if c in channels]
    r = next(k for k, line in enumerate(config.GRID) if letter in line)
    pair = (config.ROW_CODES[r], config.COLUMN_CODES[config.GRID[r].index(letter)])
    signal = epochs[:, picks].astype(np.float64).mean(axis=1)
    hit = np.isin(codes, pair)
    start = int(round(config.EPOCH_START_S * config.SAMPLING_RATE))
    t_ms = (start + np.arange(signal.shape[1]) * config.DECIMATION) / config.SAMPLING_RATE * 1000
    return {"t_ms": t_ms.tolist(), "chosen": signal[hit].mean(axis=0).tolist(),
            "others": signal[~hit].mean(axis=0).tolist(), "intended": None}


def assemble_eeg(text: str, subject: int, seed: int, rounds: int = config.REPETITIONS_PER_TRIAL):
    """Build an EEG file for `text` from one participant's real recorded flash responses.

    Every flash gets a real recorded epoch from that participant: a target response
    (with a P300) when the flash covers the letter, a non-target one otherwise. Only
    held-out trials are used, so the participant's decoder never trained on them.
    """
    letters = [c for c in text.upper() if c in config.CHARACTERS]
    skipped = sorted({c for c in text.upper() if c not in config.CHARACTERS and not c.isspace()})
    if not letters:
        raise ValueError("Type at least one letter or digit that is on the grid (A-Z, 0-9).")
    if len(letters) > 40:
        raise ValueError("Up to 40 letters at a time, please.")
    trials = MODEL.store.trials(subject)
    flat = trials.epochs.reshape(-1, *trials.epochs.shape[2:])
    labels = trials.labels.reshape(-1)
    rng = np.random.default_rng(seed)
    pools = {1: rng.permutation(np.flatnonzero(labels == 1)), 0: rng.permutation(np.flatnonzero(labels == 0))}
    cursor = {0: 0, 1: 0}

    def take(kind: int) -> int:
        if cursor[kind] >= len(pools[kind]):
            pools[kind] = rng.permutation(pools[kind])
            cursor[kind] = 0
        cursor[kind] += 1
        return int(pools[kind][cursor[kind] - 1])

    F = rounds * len(config.FLASH_CODES)
    epochs = np.zeros((len(letters), F, *flat.shape[1:]), np.float32)
    codes = np.zeros((len(letters), F), np.int16)
    out_labels = np.zeros((len(letters), F), np.int8)
    for i, ch in enumerate(letters):
        r = next(k for k, line in enumerate(config.GRID) if ch in line)
        pair = (config.ROW_CODES[r], config.COLUMN_CODES[config.GRID[r].index(ch)])
        k = 0
        for _ in range(rounds):
            for code in rng.permutation(config.FLASH_CODES):
                is_target = int(code in pair)
                epochs[i, k] = flat[take(is_target)]
                codes[i, k], out_labels[i, k] = code, is_target
                k += 1
    meta = {"format": "VOICE EEG file v1", "assembled": True, "participant": subject, "text": "".join(letters),
            "seed": seed, "rounds_per_letter": rounds, "channels": MODEL.store.manifest["channels"],
            "units": "microvolts", "epoch_ms": [-200, 800], "sampling_rate_hz": 64,
            "note": "Each flash is a real recorded response from this participant's held-out BNCI2014-008 "
                    "trials, arranged to spell the text. Not a new recording."}
    return letters, skipped, epochs, codes, out_labels, meta


def ui_text_to_eeg(text, subject, seed):
    empty = (gr.update(value=None, visible=False), "", "", pd.DataFrame())
    if MODEL.store is None:
        gr.Warning("The model is still being prepared. Try again in a minute.")
        return empty
    try:
        subject, seed = int(subject), int(seed or 0)
        letters, skipped, epochs, codes, labels, meta = assemble_eeg(text or "", subject, seed)
    except (ValueError, KeyError) as error:
        gr.Warning(str(error))
        return empty
    stem = f"VOICE_EEG_{''.join(letters)[:20]}_p{subject}"
    npz_path, csv_path = write_eeg_files(stem, epochs, codes, labels, meta)
    rows = decode_letters(subject, epochs, codes, labels)
    decoded = "".join(r["decoded"] for r in rows)
    notes = [f"Made **{len(letters)} letter(s)** × {codes.shape[1]} flashes from participant {subject}'s real "
             f"recorded brain responses: P300 responses where a flash covers the letter, normal responses elsewhere.",
             f"**Round trip:** the decoder reads this file back as `{decoded}`. {accuracy_line(rows)}"]
    if skipped:
        notes.append(f"Skipped characters not on the grid: {' '.join(html.escape(c) for c in skipped)} "
                     "(the grid has A-Z and 0-9 only; spaces are ignored).")
    preview = ("<div class='erp'><p><b>Preview: letter “" + html.escape(letters[0]) + "”</b>: average response "
               "to its row + column flashes vs the other flashes in this file.</p>"
               + erp_svg(erp_from_letter(epochs[0], codes[0], letters[0])) + "</div>")
    return gr.update(value=[npz_path, csv_path], visible=True), "\n\n".join(notes), preview, letters_frame(rows)


def _row_name(code: int) -> str:
    r = config.ROW_CODES.index(code)
    return f"row {r + 1} ({' '.join(config.GRID[r])})"


def _col_name(code: int) -> str:
    c = config.COLUMN_CODES.index(code)
    return f"column {c + 1} ({' '.join(row[c] for row in config.GRID)})"


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    span = max(hi - lo, 1e-6)
    raw = span / n
    mag = 10 ** np.floor(np.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    start = np.floor(lo / step) * step
    return [round(v, 6) for v in np.arange(start, hi + step * 0.5, step)]


def erp_svg(ex: dict) -> str:
    """Line chart: average brain response to the chosen letter's flashes vs the rest."""
    t = np.array(ex["t_ms"])
    series = [("chosen", "Chosen letter's row + column", np.array(ex["chosen"]))]
    if ex.get("intended"):
        series.append(("intended", "Intended letter's row + column", np.array(ex["intended"])))
    series.append(("other", "Other 10 flashes (average)", np.array(ex["others"])))

    W, H, L, R, T, B = 720, 310, 52, 16, 22, 46
    values = np.concatenate([v for _, _, v in series])
    ticks = _nice_ticks(float(values.min()), float(values.max()))
    y0, y1 = ticks[0], ticks[-1]
    x = lambda v: L + (v - t[0]) / (t[-1] - t[0]) * (W - L - R)
    y = lambda v: T + (y1 - v) / (y1 - y0) * (H - T - B)

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" class="erp-svg" '
           f'aria-label="Average brain response over time, chosen letter versus other flashes">']
    w0, w1 = 250, 500
    out.append(f'<rect class="erp-window" x="{x(w0):.1f}" y="{T}" width="{x(w1) - x(w0):.1f}" height="{H - T - B}"/>')
    out.append(f'<text class="erp-note" x="{(x(w0) + x(w1)) / 2:.1f}" y="{H - B - 6}" text-anchor="middle">P300 window</text>')
    for v in ticks:
        out.append(f'<line class="erp-grid" x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
        out.append(f'<text class="erp-tick" x="{L - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:g}</text>')
    for v in (-200, 0, 200, 400, 600, 800):
        out.append(f'<text class="erp-tick" x="{x(v):.1f}" y="{H - B + 18}" text-anchor="middle">{v}</text>')
    out.append(f'<text class="erp-tick" x="{(L + W - R) / 2:.1f}" y="{H - 4}" text-anchor="middle">'
               f'time after the flash (ms)</text>')
    out.append(f'<text class="erp-tick" x="12" y="{(T + H - B) / 2:.1f}" text-anchor="middle" '
               f'transform="rotate(-90 12 {(T + H - B) / 2:.1f})">µV</text>')
    out.append(f'<line class="erp-flash" x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{T}" y2="{H - B}"/>')
    out.append(f'<text class="erp-note" x="{x(0) + 4:.1f}" y="{H - B - 6}">flash</text>')
    for key, _, v in reversed(series):
        pts = " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in zip(t, v))
        out.append(f'<polyline class="erp-line erp-{key}" points="{pts}"/>')
    pk = int(np.argmax(np.where((t >= w0) & (t <= w1), series[0][2], -np.inf)))
    out.append(f'<circle class="erp-peak" cx="{x(t[pk]):.1f}" cy="{y(series[0][2][pk]):.1f}" r="5"/>')
    out.append(f'<text class="erp-label" x="{x(t[pk]):.1f}" y="{y(series[0][2][pk]) - 10:.1f}" '
               f'text-anchor="middle">P300 +{series[0][2][pk]:.1f} µV</text>')
    # hover: one invisible column per time point with exact values
    step = (x(t[1]) - x(t[0])) if len(t) > 1 else 10
    for i, a in enumerate(t):
        tip = f"{a:.0f} ms — " + "; ".join(f"{name}: {v[i]:+.2f} µV" for _, name, v in series)
        out.append(f'<rect class="erp-hit" x="{x(a) - step / 2:.1f}" y="{T}" width="{step:.1f}" '
                   f'height="{H - T - B}"><title>{html.escape(tip)}</title></rect>')
    out.append("</svg>")
    legend = "".join(f'<span class="erp-key"><i class="erp-swatch erp-{k}"></i>{html.escape(n)}</span>'
                     for k, n, _ in series)
    return f'<div class="erp-legend">{legend}</div>' + "".join(out)


def explanation_html(sess: SpellerSession | None) -> str:
    ex = sess.explanation if sess else None
    if not ex:
        return ('<div class="erp-empty">Decode a trial to see the brain signal behind the letter: '
                'which flashes made the participant\'s brain respond, and how strongly.</div>')
    t = sess.current or (sess.trial_log[-1] if sess.trial_log else {})
    row, col = ex["pair"]
    letter = config.GRID[config.ROW_CODES.index(row)][config.COLUMN_CODES.index(col)]
    sm = ex["summary"]
    rounds = ex["rounds"]
    lift = sm["chosen_window_uv"] - sm["others_window_uv"]
    win = ex["window_uv"]
    strongest_row = max(config.ROW_CODES, key=lambda c: win[str(c)])
    strongest_col = max(config.COLUMN_CODES, key=lambda c: win[str(c)])

    lines = [
        f"<p><b>Why “{html.escape(letter)}”?</b> The participant was staring at one letter and silently counting its flashes. "
        f"When the letter you are looking at flashes, the brain produces a positive bump about 250–500 ms later: "
        f"the <b>P300</b> (“that’s the one!”) response.</p>",
        f"<p>Averaged over the {rounds} round{'s' if rounds != 1 else ''} the decoder used, flashes of "
        f"<b>{_row_name(row)}</b> and <b>{_col_name(col)}</b> produced "
        f"<b>{sm['chosen_window_uv']:+.1f} µV</b> in the P300 window, peaking at {sm['peak_ms']:.0f} ms. "
        f"The other 10 flashes averaged <b>{sm['others_window_uv']:+.1f} µV</b>, a difference of {lift:+.1f} µV. "
        f"That row and column cross at <b>{html.escape(letter)}</b>.</p>",
    ]
    intended = t.get("intended")
    if intended and intended == letter:
        lines.append(f'<p class="erp-ok">✓ Correct: the participant was looking at “{html.escape(intended)}”.</p>')
    elif intended and ex.get("intended_pair"):
        ir, ic = ex["intended_pair"]
        wrong = []
        if ir != row:
            wrong.append(f"{_row_name(ir)} gave {win[str(ir)]:+.1f} µV, less than {_row_name(row)} at {win[str(row)]:+.1f} µV")
        if ic != col:
            wrong.append(f"{_col_name(ic)} gave {win[str(ic)]:+.1f} µV, less than {_col_name(col)} at {win[str(col)]:+.1f} µV")
        lines.append(f'<p class="erp-bad">✗ The participant was actually looking at “{html.escape(intended)}”. '
                     f"Its signal was weaker than noise on the wrong line: {'; '.join(wrong)}. "
                     "Try Reject to see the next best letter.</p>")
    lines.append(f'<p class="erp-small">Chart: average of the {", ".join(ex["electrodes"])} electrodes (back and top '
                 f"of the head, where the P300 is strongest). The decoder itself weighs all 8 electrodes and every time "
                 f"point, so this picture is a simplified view of its evidence. Strongest row: {_row_name(strongest_row)}; "
                 f"strongest column: {_col_name(strongest_col)}. Hover the chart for exact values.</p>")
    return '<div class="erp">' + erp_svg(ex) + "".join(lines) + "</div>"


# ------------------------------------------------------------------ UI
def source_badge() -> str:
    if MODEL.store is None:
        return '<span class="voice-badge">Preparing model…</span>'
    if MODEL.store.manifest["synthetic"]:
        return '<span class="voice-badge synthetic">SYNTHETIC test data</span>'
    via = MODEL.store.manifest.get("download")
    via_text = f" · via {via}" if via else ""
    return f'<span class="voice-badge">{MODEL.store.source} · recorded EEG{via_text}</span>'


def status_text() -> str:
    if MODEL.error:
        return f"**{MODEL.status}**\n\n```\n{MODEL.error}\n```"
    if MODEL.store is None:
        last = MODEL.progress()
        return f"⏳ {MODEL.status}" + (f"\n\n`{last}`" if last else "")
    if MODEL.store.manifest["synthetic"]:
        return "> This app uses **synthetic test data**. Build with the real dataset before sharing results."
    return ""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="VOICE · P300 Speller") as demo:
        gr.Markdown("# VOICE\nAdaptive P300 brain-computer interface speller", elem_id="voice-title")
        badge = gr.HTML(source_badge())
        status = gr.Markdown(status_text())

        sess = gr.State(None)

        with gr.Tabs():
            with gr.Tab("Speller"):

                with gr.Row(equal_height=False):
                    with gr.Column(scale=6):
                        grid_buttons = []
                        for row in config.GRID:
                            with gr.Row():
                                for ch in row:
                                    grid_buttons.append(gr.Button(ch, variant="secondary", interactive=False,
                                                                  min_width=40, elem_classes="grid-btn"))
                        gr.Markdown("**Manual correction:** when the decoder is idle you can click a letter to type it "
                                    "yourself, like a caregiver fixing a mistake. These letters do **not** come from "
                                    "brain data and are marked MANUAL in the report.",
                                    elem_classes="voice-note")
                        message = gr.Textbox(label="Message", interactive=False, lines=2,
                                             placeholder="Your message appears here", elem_id="voice-message")
                        with gr.Row():
                            edit_buttons = {a: gr.Button(label, size="sm") for a, label in
                                            [("undo", "Undo"), ("redo", "Redo"), ("space", "Space"),
                                             ("backspace", "⌫ Backspace"), ("clear", "Clear")]}

                    with gr.Column(scale=5):
                        with gr.Group():
                            with gr.Row():
                                subject = gr.Dropdown([str(s) for s in SUBJECTS], value=str(SUBJECTS[0]),
                                                      label="Participant", scale=2)
                                start_btn = gr.Button("Start session", variant="primary", scale=1,
                                                      interactive=MODEL.store is not None)
                            subject_info = gr.Markdown(subject_note(SUBJECTS[0]))
                            with gr.Row():
                                decode_btn = gr.Button("Decode next trial", variant="primary", interactive=False)
                                animate = gr.Checkbox(value=True, label="Show each repetition")

                        decoder = gr.HTML(decoder_html(None))
                        with gr.Row():
                            accept_btn = gr.Button("Accept", variant="primary", interactive=False)
                            reject_btn = gr.Button("Reject → next best", variant="stop", interactive=False)
                        evidence = gr.HTML(evidence_html(None))

                        stats = gr.Markdown(stats_md(None))
                        history = gr.Dataframe(history_frame(None), label="Trial history", interactive=False,
                                               max_height=260, wrap=True)
                        with gr.Row():
                            finish_btn = gr.Button("Finish session", interactive=False)
                        report = gr.File(label="Session report (JSON)", visible=False)

                with gr.Accordion("Why this letter? The brain signal behind it", open=True):
                    explain = gr.HTML(explanation_html(None))

            with gr.Tab("EEG file → text"):
                gr.Markdown(
                    "Upload an EEG file and VOICE decodes the letters in it.\n\n"
                    "- **Original recordings:** a BNCI 008-2014 file such as `A01.mat` (the dataset MOABB downloads).\n"
                    "- **VOICE EEG files:** the `.npz` or `.csv` files made in the *Text → EEG file* tab.\n\n"
                    "Each letter is 120 flashes (10 rounds of 6 rows + 6 columns). If the file contains the "
                    "answer key, VOICE also shows which letters it got right.")
                with gr.Row():
                    upload = gr.File(label="EEG file (.mat, .npz or .csv)", file_types=[".mat", ".npz", ".csv"],
                                     type="filepath", scale=3)
                    with gr.Column(scale=1):
                        file_subject = gr.Dropdown([str(x) for x in SUBJECTS], value=str(SUBJECTS[0]),
                                                   label="Decode with participant's decoder")
                        decode_file_btn = gr.Button("Decode file", variant="primary")
                file_summary = gr.Markdown()
                file_text = gr.Textbox(label="Decoded text", interactive=False, elem_id="voice-file-text")
                file_table = gr.Dataframe(label="Letter by letter", interactive=False, max_height=320)

            with gr.Tab("Text → EEG file"):
                gr.Markdown(
                    "Type a word and VOICE builds the EEG a participant's brain would produce while spelling it, "
                    "**using that participant's real recorded responses**: a P300 response for every flash that "
                    "covers the letter, a normal response for the rest. Download it as `.npz` or `.csv`, or upload "
                    "it in the *EEG file → text* tab to decode it again.\n\n"
                    "It is assembled from real recordings, not a new recording: a brain response only exists when "
                    "someone looks at a letter while the grid flashes.")
                with gr.Row():
                    gen_text = gr.Textbox(label="Text (A-Z, 0-9)", placeholder="HELLO", max_length=60, scale=3)
                    gen_subject = gr.Dropdown([str(x) for x in SUBJECTS], value=str(SUBJECTS[0]), label="Participant", scale=1)
                    gen_seed = gr.Number(value=1, precision=0, label="Random seed", scale=1)
                gen_btn = gr.Button("Make EEG file", variant="primary")
                gen_files = gr.File(label="Download", file_count="multiple", visible=False, interactive=False)
                gen_summary = gr.Markdown()
                gen_preview = gr.HTML()
                gen_table = gr.Dataframe(label="Round trip: what the decoder reads back", interactive=False,
                                         max_height=320)

        gr.Markdown(
            "VOICE replays **recorded, held-out** trials from the BNCI2014-008 P300 dataset "
            "(8 participants, 8 EEG channels), or clearly labelled synthetic trials in test builds. "
            "Each participant's decoder never saw the trials it spells. "
            f"Research demonstration, not a medical device or live EEG acquisition. v{__version__}",
            elem_classes="voice-note")

        # While the model files are being built, poll and unlock the app when ready.
        timer = gr.Timer(5, active=MODEL.store is None and MODEL.error is None)

        def poll(subject_value):
            ready = MODEL.store is not None
            return (source_badge(), status_text(), gr.update(interactive=ready),
                    subject_note(subject_value), gr.Timer(active=not ready and MODEL.error is None))

        timer.tick(poll, subject, [badge, status, start_btn, subject_info, timer])
        demo.load(poll, subject, [badge, status, start_btn, subject_info, timer])

        outputs = [sess, decoder, evidence, message, history, stats, decode_btn, accept_btn,
                   reject_btn, finish_btn, report, explain, *grid_buttons]

        subject.change(subject_note, subject, subject_info)
        decode_file_btn.click(ui_decode_file, [upload, file_subject], [file_summary, file_text, file_table],
                              concurrency_limit=1, concurrency_id="files")
        gen_btn.click(ui_text_to_eeg, [gen_text, gen_subject, gen_seed],
                      [gen_files, gen_summary, gen_preview, gen_table],
                      concurrency_limit=1, concurrency_id="files")
        start_btn.click(start, [sess, subject], outputs)
        decode_btn.click(decode, [sess, animate], outputs)
        accept_btn.click(accept, sess, outputs)
        reject_btn.click(reject, sess, outputs)
        finish_btn.click(finish, sess, outputs)
        for action, button in edit_buttons.items():
            button.click(editor(action), sess, outputs)
        for ch, button in zip(config.CHARACTERS, grid_buttons):
            button.click(typer(ch), sess, outputs)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="VOICE Gradio app")
    parser.add_argument("--share", action="store_true", help="public gradio.live link (Colab)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    args, _ = parser.parse_known_args()
    demo.queue(default_concurrency_limit=8).launch(
        server_name=os.environ.get("HOST", "0.0.0.0"),
        server_port=args.port,
        share=args.share,
        css=CSS,
        theme=gr.themes.Soft(),
        show_error=True,
        max_file_size="60mb",
    )


MODEL.ensure()
demo = build_ui()

if __name__ == "__main__":
    main()
