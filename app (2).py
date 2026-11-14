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
        result = self.decoder.decode(self.trials.repetitions(i))
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
        self._log("TRIAL_DECODED", trial=self.current["trial"], decoded=self.candidate,
                  repetitions=state["repetitions"], reason=result["reason"])
        return {"trial": self.current, "trace": result["trace"]}

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
    from scipy.signal import butter, sosfiltfilt

    folder = Path(data_dir or os.environ.get("VOICE_DATA_DIR", Path.home() / "voice_data")).expanduser()
    expected = config.TRIALS_PER_SUBJECT * config.EVENTS_PER_TRIAL
    fs = config.SAMPLING_RATE
    sos = butter(4, [config.FILTER_LOW_HZ, config.FILTER_HIGH_HZ], btype="bandpass", fs=fs, output="sos")
    first = int(round(config.EPOCH_START_S * fs))
    last = int(round(config.EPOCH_END_S * fs))
    offsets = np.arange(first, last + 1)[:: config.DECIMATION]      # 65 samples
    pad = max(0, -first, last)

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

        eeg_uv = np.asarray(run.X, dtype=np.float64).T              # (8, n_samples), microvolts
        target_channel = np.asarray(run.y).astype(int).ravel()
        flash_channel = np.asarray(run.y_stim).astype(int).ravel()
        classes = [str(c).strip().lower() for c in np.atleast_1d(run.classes)]
        del run
        if eeg_uv.shape[0] != len(CHANNELS):
            sys.exit(f"Subject {s}: expected {len(CHANNELS)} EEG channels, found {eeg_uv.shape[0]}.")
        if "target" not in classes:
            sys.exit(f"Subject {s}: unexpected class names {classes}.")
        target_value = classes.index("target") + 1

        previous = np.r_[0, flash_channel[:-1]]
        onsets = np.flatnonzero((flash_channel != 0) & (previous == 0))
        codes = flash_channel[onsets] + 2                            # 1-12 -> 3-14, as MOABB
        labels = (target_channel[onsets] == target_value).astype(np.int8)
        if len(onsets) != expected:
            sys.exit(f"Subject {s}: expected {expected} flashes, found {len(onsets)}.")
        if np.any(target_channel[onsets] == 0):
            sys.exit(f"Subject {s}: some flashes have no target/non-target mark.")

        eeg_uv = sosfiltfilt(sos, eeg_uv, axis=1)
        eeg_uv = np.pad(eeg_uv, ((0, 0), (pad, pad)))
        epochs = eeg_uv[:, onsets[:, None] + offsets[None, :] + pad].transpose(1, 0, 2)
        out[s] = (np.clip(epochs, -60000, 60000).astype(np.float32), labels, codes.astype(np.int8))
        del eeg_uv, epochs, target_channel, flash_channel
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

        with gr.Row(equal_height=False):
            with gr.Column(scale=6):
                grid_buttons = []
                for row in config.GRID:
                    with gr.Row():
                        for ch in row:
                            grid_buttons.append(gr.Button(ch, variant="secondary", interactive=False,
                                                          min_width=40, elem_classes="grid-btn"))
                gr.Markdown("When the decoder is idle you can also click a letter to type it yourself.",
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
                   reject_btn, finish_btn, report, *grid_buttons]

        subject.change(subject_note, subject, subject_info)
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
    )


MODEL.ensure()
demo = build_ui()

if __name__ == "__main__":
    main()
