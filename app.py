"""VOICE: Gradio web app for the adaptive P300 speller.

Local:   python app.py              -> http://localhost:7860
Render:  python app.py              (binds 0.0.0.0:$PORT automatically)
"""

from __future__ import annotations

import html
import json
import os
import tempfile
import time

import gradio as gr
import pandas as pd

from voice import __version__, config
from voice.artifacts import ArtifactStore
from voice.session import SpellerSession

STEP_DELAY_S = float(os.environ.get("VOICE_STEP_DELAY_S", "0.35"))  # pause between repetitions while a trial is decoded

try:
    STORE: ArtifactStore | None = ArtifactStore(os.environ.get("VOICE_ARTIFACT_DIR", config.ARTIFACT_DIR))
    LOAD_ERROR = None
except FileNotFoundError as error:
    STORE, LOAD_ERROR = None, str(error)

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
    sess = SpellerSession(STORE, int(subject))
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
    meta = STORE.manifest["subjects"].get(str(subject), {})
    if not meta:
        return ""
    return (f"{meta['held_out_trials']} held-out trials. Build-time accuracy: "
            f"**{pct(meta['adaptive_accuracy'])}** adaptive ({meta['mean_repetitions']} repetitions on average), "
            f"**{pct(meta['fixed_10_repetition_accuracy'])}** with all 10 repetitions.")


# ------------------------------------------------------------------ UI
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="VOICE · P300 Speller") as demo:
        if STORE is None:
            gr.Markdown(f"# VOICE\n\n**The model files are missing.**\n\n`{LOAD_ERROR}`\n\n"
                        "Run `python scripts/build_artifacts.py` and restart the app.")
            return demo

        synthetic = STORE.manifest["synthetic"]
        badge = ('<span class="voice-badge synthetic">SYNTHETIC test data</span>' if synthetic
                 else f'<span class="voice-badge">{STORE.source} · recorded EEG</span>')
        gr.Markdown("# VOICE\nAdaptive P300 brain-computer interface speller", elem_id="voice-title")
        gr.HTML(badge)
        if synthetic:
            gr.Markdown("> This deployment uses **synthetic test data**. Rebuild with the real dataset before sharing results.")

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
                        subject = gr.Dropdown([str(s) for s in STORE.subjects], value=str(STORE.subjects[0]),
                                              label="Participant", scale=2)
                        start_btn = gr.Button("Start session", variant="primary", scale=1)
                    subject_info = gr.Markdown(subject_note(STORE.subjects[0]))
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

        source_text = ("simulated trials shaped like the BNCI2014-008 recordings" if synthetic else
                       "**recorded, held-out** trials from the BNCI2014-008 P300 dataset (8 participants, 8 EEG channels)")
        gr.Markdown(
            f"VOICE replays {source_text}. Each participant's decoder never saw the trials it spells. "
            f"Research demonstration, not a medical device or live EEG acquisition. v{__version__}",
            elem_classes="voice-note")

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


demo = build_ui()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=8).launch(
        server_name=os.environ.get("HOST", "0.0.0.0"),
        server_port=int(os.environ.get("PORT", "7860")),
        css=CSS,
        theme=gr.themes.Soft(),
        show_error=True,
    )
