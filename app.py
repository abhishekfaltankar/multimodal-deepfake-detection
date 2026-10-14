"""Streamlit app: upload media -> check voice and face -> REAL / FAKE / UNCERTAIN.

UI and report improvements only. Existing settings and detect.analyze(path)
contract are preserved.
"""
import os
import tempfile
import csv
import io
from datetime import datetime

import streamlit as st
import settings  # Heavy AI libraries remain loaded only when Analyze is clicked.

st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

VIDEO_EXT = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
IMAGE_EXT = set(settings.IMAGE_EXT)


@st.cache_resource(
    show_spinner="Loading the AI models. The first time this downloads several GB "
    "and can take 10+ minutes. Keep this page open..."
)
def load_models():
    import detect
    detect.load_models()
    return True


# ======================== BEAUTIFUL UI THEME ========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 5% 0%, rgba(37,99,235,.20), transparent 35%),
        radial-gradient(ellipse at 95% 5%, rgba(20,184,166,.14), transparent 30%),
        #08111f;
    color: #edf4ff;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stHeader"] { background: rgba(8,17,31,.8); }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111f35, #0b1526);
    border-right: 1px solid rgba(165,190,230,.16);
}
[data-testid="stSidebar"] * { color: #e5efff; }
.block-container { max-width: 1200px; padding-top: 2rem; padding-bottom: 3rem; }
h1, h2, h3, h4, [data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif !important;
}
.hero {
    padding: 30px;
    border: 1px solid rgba(94,234,212,.25);
    border-radius: 24px;
    background: linear-gradient(120deg, #142945, #111d31);
    box-shadow: 0 18px 45px rgba(0,0,0,.20);
    margin-bottom: 24px;
}
.eyebrow {
    color: #5eead4; font-size: .78rem; font-weight: 700;
    letter-spacing: .16em; text-transform: uppercase;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2rem, 5vw, 3.1rem); line-height: 1.12;
    font-weight: 700; margin: 12px 0; color: #f4f8ff;
}
.hero-copy { color: #b8c8e0; font-size: 1rem; line-height: 1.7; }
.panel {
    border: 1px solid rgba(165,190,230,.17);
    background: linear-gradient(145deg, #16243b, #101b2e);
    border-radius: 18px; padding: 20px; margin-bottom: 12px;
}
.panel-label {
    color: #9fb3d1; font-size: .77rem; font-weight: 700;
    letter-spacing: .1em; text-transform: uppercase;
}
.panel-value {
    color: #f4f8ff; font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35rem; font-weight: 700; margin-top: 8px;
    overflow-wrap: anywhere;
}
.panel-sub { color: #a8b7cf; font-size: .88rem; line-height: 1.6; margin-top: 7px; }
.verdict {
    padding: 24px; border: 1px solid rgba(165,190,230,.18);
    border-radius: 20px; background: #111e32; margin: 12px 0 20px;
}
.verdict-fake {
    border-color: rgba(248,113,113,.55);
    background: linear-gradient(120deg, rgba(127,29,29,.35), #111e32);
}
.verdict-real {
    border-color: rgba(94,234,212,.48);
    background: linear-gradient(120deg, rgba(13,148,136,.23), #111e32);
}
.verdict-uncertain {
    border-color: rgba(251,191,36,.5);
    background: linear-gradient(120deg, rgba(146,64,14,.25), #111e32);
}
.verdict-kicker {
    color: #a8b7cf; font-size: .78rem; font-weight: 700;
    letter-spacing: .14em; text-transform: uppercase;
}
.verdict-main {
    color: #f8fbff; font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem; font-weight: 700; margin: 8px 0;
}
div.stButton > button[kind="primary"],
div.stDownloadButton > button[kind="primary"] {
    background: linear-gradient(100deg, #5eead4, #60a5fa);
    color: #071b27; border: none; border-radius: 12px;
    min-height: 3rem; font-weight: 700;
}
div.stButton > button[kind="primary"]:hover,
div.stDownloadButton > button[kind="primary"]:hover {
    filter: brightness(1.08); border: none;
}
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(94,234,212,.42);
    border-radius: 16px; padding: 12px; background: rgba(17,29,49,.55);
}
[data-testid="stMetric"] {
    background: #14233a; border: 1px solid rgba(165,190,230,.16);
    border-radius: 16px; padding: 16px;
}
[data-testid="stMetricLabel"] { color: #a8b7cf !important; }
[data-testid="stMetricValue"] { color: #f4f8ff !important; }
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #5eead4, #60a5fa);
}
div[data-testid="stExpander"] {
    border: 1px solid rgba(165,190,230,.17);
    border-radius: 14px; background: rgba(17,29,49,.5);
}
hr { border-color: rgba(165,190,230,.17); }
.small-muted { color: #9eafc9; font-size: .84rem; }
@media (max-width: 700px) {
    .block-container { padding: 1rem .8rem 2rem; }
    .hero { padding: 20px; }
    .verdict-main { font-size: 1.6rem; }
}
</style>
""", unsafe_allow_html=True)
# ===================================================================


# ============================ SIDEBAR ==============================
with st.sidebar:
    st.markdown("## 🛡️ Deepfake Detector")
    st.caption("Multimodal media screening")
    st.divider()

    st.markdown("### Model configuration")
    st.write("**🎙️ Voice model**")
    st.code(settings.AUDIO_MODEL, language=None)
    st.write("**🙂 Face model**")
    st.code(
        settings.VIDEO_MODEL
        if settings.ENABLE_FACE
        else "(face check is off: voice-only mode)",
        language=None,
    )

    st.divider()
    st.markdown("### Verdict thresholds")
    st.write(f"**FAKE** · combined score ≥ `{settings.FAKE_ABOVE:.2f}`")
    st.write(f"**REAL** · combined score ≤ `{settings.REAL_BELOW:.2f}`")
    st.write("**UNCERTAIN** · between both thresholds")

    st.divider()
    st.markdown("### Important limitations")
    st.caption(
        "The face model was trained on older face-swap data, so newer AI-generated "
        "video may fool it. Music, compression, noise, and new voice generators may "
        "affect audio results. Scores are model estimates, not proof of authenticity."
    )
    st.caption(
        "Avoid uploading private or sensitive media to a public demo. Check the "
        "hosting environment's privacy and retention settings."
    )
# ===================================================================


# ============================ MAIN PAGE ============================
st.markdown("""
<div class="hero">
    <div class="eyebrow">AI-powered media verification</div>
    <div class="hero-title">🛡️ Deepfake Detector</div>
    <div class="hero-copy">
        Investigate suspicious media through separate voice and face checks.
        Review the combined prediction, understand each model's score, and export
        a detailed analysis report. Built for careful investigation—not as final
        proof of authenticity.
    </div>
</div>
""", unsafe_allow_html=True)

feature1, feature2, feature3 = st.columns(3)
with feature1:
    st.markdown("""
    <div class="panel">
        <div class="panel-label">01 · Input</div>
        <div class="panel-value">One media file</div>
        <div class="panel-sub">Preview a video, audio clip, or supported image.</div>
    </div>
    """, unsafe_allow_html=True)
with feature2:
    st.markdown("""
    <div class="panel">
        <div class="panel-label">02 · Analysis</div>
        <div class="panel-value">Voice + face</div>
        <div class="panel-sub">Review the separate outputs from enabled detectors.</div>
    </div>
    """, unsafe_allow_html=True)
with feature3:
    st.markdown("""
    <div class="panel">
        <div class="panel-label">03 · Outcome</div>
        <div class="panel-value">Detailed report</div>
        <div class="panel-sub">Inspect scores, prediction, notes, and export results.</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.markdown("## 01 — Upload your media")
st.caption("Choose a supported file to preview it before running the analysis.")

if settings.ffmpeg_exe() is None:
    st.error("FFmpeg was not found. Run `pip install imageio-ffmpeg` and restart the app.")
    st.stop()

uploaded = st.file_uploader(
    "Choose a file to inspect",
    type=["mp4", "mov", "webm", "mkv", "avi", "mp3", "wav", "m4a", "jpg", "jpeg", "png"],
    help="Supported formats depend on your existing detect.py pipeline and installed codecs.",
)

if uploaded is not None:
    ext = os.path.splitext(uploaded.name)[1].lower()
    preview_col, info_col = st.columns([1.25, 1], gap="large")

    with preview_col:
        st.markdown("### Media preview")
        if ext in VIDEO_EXT:
            st.video(uploaded)
        elif ext in IMAGE_EXT:
            st.image(uploaded, use_container_width=True)
        else:
            st.audio(uploaded)

    with info_col:
        st.markdown("### File details")
        safe_display_name = (
            uploaded.name.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        st.markdown(f"""
        <div class="panel">
            <div class="panel-label">Selected file</div>
            <div class="panel-value">{safe_display_name}</div>
            <div class="panel-sub">
                Format: {ext.lstrip('.').upper() or 'Unknown'}<br>
                Size: {uploaded.size / (1024 * 1024):.2f} MB
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.caption(
            "Tip: clearer speech, a visible face, and less compression may help the "
            "models, but they do not guarantee a correct result."
        )
        analyze_clicked = st.button(
            "✨ Analyze media",
            type="primary",
            use_container_width=True,
        )

    if analyze_clicked:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded.getbuffer())
            path = tmp.name

        r = None
        started_at = datetime.now()

        try:
            load_models()
            import detect
            with st.spinner("Inspecting media signals… this can take a minute on CPU."):
                r = detect.analyze(path)
        except Exception as e:
            st.error(f"Could not analyse this file: {e}")
        finally:
            if os.path.exists(path):
                os.remove(path)

        if r:
            score = r["score"]
            verdict = r["verdict"]
            a, v = r["audio"], r["video"]
            elapsed = (datetime.now() - started_at).total_seconds()

            # ---------------- OVERALL PREDICTION ----------------
            if verdict == "FAKE":
                verdict_class = "verdict-fake"
                verdict_icon = "🚨"
                verdict_title = "FAKE DETECTED"
                verdict_description = (
                    "The combined score crossed your configured fake threshold. "
                    "Review individual signals and the original source."
                )
            elif verdict == "REAL":
                verdict_class = "verdict-real"
                verdict_icon = "✓"
                verdict_title = "LIKELY REAL"
                verdict_description = (
                    "The combined score is within your configured real range. "
                    "This does not guarantee that the media is authentic."
                )
            else:
                verdict_class = "verdict-uncertain"
                verdict_icon = "!"
                verdict_title = "UNCERTAIN"
                verdict_description = (
                    "The score falls between your configured thresholds. "
                    "Review the voice and face results separately."
                )

            st.divider()
            st.markdown("## 02 — Analysis results")
            st.markdown(f"""
            <div class="verdict {verdict_class}">
                <div class="verdict-kicker">Overall prediction</div>
                <div class="verdict-main">{verdict_icon} {verdict_title}</div>
                <div class="hero-copy">{verdict_description}</div>
            </div>
            """, unsafe_allow_html=True)

            # ---------------- SCORE CARDS ----------------
            c1, c2, c3 = st.columns(3)
            c1.metric("Combined fake score", f"{score:.1%}")
            c2.metric("🎙️ Voice score", f"{a['score']:.1%}" if a else "Not checked")
            c3.metric("🙂 Face score", f"{v['score']:.1%}" if v else "Not checked")

            st.markdown("### Combined score indicator")
            st.progress(
                min(max(float(score), 0.0), 1.0),
                text=f"Model-estimated fake score: {score:.1%}",
            )
            st.caption(
                f"REAL ≤ {settings.REAL_BELOW:.2f}  ·  "
                f"FAKE ≥ {settings.FAKE_ABOVE:.2f}  ·  "
                "Scores between the thresholds are UNCERTAIN."
            )

            # ---------------- INDIVIDUAL SIGNALS ----------------
            st.markdown("### Signal breakdown")
            left, right = st.columns(2)

            with left:
                st.markdown(
                    '<div class="panel"><div class="panel-label">🎙️ Voice analysis</div>',
                    unsafe_allow_html=True,
                )
                if a:
                    st.markdown(
                        f'<div class="panel-value">{a["score"]:.1%} fake score</div>'
                        f'<div class="panel-sub">{a["speech_sec"]:.1f} seconds of speech analysed'
                        f'<br>{a["windows"]} analysis windows</div>',
                        unsafe_allow_html=True,
                    )
                    st.progress(min(max(float(a["score"]), 0.0), 1.0))
                else:
                    st.markdown(
                        '<div class="panel-value">Not checked</div>'
                        '<div class="panel-sub">No voice result was returned by the existing pipeline.</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            with right:
                st.markdown(
                    '<div class="panel"><div class="panel-label">🙂 Face analysis</div>',
                    unsafe_allow_html=True,
                )
                if v:
                    st.markdown(
                        f'<div class="panel-value">{v["score"]:.1%} fake score</div>'
                        f'<div class="panel-sub">{v["frames_used"]} of {v["frames_checked"]} '
                        f'sampled frames contained a face<br>'
                        f'Highest single-frame score: {v["max_frame_score"]:.1%}</div>',
                        unsafe_allow_html=True,
                    )
                    st.progress(min(max(float(v["score"]), 0.0), 1.0))
                else:
                    st.markdown(
                        '<div class="panel-value">Not checked</div>'
                        '<div class="panel-sub">No face result was returned by the existing pipeline.</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            # ---------------- EXISTING MODEL NOTES ----------------
            if r["notes"]:
                st.markdown("### Model notes")
                for n in r["notes"]:
                    st.info(n)

            # ---------------- FULL REPORT / CSV EXPORT ----------------
            # Presentation-only: read existing outputs; do not change inference.
            report_rows = [
                ("File name", uploaded.name),
                ("File format", ext.lstrip(".").upper()),
                ("File size", f"{uploaded.size / (1024 * 1024):.2f} MB"),
                ("Prediction", verdict),
                ("Combined fake score", f"{score:.1%}"),
                ("Voice fake score", f"{a['score']:.1%}" if a else "Not checked"),
                ("Face fake score", f"{v['score']:.1%}" if v else "Not checked"),
                (
                    "Speech duration analysed",
                    f"{a['speech_sec']:.1f} seconds" if a else "Not available",
                ),
                ("Voice analysis windows", str(a["windows"]) if a else "Not available"),
                (
                    "Frames containing a face",
                    f"{v['frames_used']} / {v['frames_checked']}" if v else "Not available",
                ),
                (
                    "Highest single-frame score",
                    f"{v['max_frame_score']:.1%}" if v else "Not available",
                ),
                ("REAL threshold", f"{settings.REAL_BELOW:.2f}"),
                ("FAKE threshold", f"{settings.FAKE_ABOVE:.2f}"),
                ("Analysis duration", f"{elapsed:.1f} seconds"),
                ("Report generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ]

            st.divider()
            st.markdown("## 03 — Full analysis report")
            st.caption(
                "A record of this analysis run. Scores are model outputs, not calibrated "
                "probabilities or proof."
            )

            with st.expander("📄 View full report", expanded=True):
                for label, value in report_rows:
                    st.write(f"**{label}:** {value}")
                if r["notes"]:
                    st.markdown("#### Additional notes")
                    for note in r["notes"]:
                        st.write(f"- {note}")

            csv_buffer = io.StringIO()
            csv_writer = csv.writer(csv_buffer)
            csv_writer.writerow(["Field", "Result"])
            csv_writer.writerows(report_rows)

            safe_filename = os.path.splitext(os.path.basename(uploaded.name))[0]
            st.download_button(
                "⬇️ Download report as CSV",
                data=csv_buffer.getvalue(),
                file_name=f"{safe_filename}_deepfake_report.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )

            st.info(
                "Interpret this result alongside the original source and other evidence. "
                "A low fake score does not prove authenticity."
            )
else:
    st.markdown("""
    <div class="panel">
        <div class="panel-label">Ready when you are</div>
        <div class="panel-value">Your report will appear here after analysis.</div>
        <div class="panel-sub">
            The report includes the overall prediction, combined score, separate voice
            and face scores, model notes, analysis metadata, and a CSV export.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.markdown(
    '<div class="small-muted">Deepfake Detector · AI-assisted screening prototype · '
    'Always verify important findings independently.</div>',
    unsafe_allow_html=True,
)
