"""Streamlit app: upload a video -> checks the VOICE and the FACE -> REAL / FAKE / UNCERTAIN."""
import os
import tempfile

import streamlit as st

import settings   # light file: the heavy AI libraries are only loaded when you click Analyze

st.set_page_config(page_title="Deepfake Detector", page_icon="🎙️", layout="centered")

VIDEO_EXT = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
IMAGE_EXT = set(settings.IMAGE_EXT)


@st.cache_resource(show_spinner="Loading the AI models. The first time this downloads several GB "
                                "and can take 10+ minutes. Keep this page open...")
def load_models():
    import detect
    detect.load_models()
    return True


st.title("🎙️ Deepfake Detector")
st.caption("Upload a video. The voice and the face are checked separately, then combined "
           "into one verdict.")

with st.sidebar:
    st.header("About")
    st.write("**Voice model**")
    st.code(settings.AUDIO_MODEL, language=None)
    st.write("**Face model**")
    st.code(settings.VIDEO_MODEL if settings.ENABLE_FACE else "(face check is off: voice-only mode)",
            language=None)
    st.write(f"**Verdict rules**  \nFAKE if combined score ≥ {settings.FAKE_ABOVE:.2f}  \n"
             f"REAL if ≤ {settings.REAL_BELOW:.2f}  \nOtherwise UNCERTAIN")
    st.write("**Limits**  \nThe face model was trained on older face-swap data, so brand-new "
             "AI-generated video can fool it. Music, heavy compression and new voice generators "
             "can fool the voice model. Treat results as probabilities, not proof.")

if settings.ffmpeg_exe() is None:
    st.error("ffmpeg was not found. Run `pip install imageio-ffmpeg` and restart the app.")
    st.stop()

uploaded = st.file_uploader(
    "Upload a video (audio or an image also works)",
    type=["mp4", "mov", "webm", "mkv", "avi", "mp3", "wav", "m4a", "jpg", "jpeg", "png"],
)

if uploaded is not None:
    ext = os.path.splitext(uploaded.name)[1].lower()
    if ext in VIDEO_EXT:
        st.video(uploaded)
    elif ext in IMAGE_EXT:
        st.image(uploaded)
    else:
        st.audio(uploaded)

    if st.button("Analyze", type="primary", use_container_width=True):
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded.getbuffer())
            path = tmp.name
        r = None
        try:
            load_models()
            import detect
            with st.spinner("Checking voice and face... (can take a minute on CPU)"):
                r = detect.analyze(path)
        except Exception as e:
            st.error(f"Could not analyse this file: {e}")
        finally:
            if os.path.exists(path):
                os.remove(path)

        if r:
            score = r["score"]
            if r["verdict"] == "FAKE":
                st.error(f"### 🚨 FAKE detected  \nCombined fake score: {score:.1%}")
            elif r["verdict"] == "REAL":
                st.success(f"### ✅ Looks REAL  \nCombined fake score: {score:.1%}")
            else:
                st.warning(f"### ⚠️ UNCERTAIN  \nCombined fake score: {score:.1%}. "
                           "The models are not sure. Try a cleaner clip.")
            st.progress(min(max(score, 0.0), 1.0), text="Fake likelihood")

            c1, c2 = st.columns(2)
            a, v = r["audio"], r["video"]
            c1.metric("🎙️ Voice", f"{a['score']:.1%} fake" if a else "not checked")
            c2.metric("🙂 Face", f"{v['score']:.1%} fake" if v else "not checked")

            for n in r["notes"]:
                st.info(n)
            with st.expander("Details"):
                if a:
                    st.write(f"Voice: {a['speech_sec']:.1f}s of speech analysed in {a['windows']} windows.")
                if v:
                    st.write(f"Face: {v['frames_used']} of {v['frames_checked']} sampled frames "
                             f"contained a face (highest single-frame score {v['max_frame_score']:.1%}).")
else:
    st.info("Upload a video to begin.")