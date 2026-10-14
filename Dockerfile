FROM python:3.11-slim

# ffmpeg is needed to read the audio from uploaded videos
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as a non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: download the model while building so the app starts faster.
# Downloads both models while building (model names match detect.py and visual.py). Uncomment to use:
# RUN python -c "from transformers import pipeline; pipeline('antispoofing', model='Speech-Arena-2025/DF_Arena_500M_V_1', trust_remote_code=True); from transformers import AutoModel; AutoModel.from_pretrained('HoopitAI/video-deepfake-detection-GenD_DINOv3_L_FF', trust_remote_code=True)"

COPY --chown=user . .

EXPOSE 7860
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
