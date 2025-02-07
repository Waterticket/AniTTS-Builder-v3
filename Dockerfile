FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

# Set working directory
WORKDIR /workspace

# Install dependencies
RUN apt-get update && apt-get install -y --fix-missing \
    git \
    curl \
    wget \
    vim \
    python3-pip \
    ffmpeg \
    && apt-get clean && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Clone the AniTTS-Builder-v3 repository
RUN git clone https://github.com/N01N9/AniTTS-Builder-v3.git

# Set working directory to the cloned repository
WORKDIR /workspace/AniTTS-Builder-v3

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Create necessary directories
RUN mkdir -p data/audio_mp3 data/audio_wav data/result data/transcribe data/video \
    module/model/redimmet module/model/whisper

# Set environment variables if needed (modify according to your project requirements)
ENV PYTHONUNBUFFERED=1

CMD ["/bin/bash"]
