#!/bin/bash

FOLDER_NAME=$1

if [ -z "$FOLDER_NAME" ]; then
    echo "Usage: $0 <data_folder_name>"
    exit 1
fi

# Change to the script's directory
cd "$(dirname "$0")"

# Check anitts-builder-v3 image exists
if [ -z "$(docker images -q anitts-builder-v3)" ]; then
    echo "AniTTS-Builder-v3 image not found. Build image."
    docker build --no-cache -t anitts-builder-v3 .
fi

# Create data_<FOLDER_NAME> folder and its subdirectories in the current script directory
DATA_DIR="./data_${FOLDER_NAME}"

mkdir -p "$DATA_DIR/audio_mp3"
mkdir -p "$DATA_DIR/audio_wav"
mkdir -p "$DATA_DIR/result"
mkdir -p "$DATA_DIR/transcribe"
mkdir -p "$DATA_DIR/video"

echo "Folder $DATA_DIR and subfolders created successfully."

# Run the container
docker run -it --rm -p 7860:7860 --gpus all \
    --name "anitts-container-${FOLDER_NAME}" \
    -v "$(pwd)/data_${FOLDER_NAME}:/workspace/AniTTS-Builder-v3/data" \
    -v "$(pwd)/module/model:/workspace/AniTTS-Builder-v3/module/model" anitts-builder-v3 bash -c "python main.py"
