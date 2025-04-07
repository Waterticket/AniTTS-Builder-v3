# AniTTS-Builder-v3

## About

- Summary
  
  This program processes anime videos and subtitles to create Text-to-Speech (TTS) datasets. It extracts and cleans the audio by removing background noise, then slices it into smaller segments. Finally, the program clusters the audio by speaker for easy organization, streamlining the creation of speaker-specific TTS datasets.

  This program operates based on the models from Audio-separator and Speechbrain.

- Developer
  - N01N9

## Installation

This program is developed to run through dockers. Please install CUDA 12.1 + CUDNN 9.X versions and Docker.

1. Install Docker and CUDA 12.1 + CUDNN 9.X version.
   
2. Clone the repositories using the following command : "git clone https://github.com/N01N9/AniTTS-Builder-v3.git". If git is not installed, download the zip file and unzip it. 
   
3. Run the setup.bat file.
   
4. Run the start.bat file and please enter the project name consisting of English and numbers only.

5. When the docker container is up, type: "python main.py"

### For Linux/Unix systems

1. Same as Installation

2. Same as Installation

3. Run the `run.sh` file. Enter the project name to `<data_folder_name>` parameter consisting of English and numbers only.

```sh
./run.sh <data_folder_name>
```
   
## Usage

1. To run this program, you must insert animated video files or audio files in the data_<Project Name>/video path.

2. Access the gradio web server. You can access it through: http://localhost:7860

3. From the first tab to the third tab, press the buttons in order from top to bottom. You can see the progress screen in the CMD window. While one task is running, the other button is disabled.

## Precautions

- The developer's GPU is an RTX 4070ti SUPER, and it took about 6 hours to process approximately 240 minutes (1 season) of animation. Be cautious to prevent the program from terminating during the process, and it is recommended to ensure more than 10GB of free storage space before running the program.

- This program is more likely to function correctly with larger datasets. Therefore, if the animation dataset is insufficient or if you are attempting to extract the voice of a character with limited data, the reliability of the program cannot be guaranteed.

## References

- https://github.com/N01N9/AniTTS-Builder-webUI
- https://github.com/IDRnD/redimnet
- https://github.com/ZFTurbo/Music-Source-Separation-Training
