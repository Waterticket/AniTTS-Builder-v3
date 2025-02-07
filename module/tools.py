import os
import ffmpeg
import requests
from tqdm import tqdm
import shutil

def convert_to_wav(input_file, output_wav):
    """
    Convert an audio or video file to WAV format.
    
    Args:
        input_file (str): Path to the input file (audio or video).
        output_wav (str): Path to save the output WAV file.
    """
    try:
        ext = os.path.splitext(input_file)[1].lower()
        
        if ext in [".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm", ".mp3", ".aac", ".flac", ".ogg", ".m4a", ".wav"]:
            print(f"Processing: {input_file} -> {output_wav}")
            (
                ffmpeg
                .input(input_file)
                .output(output_wav, format="wav", acodec="pcm_s16le", ac=1)
                .run(overwrite_output=True)
            )
            print(f"Conversion completed: {output_wav}")
        else:
            print(f"Unsupported file format: {ext}")
    except ffmpeg.Error as e:
        print(f"Error: {e.stderr.decode()}")

def batch_convert_to_wav(input_folder, output_folder):
    """
    Convert all audio and video files in a folder to WAV format.
    
    Args:
        input_folder (str): Path to the folder containing input files.
        output_folder (str): Path to save the converted WAV files.
    """
    os.makedirs(output_folder, exist_ok=True)
    supported_extensions = (".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm", ".mp3", ".aac", ".flac", ".ogg", ".m4a", ".wav")
    
    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)
        if os.path.isfile(input_path) and file_name.lower().endswith(supported_extensions):
            output_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".wav")
            convert_to_wav(input_path, output_path)

def convert_wav_to_mp3(input_wav, output_mp3):
    """
    Convert a WAV file to MP3 format (mono, 16kHz).
    
    Args:
        input_wav (str): Path to the input WAV file.
        output_mp3 (str): Path to save the output MP3 file.
    """
    try:
        (
            ffmpeg
            .input(input_wav)
            .output(output_mp3, format='mp3', acodec='libmp3lame', ar='16000', ac=1)
            .run(overwrite_output=True)
        )
        print(f"Converted to MP3: {output_mp3}")
    except ffmpeg.Error as e:
        error_message = e.stderr.decode() if e.stderr else "Unknown error"
        print(f"FFmpeg Error: {error_message}")

def batch_convert_wav_to_mp3(input_folder, output_folder):
    """
    Convert all WAV files in a folder to MP3 format (mono, 16kHz).
    
    Args:
        input_folder (str): Path to the folder containing WAV files.
        output_folder (str): Path to save the converted MP3 files.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)
        if os.path.isfile(input_path) and file_name.lower().endswith(".wav"):
            output_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".mp3")
            convert_wav_to_mp3(input_path, output_path)

def download_file(url, save_path):
    """
    Download a file from a URL and save it to a specified path.
    
    Args:
        url (str): URL of the file to download.
        save_path (str): Path to save the downloaded file.
    """
    if os.path.exists(save_path):
        print(f"File already exists: {save_path}")
        return
    
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get("content-length", 0))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "wb") as file, tqdm(
        desc=os.path.basename(save_path), total=total_size, unit="B", unit_scale=True, unit_divisor=1024
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
            bar.update(len(chunk))
    
    print(f"Downloaded: {save_path}")

def download_pretrained_models():
    """
    Download necessary pretrained models from Hugging Face.
    """
    model_urls = {
        "vocal_models/Kim_MelBandRoformer.ckpt": "https://huggingface.co/Sucial/MSST-WebUI/resolve/main/All_Models/vocal_models/Kim_MelBandRoformer.ckpt",
        "vocal_models/model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt": "https://huggingface.co/Sucial/MSST-WebUI/resolve/main/All_Models/vocal_models/model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        "single_stem_models/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt": "https://huggingface.co/Sucial/MSST-WebUI/resolve/main/All_Models/single_stem_models/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
        "single_stem_models/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt": "https://huggingface.co/Sucial/MSST-WebUI/resolve/main/All_Models/single_stem_models/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
    }
    
    for filename, url in model_urls.items():
        save_path = os.path.join("./module/model/MSST_WebUI/pretrain/", filename)
        download_file(url, save_path)

def move_matching_text_files(folder1, folder2):
    """
    Moves text files from folder1 to the corresponding subfolders in folder2 
    based on matching filenames with WAV files.

    Args:
        folder1 (str): Path to the folder containing text files.
        folder2 (str): Path to the main folder containing subfolders with WAV files.
    """
    # Get all text file names in folder1 (without extension)
    text_files = {os.path.splitext(f)[0]: os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith(".txt")}
    
    # Track moved text files
    moved_files = set()

    # Iterate through all subfolders in folder2
    for root, _, files in os.walk(folder2):
        for file in files:
            if file.endswith(".wav"):
                wav_name = os.path.splitext(file)[0]  # Get the name without extension
                if wav_name in text_files:
                    text_file_path = text_files[wav_name]
                    target_path = os.path.join(root, os.path.basename(text_file_path))
                    
                    # Move text file
                    shutil.move(text_file_path, target_path)
                    moved_files.add(wav_name)
                    print(f"Moved: {text_file_path} -> {target_path}")

    # Remove remaining text files in folder1
    remaining_files = set(text_files.keys()) - moved_files
    for file_name in remaining_files:
        file_path = text_files[file_name]
        os.remove(file_path)
        print(f"Deleted: {file_path}")