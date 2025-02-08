import os
import torch
import torchaudio
import numpy as np
from glob import glob
import whisperx

def load_model(model_id, cache_dir=None):
    """
    Load the model using whisperx.

    Args:
        model_id (str): Whisper model identifier (e.g., "large", "medium", etc.).
        cache_dir (str, optional): Cache directory (not used in whisperx, kept for compatibility).

    Returns:
        Model: Loaded whisperx model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"
    model = whisperx.load_model(model_id, device, compute_type=compute_type)
    return model

def detect_non_silent(audio, sr=16000, silence_db=-40, min_silence_len_sec=1, min_gap_sec=1, min_duration_sec=0.05):
    """
    Detect non-silent segments in an audio waveform and merge segments separated by short silences.

    Args:
        audio (np.ndarray): Mono audio waveform.
        sr (int): Sample rate.
        silence_db (float): Decibel threshold to consider as silence.
        min_silence_len_sec (float): Minimum duration to consider as silence.
        min_gap_sec (float): Maximum gap between non-silent segments to merge them.
        min_duration_sec (float): Minimum duration for a valid non-silent segment.

    Returns:
        List[Tuple[float, float]]: List of (start_time, end_time) tuples for non-silent segments.
    """
    if len(audio) == 0:
        return []

    db_audio = 20 * np.log10(np.abs(audio) + 1e-10)
    silent = (db_audio < silence_db)

    segments, start, is_silent = [], 0, silent[0]
    for i in range(1, len(silent)):
        if silent[i] != is_silent:
            segments.append((is_silent, start, i))
            start, is_silent = i, silent[i]
    segments.append((is_silent, start, len(silent)))

    non_silent_segments = [(s / sr, e / sr) for is_silent, s, e in segments if not is_silent]

    merged_segments = []
    if non_silent_segments:
        prev_start, prev_end = non_silent_segments[0]
        for start, end in non_silent_segments[1:]:
            if start - prev_end <= min_gap_sec:
                prev_end = end
            else:
                merged_segments.append((prev_start, prev_end))
                prev_start, prev_end = start, end
        merged_segments.append((prev_start, prev_end))

    return [(s, e) for s, e in merged_segments if e - s >= min_duration_sec]

def audio_normalize(audio):
    """
    Normalize an audio waveform to -20 dB RMS.

    Args:
        audio (tuple): Tuple containing (waveform, sample_rate).

    Returns:
        tuple: Normalized waveform and sample rate.
    """
    waveform, sample_rate = audio
    waveform = waveform.squeeze().numpy()
    rms = np.sqrt(np.mean(waveform ** 2))
    target_rms = 10 ** (-20 / 20)
    if rms > 0:
        waveform *= target_rms / rms
    return waveform, sample_rate

def extract_timestamps(timestamps, waveform, cache_dir, samplerate=16000, model_id="large"):
    """
    Extract precise timestamps from non-silent audio segments using whisperx.

    Args:
        timestamps (list): List of detected (start, end) time tuples for non-silent segments.
        waveform (np.ndarray): Normalized audio waveform.
        cache_dir (str): Cache directory (not used in whisperx, kept for compatibility).
        samplerate (int): Sample rate.
        model_id (str): Whisper model identifier (e.g., "large", "medium", etc.).

    Returns:
        list: List of refined (start, end) timestamp tuples.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_id, cache_dir)
    refined_timestamps = []

    for seg_start, seg_end in timestamps:
        segment = waveform[int(seg_start * samplerate):int(seg_end * samplerate)]
        # Transcribe the segment using whisperx
        result = whisperx.transcribe(model, segment, batch_size=1, fp16=(device == "cuda"))
        # Align the transcription result to improve timestamp accuracy
        aligned_result = whisperx.align(result["segments"], segment, model, device)
        for seg in aligned_result:
            if seg.get("start") is not None and seg.get("end") is not None and seg["start"] < seg["end"]:
                # Adjust timestamps relative to the entire waveform
                refined_timestamps.append((seg_start + seg["start"], seg_start + seg["end"]))
    return refined_timestamps

def save_slices(info, wav_output_dir):
    """
    Save sliced audio segments from the original audio file as WAV files.

    Args:
        info (list): List of tuples containing (audio file path, list of timestamps).
        wav_output_dir (str): Directory where sliced WAV files will be saved.
    """
    for idx, (wavfile, timestamps) in enumerate(info):
        waveform, sample_rate = torchaudio.load(wavfile)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        for (start, end) in timestamps:
            sliced_waveform = waveform[:, int(start * sample_rate):int(end * sample_rate)]
            out_path = os.path.join(wav_output_dir, f"{str(idx).zfill(5)}.wav")
            torchaudio.save(out_path, sliced_waveform, sample_rate)
        os.remove(wavfile)

def process_audio_files(input_folder, output_dir, cache_dir, model_id="large"):
    """
    Process all MP3 files in the specified folder by detecting speech segments and saving sliced audio.

    Args:
        input_folder (str): Directory containing MP3 files.
        output_dir (str): Directory where sliced WAV files will be saved.
        cache_dir (str): Cache directory (not used in whisperx, kept for compatibility).
        model_id (str): Whisper model identifier (e.g., "large", "medium", etc.).
    """
    info = []
    for wav_file in glob(os.path.join(input_folder, "*.mp3")):
        audio = torchaudio.load(wav_file)
        normalized_audio, sample_rate = audio_normalize(audio)
        timestamps = detect_non_silent(normalized_audio, sr=sample_rate)
        timestamps = extract_timestamps(timestamps, normalized_audio, cache_dir, samplerate=sample_rate, model_id=model_id)
        info.append((wav_file, timestamps))
    save_slices(info, output_dir)
