import os
import torch
import torchaudio
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
from glob import glob

def load_model(model_id, cache_dir):
    """
    Load the Whisper ASR model with given model ID and cache directory.

    Args:
        model_id (str): Hugging Face model ID.
        cache_dir (str): Directory to cache the model.

    Returns:
        pipeline: ASR pipeline for transcription.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16

    processor = WhisperProcessor.from_pretrained(model_id, torch_dtype=torch_dtype, cache_dir=cache_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch_dtype, cache_dir=cache_dir).to(device)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
        return_language=True
    )

    return pipe

def detect_non_silent(audio, sr=16000, silence_db=-40, min_silence_len_sec=1, min_gap_sec=1, min_duration_sec=0.05):
    """
    Detect non-silent segments in an audio array and merge short pauses.

    Args:
        audio (np.ndarray): Mono audio waveform.
        sr (int): Sample rate.
        silence_db (float): Threshold for silence detection.
        min_silence_len_sec (float): Minimum duration to consider as silence.
        min_gap_sec (float): Minimum gap between non-silent segments for merging.
        min_duration_sec (float): Minimum duration of final non-silent segments.

    Returns:
        List[Tuple[float, float]]: List of (start_time, end_time) for non-silent segments.
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
    Normalize an audio waveform to -20dB RMS.

    Args:
        audio (tuple): (waveform, sample_rate).

    Returns:
        tuple: Normalized waveform and sample rate.
    """
    waveform, sample_rate = audio
    waveform = waveform.squeeze().numpy()
    rms = np.sqrt(np.mean(waveform**2))
    target_rms = 10 ** (-20 / 20)
    if rms > 0:
        waveform *= target_rms / rms
    return waveform, sample_rate

def extract_timestamps(timestamps, waveform, cache_dir, samplerate=16000, model_id="waveletdeboshir/whisper-large-v3-no-numbers"):
    """
    Extract precise timestamps from non-silent audio segments.

    Args:
        timestamps (list): List of detected timestamps.
        waveform (np.ndarray): Normalized audio waveform.
        cache_dir (str): Model cache directory.
        samplerate (int): Sample rate.
        model_id (str): Whisper model ID.

    Returns:
        list: Refined timestamps.
    """
    pipe = load_model(model_id, cache_dir)
    refined_timestamps = []

    for start, end in timestamps:
        segment = waveform[int(start * samplerate):int(end * samplerate)]
        savetime, temp_timestamps = 0, []

        while len(segment) / samplerate >= 30 or savetime == 0:
            chunk = segment[:30 * samplerate]
            result = pipe(chunk, generate_kwargs={"num_beams": 1, "temperature": 0.0, "return_timestamps": True, "task": "transcribe"})
            if not result['chunks']:
                break

            for i in result['chunks']:
                if i['timestamp'] and i['timestamp'][0] is not None and i['timestamp'][1] is not None and i['timestamp'][0] < i['timestamp'][1] < 30:
                    temp_timestamps.append((i['timestamp'][0] + savetime, i['timestamp'][1] + savetime))
                    savetime = i['timestamp'][1]

            segment = segment[int(savetime * samplerate):]

        refined_timestamps += [(start + ts[0], start + ts[1]) for ts in temp_timestamps]

    return refined_timestamps

def transcribe_audio(timestamps, waveform, cache_dir, samplerate=16000, lang="english", model_id="waveletdeboshir/whisper-large-v3-no-numbers"):
    """
    Perform speech-to-text transcription on detected audio segments.

    Args:
        timestamps (list): List of timestamps.
        waveform (np.ndarray): Normalized audio waveform.
        cache_dir (str): Model cache directory.
        samplerate (int): Sample rate.
        lang (str): Target language.
        model_id (str): Whisper model ID.

    Returns:
        list: Transcribed text.
    """
    pipe = load_model(model_id, cache_dir)
    return [pipe(waveform[int(start * samplerate):int(end * samplerate)], generate_kwargs={"num_beams": 1, "temperature": 0.0, "return_timestamps": True, "task": "transcribe", "language": lang})['text'] for start, end in timestamps]

def save_slices(info, wav_output_dir, txt_output_dir):
    """
    Save sliced audio segments and corresponding transcriptions.

    Args:
        info (list): List of tuples (audio file, timestamps, text).
        wav_output_dir (str): Directory to save WAV files.
        txt_output_dir (str): Directory to save transcription text.
    """
    for idx, (wavfile, timestamps, texts) in enumerate(info):
        waveform, sample_rate = torchaudio.load(wavfile)
        waveform = waveform.mean(dim=0, keepdim=True) if waveform.shape[0] > 1 else waveform

        for (start, end), text in zip(timestamps, texts):
            sliced_waveform = waveform[:, int(start * sample_rate):int(end * sample_rate)]
            torchaudio.save(os.path.join(wav_output_dir, f"{str(idx).zfill(5)}.wav"), sliced_waveform, sample_rate)

            with open(os.path.join(txt_output_dir, f"{str(idx).zfill(5)}.txt"), "w", encoding="utf-8") as f:
                f.write(text)

        os.remove(wavfile)

def process_audio_files(input_folder, output_dir1, output_dir2, cache_dir, lang, model_id):
    """
    Process all audio files in a directory, detect speech, and save results.

    Args:
        input_folder (str): Folder containing MP3 files.
        output_dirs (tuple): (WAV output folder, TXT output folder).
        cache_dir (str): Model cache directory.
        lang (str): Target transcription language.
    """
    output_dirs = (output_dir1, output_dir2)
    info = []
    for wav_file in glob(os.path.join(input_folder, "*.mp3")):
        audio = torchaudio.load(wav_file)
        normalized_audio, _ = audio_normalize(audio)

        timestamps = detect_non_silent(normalized_audio)
        timestamps = extract_timestamps(timestamps, normalized_audio, cache_dir, model_id=model_id)
        texts = transcribe_audio(timestamps, normalized_audio, cache_dir, lang=lang, model_id=model_id)

        info.append((wav_file, timestamps, texts))

    save_slices(info, *output_dirs)
