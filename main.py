from module.tools import batch_convert_to_wav, download_pretrained_models, batch_convert_wav_to_mp3
from module.whisper import process_audio_files
from module.msst import msst_for_main
from module.clustering import clustering_for_main
import os
import gradio as gr

# Set the current working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

with gr.Blocks() as demo:
    gr.Markdown("## AniTTS Builder-v3")
    
    # Define hidden textboxes for various folder paths
    video_folder = gr.Textbox(value="./data/video", interactive=False, visible=False)
    wav_folder = gr.Textbox(value="./data/audio_wav", interactive=False, visible=False)
    mp3_folder = gr.Textbox(value="./data/audio_mp3", interactive=False, visible=False)
    text_folder = gr.Textbox(value="./data/transcribe", interactive=False, visible=False)
    result_folder = gr.Textbox(value="./data/result", interactive=False, visible=False)
    whisper_cache_dir = gr.Textbox(value="./module/model/whisper", interactive=False, visible=False)
    embeddings_cache_dir = gr.Textbox(value="./module/model/redimmet", interactive=False, visible=False)
    
    # State variable to track button enabled/disabled status
    button_state = gr.State(value=True)

    with gr.Tabs():
        with gr.Tab("Remove BGM"):
            gr.Markdown("### Remove BGM from Audio")
            btn_convert_wav = gr.Button("Convert to WAV")
            btn_download_model = gr.Button("Download Transcribe Models")
            btn_msst_wav = gr.Button("Remove Wav BGM")

        with gr.Tab("Generate Timestamps"):
            gr.Markdown("### Transcribe Audio")
            btn_convert_mp3 = gr.Button("Convert to MP3")
            # Change default model ID from 'openai/whisper-large-v3' to 'large' for whisperx
            txt_model_id = gr.Textbox(label="Model ID", value="large")
            btn_transcribe = gr.Button("Generate Timestamps")

        with gr.Tab("Embedding & Clustering"):
            gr.Markdown("### Embeddings & Clustering")
            btn_clustering = gr.Button("Run Embeddings & Clustering")

    # List of all buttons for enabling/disabling during processing
    all_buttons = [btn_convert_wav, btn_download_model, btn_msst_wav, btn_convert_mp3, btn_transcribe, btn_clustering]

    # Function to disable all buttons
    def disable_all():
        return [gr.update(interactive=False) for _ in all_buttons] + [False]

    # Function to enable all buttons
    def enable_all():
        return [gr.update(interactive=True) for _ in all_buttons] + [True]

    # Chain for "Remove BGM" tab buttons
    btn_convert_wav.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(batch_convert_to_wav, inputs=[video_folder, wav_folder], outputs=[]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_download_model.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(download_pretrained_models, inputs=[], outputs=[]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_msst_wav.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(msst_for_main, inputs=[wav_folder], outputs=[]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    # Chain for "Generate Timestamps" tab buttons
    btn_convert_mp3.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(batch_convert_wav_to_mp3, inputs=[wav_folder, mp3_folder], outputs=[]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_transcribe.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(process_audio_files, inputs=[mp3_folder, wav_folder, whisper_cache_dir, txt_model_id], outputs=[]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    # Chain for "Embedding & Clustering" tab buttons
    btn_clustering.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(clustering_for_main, inputs=[wav_folder, result_folder, embeddings_cache_dir], outputs=[]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

# Launch the Gradio demo
demo.launch(server_name="0.0.0.0", server_port=7860)
