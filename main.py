from module.tools import batch_convert_to_wav, download_pretrained_models, batch_convert_wav_to_mp3
from module.whisper import process_audio_files
from module.msst import msst_for_main
from module.clustering import clustering_for_main
import os
import gradio as gr

os.chdir(os.path.dirname(os.path.abspath(__file__)))

with gr.Blocks() as demo:
    gr.Markdown("## AniTTS Builder-v3")
    video_folder = gr.Textbox(value="./data/video", interactive=False, visible=False)
    wav_folder = gr.Textbox(value="./data/audio_wav", interactive=False, visible=False)
    mp3_folder = gr.Textbox(value="./data/audio_mp3", interactive=False, visible=False)
    text_folder = gr.Textbox(value="./data/transcribe", interactive=False, visible=False)
    result_folder = gr.Textbox(value="./data/result", interactive=False, visible=False)
    whisper_cache_dir = gr.Textbox(value="./module/model/whisper", interactive=False, visible=False)
    embeddings_cache_dir = gr.Textbox(value="./module/model/redimmet", interactive=False, visible=False)
    # 버튼 활성화 상태 저장용 state 추가
    button_state = gr.State(value=True)

    with gr.Tabs():
        with gr.Tab("Remove BGM"):
            gr.Markdown("### Remove BGM from Audio")
            btn_convert_wav = gr.Button("Convert to WAV")
            btn_download_model = gr.Button("Download Transcribe Models")
            btn_msst_wav = gr.Button("Remove Wav BGM")

        with gr.Tab("Transcribe"):
            gr.Markdown("### Transcribe Audio")
            btn_convert_mp3 = gr.Button("Convert to MP3")
            txt_language = gr.Textbox(label="Language", value="english")
            txt_model_id = gr.Textbox(label="Model ID", value="openai/whisper-large-v3")
            btn_transcribe = gr.Button("Transcribe")

        with gr.Tab("Embedding & Clustering"):
            gr.Markdown("### Embeddings & Clustering")
            btn_clustering = gr.Button("Run Embeddings & Clustering")

    # 모든 버튼을 리스트에 저장
    all_buttons = [btn_convert_wav, btn_download_model, btn_msst_wav, btn_convert_mp3, btn_transcribe, btn_clustering]

    # 모든 버튼 비활성화 함수
    def disable_all():
        return [gr.update(interactive=False) for _ in all_buttons] + [False]

    # 모든 버튼 활성화 함수
    def enable_all():
        return [gr.update(interactive=True) for _ in all_buttons] + [True]

    # Remove BGM 탭 버튼 체인 구성
    btn_convert_wav.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(batch_convert_to_wav, inputs=[video_folder, wav_folder], outputs=[]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_download_model.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(download_pretrained_models, inputs=[], outputs=[]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_msst_wav.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(msst_for_main, inputs=[wav_folder], outputs=[]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    # Transcribe 탭 버튼 체인 구성
    btn_convert_mp3.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(batch_convert_wav_to_mp3, inputs=[wav_folder, mp3_folder], outputs=[]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_transcribe.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(process_audio_files, inputs=[mp3_folder, wav_folder, text_folder, whisper_cache_dir, txt_language, txt_model_id], outputs=[]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    # Embedding & Clustering 탭 버튼 체인 구성
    btn_clustering.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(clustering_for_main, inputs=[wav_folder, result_folder, embeddings_cache_dir], outputs=[]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

demo.launch(server_name="0.0.0.0", server_port=7860)
