import gradio as gr
import whisper
from transformers import pipeline
# from transformers import AutoTokenizer
# from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf
import os
from TTS.api import TTS
import time
# Load models

whisper_model = whisper.load_model("base")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")
translator2 = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")
# tts_model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1")
# tts_tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)


def process_audio(audio):
    # Save recorded audio
    # print(audio)
    # filename = 'speaker.wav'
    # audio.save(filename)

    # Use the time to create a unique foldername
    foldername = str(int(time.time()))
    foldername = os.path.join("data/en-zh", foldername)
    os.makedirs(foldername, exist_ok=True)


    # Transcribe audio to text
    audio_data = whisper.load_audio(audio)

    sf.write(os.path.join(foldername, "original.wav"), audio_data, 16000)

    audio_data = whisper.pad_or_trim(audio_data)
    transcription = whisper_model.transcribe(audio_data)["text"]

    # Print the transcription
    print(f'Original texts: {transcription}')

    # write the transcription to a file
    with open(os.path.join(foldername, "original.txt"), "w") as f:
        f.write(transcription)

    # Translate the text to English
    translation = translator(transcription, max_length=512)[0]['translation_text']

    # Save the translated text
    with open(os.path.join(foldername,"translated_text.txt"), "w") as f:
        f.write(translation)

    # Generate audio from text
    # description = "A female speaker with a slightly low-pitched, quite monotone voice delivers her words at a slightly faster-than-average pace in a confined space with very clear audio."
    # input_ids = tts_tokenizer(description, return_tensors="pt").input_ids
    # prompt_input_ids = tts_tokenizer(translation, return_tensors="pt").input_ids
    # generation = tts_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    # audio_arr = generation.cpu().numpy().squeeze()

    wav = tts.tts(text=translation, speaker_wav=audio, language="en")


    # Save output audio
    output_path = os.path.join(foldername, "translated_audio.wav")
    # sf.write(output_path, audio_arr, tts_model.config.sampling_rate)
    sf.write(output_path, wav, 22050)

    return output_path, translation

def process_audio2(audio):
    foldername = str(int(time.time()))
    foldername = os.path.join("data/en-zh", foldername)
    os.makedirs(foldername, exist_ok=True)


    # Transcribe audio to text
    audio_data = whisper.load_audio(audio)

    sf.write(os.path.join(foldername, "original.wav"), audio_data, 16000)

    audio_data = whisper.pad_or_trim(audio_data)
    transcription = whisper_model.transcribe(audio_data)["text"]

    # Print the transcription
    print(f'Original texts: {transcription}')

    # write the transcription to a file
    with open(os.path.join(foldername, "original.txt"), "w") as f:
        f.write(transcription)

    # Translate the text to English
    translation = translator2(transcription, max_length=512)[0]['translation_text']

    # Save the translated text
    with open(os.path.join(foldername,"translated_text.txt"), "w") as f:
        f.write(translation)

    # Generate audio from text
    # description = "A female speaker with a slightly low-pitched, quite monotone voice delivers her words at a slightly faster-than-average pace in a confined space with very clear audio."
    # input_ids = tts_tokenizer(description, return_tensors="pt").input_ids
    # prompt_input_ids = tts_tokenizer(translation, return_tensors="pt").input_ids
    # generation = tts_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    # audio_arr = generation.cpu().numpy().squeeze()

    wav = tts.tts(text=translation, speaker_wav=audio, language="zh-cn")


    # Save output audio
    output_path = os.path.join(foldername, "translated_audio.wav")
    # sf.write(output_path, audio_arr, tts_model.config.sampling_rate)
    sf.write(output_path, wav, 22050)

    return output_path, translation


with gr.Blocks() as app:
    with gr.Row():
        record_btn = gr.Audio(sources=["microphone"], type="filepath", label="Record Audio (Chinese)")
        process_btn = gr.Button("Process Audio")
    with gr.Row():
        output_audio = gr.Audio(label="Output Audio (English)")
        output_text = gr.Textbox(label="Translated Text (English)")
        # download_link = gr.File(label="Download Audio")

    with gr.Row():
        record_btn2 = gr.Audio(sources=["microphone"], type="filepath", label="Record Audio (English)")
        process_btn2 = gr.Button("Process Audio")
    with gr.Row():
        output_audio2 = gr.Audio(label="Output Audio (Chinese)")
        output_text2 = gr.Textbox(label="Translated Text (Chinese)")
        # download_link = gr.File(label="Download Audio")


    process_btn.click(
        process_audio,
        inputs=[record_btn],
        outputs=[output_audio, output_text]
    )

    process_btn2.click(
        process_audio2,
        inputs=[record_btn2],
        outputs=[output_audio2, output_text2]
    )


app.launch()
