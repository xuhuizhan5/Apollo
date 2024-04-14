# Apollo Audio Generation and Recording Demo
## Author: Xuhui Zhan, Siyu Yang

This project repository contains scripts for generating and recording audio using advanced machine learning models, including transformer-based models for text-to-speech (TTS) synthesis. [1]

## 1. Background Information

### 1.1 Overview
The primary objective of this project is to develop a Text-to-Speech (TTS) system that can learn and mimic the unique vocal characteristics of an individual's speaking voice and use this model to synthesize singing audio. This technology aims to bridge the gap between conventional speech synthesis and the generation of musically expressive vocal tracks, which can be particularly useful in applications such as personalized music creation, virtual avatars, and assistive technologies for artists with vocal limitations.

### 1.2 Challenges

Voice Modeling Accuracy: Capturing the nuanced vocal attributes of a person's voice, such as pitch, timbre, and dynamics, which are crucial for realistic and personalized singing synthesis.

Data Requirement: Collecting sufficient and high-quality recordings of a person's voice for training the TTS model without introducing significant noise or distortion that could degrade the quality of the synthesized singing.
Melody and Lyrics Synchronization: Ensuring that the synthesized singing aligns accurately with the intended melody and rhythm of the song, which involves complex timing and prosody adjustments.

Emotional Expression: Integrating emotional expressions into the synthesized singing voice, which is essential for delivering a performance that feels engaging and true to the original singer's intent.

Performance and Scalability: Optimizing the system to handle the extensive computational requirements of training and running deep learning models for TTS while ensuring that the system can scale to accommodate multiple users or different vocal styles.

## 2. Method

### 2.1 Text-to-Speech (TTS)

![image](https://github.com/xuhuizhan5/Apollo/assets/142248146/ea511fe6-a076-4fc0-bef6-2380335aa641)

Text-to-Speech (TTS) technology converts written text into spoken voice output. TTS systems are often used in applications like virtual assistants, navigation systems, and accessibility tools for visually impaired users. The TTS process involves two main steps: text analysis, where the text is converted into a linguistic structure, and sound generation, where this structure is used to synthesize the spoken output. Modern TTS systems often use deep learning and transformer models, which allow for more natural and expressive voice synthesis compared to traditional methods.

### 2.2 Transformers in Audio Processing
Transformers are a type of deep learning model that have revolutionized many areas of machine learning, particularly natural language processing (NLP). Their ability to handle sequences of data and their scalability have also made them increasingly popular in audio processing tasks. In this repository, we utilize transformer models for generating speech from text. These models are capable of capturing the nuances of human speech, such as intonation and rhythm, more effectively than prior techniques.

## 3. TTS Psudocode

![Screenshot 2024-04-13 141923](https://github.com/xuhuizhan5/Apollo/assets/142248146/47239bca-7e09-434f-8e93-b95573ee7e6c)

## 4. Project Demo

Develop a robust model: Create a deep learning-based TTS model capable of learning detailed vocal characteristics from spoken voice inputs.
```
preload_models()

# generate audio from text
text_prompt = """
     Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
     But I also have other interests such as playing tic tac toe.
"""
audio_array = generate_audio(text_prompt)

# save audio to disk
write_wav("bark_generation.wav", SAMPLE_RATE, audio_array)
  
# play text in notebook
Audio(audio_array, rate=SAMPLE_RATE)
```

Implement voice conversion: Convert the learned vocal features to synthesize singing, adjusting for the required pitch and tempo of the song.
```
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

# generate speech by cloning a voice using default settings
# tts.tts_to_file(text="Lucas's presenting issues include difficulties falling asleep, growing reluctance to attend school",
#                 file_path="output.wav",
#                 speaker_wav="speaker.wav",
#                 language="en")
wav = tts.tts(text="原谅太阳，原谅我", speaker_wav="speaker.wav", language="zh-cn")

Audio(wav, rate=22050)
```

Ensure high fidelity and naturalness: Produce high-quality singing audio that faithfully reproduces the vocal style and nuances of the original speaker's voice.
```
fs = 44100  # Sample rate
channels = 1  # Mono recording
recording = np.array([])  # Placeholder for the recorded data
is_recording = False  # Flag to control recording

def record_audio(indata, frames, time, status):
    global recording
    recording = np.append(recording, indata.copy())

def update_recording_time():
    global is_recording
    start_time = time.time()
    while is_recording:
        elapsed_time = time.time() - start_time
        time_label.value = f"Recording... {elapsed_time:.2f} seconds"
        time.sleep(0.1)  # Update every 100ms

def start_recording(button):
    global is_recording, recording
    recording = np.array([])  # Reset recording
    is_recording = True
    print("Recording started...")
    # Start recording in a separate thread to avoid blocking
    threading.Thread(target=lambda: sd.InputStream(callback=record_audio, channels=channels, samplerate=fs).start()).start()
    # Update recording time in a separate thread
    threading.Thread(target=update_recording_time).start()

def stop_recording(button):
    global is_recording
    is_recording = False
    sd.stop()
    print("Recording stopped.")
    time_label.value = "Recording stopped."
    # Normalize to 16-bit range and save
    norm_audio = np.int16(recording / np.max(np.abs(recording)) * 32767)
    filename = 'speaker.wav'
    write(filename, fs, norm_audio)
    print(f"Audio saved as {filename}")

# Create buttons and label
start_button = widgets.Button(description="Start Recording")
stop_button = widgets.Button(description="Stop Recording")
time_label = widgets.Label(value="Press 'Start Recording' to begin")

# Bind the buttons
start_button.on_click(start_recording)
stop_button.on_click(stop_recording)

# Display widgets
display(start_button, stop_button, time_label)
```
(Link to demo streamlit/gradio)
## 5. Future Development

### 5.1 Advanced Voice Modeling Techniques

Emotional Expression: Incorporate models that can analyze and replicate the emotional state of the speaker, allowing the synthesized singing to convey more complex emotions.

Style Transfer: Develop capabilities to not only mimic the voice but also adapt different singing styles, enabling users to choose styles ranging from classical to pop or jazz.

### 5.2 Personalization and User Profiles

Develop a user profile system where individual voice preferences, styles, and settings can be saved and recalled for personalized experiences. This could also include adaptive learning from user feedback to continuously improve the voice output.

### 5.3 Integration with Music Composition Tools

Create interfaces that allow seamless integration with digital audio workstations (DAWs) and music composition software, enabling musicians and producers to directly utilize the TTS system in their creative processes.

## 6. Reference List

[1] G. Eren (2021), A deep learning toolkit for Text-to-Speech, battle-tested in research and production, retrieved from https://www.coqui.ai
