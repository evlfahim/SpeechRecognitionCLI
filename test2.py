import sounddevice as sd
import numpy as np
from pydub import AudioSegment
import queue
import keyboard
import subprocess

def record_audio(queue, sample_rate=44100):
    print("Recording audio. Press Enter to stop...")
    audio_data = []
    
    def callback(indata, frames, time, status):
        if status:
            print(status)
        queue.put(indata.copy())

    with sd.InputStream(callback=callback, channels=2, samplerate=sample_rate, dtype=np.int16):
        keyboard.wait("enter")
    
    while not queue.empty():
        audio_data.append(queue.get())

    return np.vstack(audio_data)

def save_as_mp3(audio_data, output_file, sample_rate=44100):
    audio = AudioSegment(
        audio_data.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio_data.dtype.itemsize,
        channels=2
    )
    audio.export(output_file, format="mp3")
    print(f"Audio saved as {output_file}")
    
def transcribe_audio(audio_file):
    from transformers import pipeline
    whisper_transcriber = pipeline(task="automatic-speech-recognition", model="openai/whisper-tiny.en")
    audio_text = whisper_transcriber(audio_file)
    return audio_text[0]['sentence']

def main():
    output_file = "audio.mp3"
    audio_queue = queue.Queue()

    audio_data = record_audio(audio_queue)
    save_as_mp3(audio_data, output_file)
    
    transcript = transcribe_audio(output_file)
    print("Transcription:")
    print(transcript)

if __name__ == "__main__":
    main()
