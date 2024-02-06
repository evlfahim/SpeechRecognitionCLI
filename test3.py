import sounddevice as sd
import numpy as np
from pydub import AudioSegment

def record_audio(duration=5, sample_rate=44100):
    print(f"Recording audio for {duration} seconds...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype=np.int16)
    sd.wait()
    return audio_data

def save_as_mp3(audio_data, output_file):
    audio = AudioSegment(
        audio_data.tobytes(),
        frame_rate=44100,
        sample_width=audio_data.dtype.itemsize,
        channels=2
    )
    audio.export(output_file, format="mp3")
    print(f"Audio saved as {output_file}")

def main():
    duration = int(input("Enter the duration of the recording in seconds: "))
    # output_file = input("Enter the name of the output MP3 file (without extension): ") + ".mp3"
    output_file = "audio.mp3"

    audio_data = record_audio(duration)
    save_as_mp3(audio_data, output_file)

if __name__ == "__main__":
    main()
