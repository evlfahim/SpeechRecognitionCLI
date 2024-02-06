import sounddevice as sd
import numpy as np
import queue
import sys
import threading
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def record_audio(q, duration, fs):
    print("Recording started. Press Enter to stop recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    q.put(recording)
    
def summarize(text):
    from transformers import pipeline

    summarizer = pipeline("summarization", model="Falconsai/text_summarization")

    ARTICLE = text
    print(summarizer(ARTICLE, max_length=100, min_length=10, do_sample=False))

def main():
    # Select an audio file and read it:
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    audio_sample = ds[0]["audio"]
    sampling_rate = audio_sample["sampling_rate"]

    # Load the Whisper model in Hugging Face format:
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

    # Parameters for audio recording
    duration = 10  # seconds
    fs = sampling_rate

    # Record audio in a separate thread
    q = queue.Queue()
    recording_thread = threading.Thread(target=record_audio, args=(q, duration, fs))
    recording_thread.start()

    input("")

    # Wait for the user to press Enter
    recording_thread.join()

    # Get the recorded audio
    recording = q.get()

    # Use the model and processor to transcribe the audio:
    input_features = processor(
        recording.ravel(), sampling_rate=fs, return_tensors="pt"
    ).input_features

    # Generate token ids
    predicted_ids = model.generate(input_features)

    # Decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    print("Transcription:", transcription[0])
    summarize(transcription[0])
    
    
    # ARTICLE = """ 
    # Hugging Face: Revolutionizing Natural Language Processing
    # Introduction
    # In the rapidly evolving field of Natural Language Processing (NLP), Hugging Face has emerged as a prominent and innovative force. This article will explore the story and significance of Hugging Face, a company that has made remarkable contributions to NLP and AI as a whole. From its inception to its role in democratizing AI, Hugging Face has left an indelible mark on the industry.
    # The Birth of Hugging Face
    # Hugging Face was founded in 2016 by Clément Delangue, Julien Chaumond, and Thomas Wolf. The name "Hugging Face" was chosen to reflect the company's mission of making AI models more accessible and friendly to humans, much like a comforting hug. Initially, they began as a chatbot company but later shifted their focus to NLP, driven by their belief in the transformative potential of this technology.
    # Transformative Innovations
    # Hugging Face is best known for its open-source contributions, particularly the "Transformers" library. This library has become the de facto standard for NLP and enables researchers, developers, and organizations to easily access and utilize state-of-the-art pre-trained language models, such as BERT, GPT-3, and more. These models have countless applications, from chatbots and virtual assistants to language translation and sentiment analysis.
    # Key Contributions:
    # 1. **Transformers Library:** The Transformers library provides a unified interface for more than 50 pre-trained models, simplifying the development of NLP applications. It allows users to fine-tune these models for specific tasks, making it accessible to a wider audience.
    # 2. **Model Hub:** Hugging Face's Model Hub is a treasure trove of pre-trained models, making it simple for anyone to access, experiment with, and fine-tune models. Researchers and developers around the world can collaborate and share their models through this platform.
    # 3. **Hugging Face Transformers Community:** Hugging Face has fostered a vibrant online community where developers, researchers, and AI enthusiasts can share their knowledge, code, and insights. This collaborative spirit has accelerated the growth of NLP.
    # Democratizing AI
    # Hugging Face's most significant impact has been the democratization of AI and NLP. Their commitment to open-source development has made powerful AI models accessible to individuals, startups, and established organizations. This approach contrasts with the traditional proprietary AI model market, which often limits access to those with substantial resources.
    # By providing open-source models and tools, Hugging Face has empowered a diverse array of users to innovate and create their own NLP applications. This shift has fostered inclusivity, allowing a broader range of voices to contribute to AI research and development.
    # Industry Adoption
    # The success and impact of Hugging Face are evident in its widespread adoption. Numerous companies and institutions, from startups to tech giants, leverage Hugging Face's technology for their AI applications. This includes industries as varied as healthcare, finance, and entertainment, showcasing the versatility of NLP and Hugging Face's contributions.
    # Future Directions
    # Hugging Face's journey is far from over. As of my last knowledge update in September 2021, the company was actively pursuing research into ethical AI, bias reduction in models, and more. Given their track record of innovation and commitment to the AI community, it is likely that they will continue to lead in ethical AI development and promote responsible use of NLP technologies.
    # Conclusion
    # Hugging Face's story is one of transformation, collaboration, and empowerment. Their open-source contributions have reshaped the NLP landscape and democratized access to AI. As they continue to push the boundaries of AI research, we can expect Hugging Face to remain at the forefront of innovation, contributing to a more inclusive and ethical AI future. Their journey reminds us that the power of open-source collaboration can lead to groundbreaking advancements in technology and bring AI within the reach of many.
    # """
    # summarize(ARTICLE)

if __name__ == "__main__":
    main()
