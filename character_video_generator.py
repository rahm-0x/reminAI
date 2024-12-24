from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from diffusers import StableDiffusionPipeline
import torchaudio
import os

# Paths for saving outputs
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize TTS models
tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
tts_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Initialize Image-to-Video model
animate_pipeline = StableDiffusionPipeline.from_pretrained("AnimateDiff/animate-model", torch_dtype="torch.float16")

# Function to convert text to speech
def text_to_speech(text, output_path):
    inputs = tts_processor(text=text, return_tensors="pt")
    speech = tts_model.generate_speech(inputs["input_ids"], speaker_embeddings=None)
    audio = tts_vocoder(speech)
    torchaudio.save(output_path, audio, sample_rate=16000)
    print(f"Audio saved to {output_path}")

# Function to animate image
def animate_image(image_path, audio_path, output_path):
    animation = animate_pipeline(image_path)
    # For simplicity, save video without syncing audio (initial version)
    animation.save(output_path)
    print(f"Animation saved to {output_path}")

def main():
    print("Welcome to the Character Video Generator!")
    
    # Input text for TTS
    text = input("Enter the text for your character to speak: ")
    audio_path = os.path.join(OUTPUT_DIR, "speech.wav")
    text_to_speech(text, audio_path)
    
    # Input image path
    image_path = input("Enter the path to your character's image (e.g., ./character.jpg): ")
    video_path = os.path.join(OUTPUT_DIR, "animated_video.mp4")
    animate_image(image_path, audio_path, video_path)

    print("Character video generation complete!")

if __name__ == "__main__":
    main()
