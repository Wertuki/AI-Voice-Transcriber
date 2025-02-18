import openai
import os
import whisper
from dotenv import load_dotenv

# Load API Key
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

def transcribe_audio(file_path):
    """Transcribes audio to text using OpenAI Whisper API."""
    openai.api_key = API_KEY

    model = whisper.load_model("base")
    result = model.transcribe(file_path)

    return result["text"]

if __name__ == "__main__":
    audio_file = input("Enter the path to your audio file: ")
    transcript = transcribe_audio(audio_file)
    print("\nTranscription:\n", transcript)
