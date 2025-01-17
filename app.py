import yt_dlp
import whisper
import os
import torch

def download_audio(url, output_path="audio.mp3"):
    # Remove .mp3 extension as yt-dlp will add it automatically
    output_path = output_path.replace('.mp3', '')
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_path,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return True
    except Exception as e:
        print(f"Error downloading audio: {str(e)}")
        return False

def transcribe_audio(audio_path):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = whisper.load_model("base", device=device)
    
    # Transcribe with appropriate device settings
    result = model.transcribe(audio_path, language="en")
    return result["text"]

def main():
    try:
        # Get YouTube URL from user
        url = input("Enter YouTube URL: ")
        
        # Download audio
        print("Downloading audio...")
        audio_path = "audio.mp3"
        
        if not download_audio(url, audio_path):
            print("Failed to download audio. Please check the URL and try again.")
            return
            
        # Verify file exists
        if not os.path.exists(audio_path):
            print(f"Audio file not found at {audio_path}")
            return
            
        # Transcribe
        print("Transcribing...")
        transcription = transcribe_audio(audio_path)
        
        # Save transcription
        with open("transcription.txt", "w", encoding="utf-8") as f:
            f.write(transcription)
        
        print("Transcription completed and saved to 'transcription.txt'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        # Clean up audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)

if __name__ == "__main__":
    main()
