import streamlit as st
import yt_dlp
import whisper
import os
import torch
import time

def download_audio(url, output_path="audio.mp3", ffmpeg_location=None):
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
    
    if ffmpeg_location:
        ydl_opts['ffmpeg_location'] = ffmpeg_location
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return f"{output_path}.mp3"

def transcribe_audio(audio_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base", device=device)
    result = model.transcribe(audio_path, language="en")
    return result["text"]

def get_video_info(url):
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                'thumbnail_url': info.get('thumbnail'),
                'title': info.get('title'),
                'duration': info.get('duration')
            }
    except Exception as e:
        st.error(f"Error fetching video info: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="YouTube Transcript", page_icon="🎯")
    st.title("YouTube Transcript")
    
    # Initialize session state
    if 'transcription' not in st.session_state:
        st.session_state.transcription = None
    if 'audio_path' not in st.session_state:
        st.session_state.audio_path = None
    if 'video_info' not in st.session_state:
        st.session_state.video_info = None
    
    url = st.text_input("Enter YouTube URL:")
    
    if url:
        # Show video preview
        video_info = get_video_info(url)
        if video_info:
            st.session_state.video_info = video_info
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(video_info['thumbnail_url'], use_container_width=True)
            with col2:
                st.subheader(video_info['title'])
                minutes = video_info['duration'] // 60
                seconds = video_info['duration'] % 60
                st.write(f"Duration: {minutes}:{seconds:02d}")
    
    if st.button("Process Video"):
        if url:
            try:
                # Download audio
                with st.spinner("Downloading audio..."):
                    ffmpeg_location = st.text_input("Enter FFmpeg location (optional):")
                    audio_path = download_audio(url, ffmpeg_location=ffmpeg_location)
                    st.session_state.audio_path = audio_path
                    st.success("Audio downloaded successfully!")
                    
                # Show audio player
                st.subheader("Audio Preview")
                st.audio(audio_path)
                
                # Add download button for audio
                with open(audio_path, 'rb') as audio_file:
                    st.download_button(
                        label="Download Audio",
                        data=audio_file,
                        file_name="audio.mp3",
                        mime="audio/mp3"
                    )
                
                # Transcribe
                with st.spinner("Transcribing... This may take a few minutes."):
                    transcription = transcribe_audio(audio_path)
                    st.session_state.transcription = transcription
                
                # Clean up audio file
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                
                st.success("Transcription completed!")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                if os.path.exists(audio_path):
                    os.remove(audio_path)
    
    # Display transcription
    if st.session_state.transcription:
        st.subheader("Transcription:")
        st.text_area("", st.session_state.transcription, height=300)
        
        st.download_button(
            label="Download Transcription",
            data=st.session_state.transcription,
            file_name="transcription.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
