
import streamlit as st
import os
import tempfile
import time
from pathlib import Path
from moviepy.editor import VideoFileClip, AudioFileClip
from gtts import gTTS
import whisper
import argostranslate.package
import argostranslate.translate

def recognize_speech_from_audio(audio_path, source_language):
    st.info(f"Transcribing audio in {source_language} using Whisper...")

    lang_map = {
        "en": "english",
        "bn": "bengali",
        "hi": "hindi",
        "te": "telugu"
    }
    whisper_lang = lang_map.get(source_language.lower(), "english")

    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language=whisper_lang)
    text = result["text"].strip()

    st.success("Speech recognition complete.")
    return text

def translate_text(text, source_language, destination_language):
    st.info(f"Translating text from {source_language} → {destination_language} using Argos Translate...")

    src = source_language.lower()
    dest = destination_language.lower()

    installed_packages = argostranslate.package.get_installed_packages()
    installed_codes = [f"{p.from_code}-{p.to_code}" for p in installed_packages]
    package_code = f"{src}-{dest}"

    if package_code not in installed_codes:
        st.warning(f"Installing Argos model for {src} → {dest} (first-time setup)...")
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        matching_packages = [p for p in available_packages if p.from_code == src and p.to_code == dest]

        if not matching_packages:
            st.error(f"No Argos model available for {src}->{dest}")
            return text

        download_path = matching_packages[0].download()
        argostranslate.package.install_from_path(download_path)

    translated_text = argostranslate.translate.translate(text, src, dest)
    st.success("Translation complete.")
    return translated_text

def process_and_dub_video(uploaded_file, source_lang, dest_lang):
    st.header("Processing Workflow")

    with tempfile.TemporaryDirectory() as temp_dir:
        original_video_path = os.path.join(temp_dir, "original_video.mp4")
        tts_audio_path = os.path.join(temp_dir, "tts_audio.mp3")
        final_video_path = os.path.join(temp_dir, "dubbed_video.mp4")

        with open(original_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Video saved temporarily: {uploaded_file.name}")
        
        video_clip = None
        audio_clip = None
        new_audio_clip = None
        final_clip = None

        try:
            video_clip = VideoFileClip(original_video_path)
            audio_clip = video_clip.audio

            original_audio_path = os.path.join(temp_dir, "original_audio.mp3")
            audio_clip.write_audiofile(original_audio_path, logger=None)
            st.success("Audio extracted from video.")

            transcribed_text = recognize_speech_from_audio(original_audio_path, source_lang)
            st.code(f"Original Transcript: {transcribed_text}", language="markdown")

            translated_text = translate_text(transcribed_text, source_lang, dest_lang)
            st.code(f"Translated Text: {translated_text}", language="markdown")

            st.info(f"Generating new voiceover in {dest_lang} using gTTS...")
            tts_lang_map = {"en": "en", "fr": "fr", "es": "es", "hi": "hi", "bn": "bn", "te": "te"}
            tts_lang = tts_lang_map.get(dest_lang, "en")

            tts = gTTS(text=translated_text, lang=tts_lang, slow=False)
            tts.save(tts_audio_path)
            st.success("New voiceover generated successfully.")

            new_audio_clip = AudioFileClip(tts_audio_path)

            if new_audio_clip.duration > video_clip.duration:
                new_audio_clip = new_audio_clip.subclip(0, video_clip.duration)

            final_clip = video_clip.set_audio(new_audio_clip)

            st.info("Merging new audio with video and exporting...")
            final_clip.write_videofile(
                final_video_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                logger=None
            )
            st.success("Dubbing process complete!")

            st.header("Dubbed Video Output")
            st.video(final_video_path)

            with open(final_video_path, 'rb') as file:
                st.download_button(
                    label="Download Dubbed Video",
                    data=file,
                    file_name=f"dubbed_{uploaded_file.name}",
                    mime="video/mp4",
                    type="primary"
                )

        except Exception as e:
            st.error(f"An error occurred during video processing: {e}")
            st.warning("Ensure ffmpeg is installed and required Python libraries are installed.")

        finally:
            for clip in (final_clip, new_audio_clip, audio_clip, video_clip):
                if clip is not None:
                    clip.close()

def main():
    st.set_page_config(page_title="Video Dubbing & Translation App", layout="wide")

    st.title("Vid_dub 48_2")
    st.markdown("""
    Upload a video, specify source & destination languages, and the app will automatically dub it.
    """)

    st.sidebar.header("Configuration")

    source_lang = st.sidebar.text_input("Source Language Code", value='en', max_chars=2).lower()
    dest_lang = st.sidebar.text_input("Destination Language Code", value='fr', max_chars=2).lower()

    uploaded_file = st.file_uploader("Choose a video file (MP4 recommended)", type=['mp4', 'mov', 'avi'])

    if uploaded_file and st.button("Start Dubbing Process", type="primary"):
        if source_lang == dest_lang:
            st.error("Source and destination languages must be different.")
        else:
            with st.spinner('Processing video...'):
                process_and_dub_video(uploaded_file, source_lang, dest_lang)

if __name__ == "__main__":
    main()

