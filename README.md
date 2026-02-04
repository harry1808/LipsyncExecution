# LipsyncExecution
This the a repo video dubbing with lipsync.
LipsyncExecution is a Python-based toolkit for automatic video dubbing with accurate lipsync. The system uses advanced AI models to translate, synthesize, and synchronize new speech with the mouth movements in source videos—creating seamless dubbed videos in any target language.

## Features

- **Automatic Speech Translation:** Translates source video audio to the desired target language.
- **Text-to-Speech (TTS) Synthesis:** Generates high-quality synthetic voices for the translation.
- **Lipsync Video Generation:** Precisely aligns mouth movements with the new audio track using AI/ML models.
- **Audio/Video Processing:** Uses FFmpeg and MoviePy for robust handling of input and output files.
- **Batch Processing:** Supports bulk processing for multiple videos.
- **Fault Tolerance:** Provides error handling for missing/corrupt files and ensures temporary resources are cleaned up.

## Project Structure

- `webapp/dubbing.py`: Main orchestration script for dubbing and lipsync. Handles audio extraction, translation, TTS, video lipsync, and merges outputs.
- `instance/wav2lip_assets/`: Contains models and dependencies for the Wav2Lip lipsync pipeline.
- `instance/outputs/`: Default directory for saving the processed videos.
- `instance/uploads/`: Where uploaded source materials are stored.
- `.gitignore`: Ignores large model/checkpoint/temp/output files and environments.

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [transformers (HuggingFace)](https://huggingface.co/docs/transformers/index)
- [MoviePy](https://zulko.github.io/moviepy/)
- [NumPy](https://numpy.org/)
- [Whisper (OpenAI)](https://github.com/openai/whisper)
- FFmpeg (should be available in PATH or under `ffmpeg/bin` directory in the project)
- Wav2Lip model weights

## Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/LipsyncExecution.git
    cd LipsyncExecution
    ```

2. **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv lipsyncenv
    source lipsyncenv/bin/activate  # On Windows: lipsyncenv\Scripts\activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download Wav2Lip model weights:**
    - Follow instructions in `instance/wav2lip_assets` or download the model as specified in code comments.

5. **Ensure FFmpeg is installed:**
    - [Download FFmpeg](https://ffmpeg.org/download.html) and add it to your system PATH, or place the binaries under the project's `ffmpeg/bin/` directory.

## Usage

You can run the dubbing pipeline (for example) using:

```bash
python webapp/dubbing.py --input input_video.mp4 --lang fr --output output_video.mp4
```

- `--input`: Path to input video file.
- `--lang`: Target language code (e.g., `fr` for French, `es` for Spanish).
- `--output`: Output file path for the dubbed and lipsynced video.

Refer to code comments in `webapp/dubbing.py` and `instance/wav2lip_assets/eBack/api/pipeline/wav2lip/inference.py` for advanced options and pipeline details.

## Notes & Troubleshooting

- Model files (*.pth, *.pt, etc.) are **not** included in the repository. Please download them before running lipsync.
- The system automatically checks audio and video files for integrity before merging.
- Large files and temp/artifacts are cleaned up as per `.gitignore`.
- For best results, use high-quality input videos and check translation outputs manually.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Credits

- [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) for the core lipsync methodology.
- [OpenAI Whisper](https://github.com/openai/whisper) for ASR transcription.
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) for sequence-to-sequence translation models.

For questions or contributions, please open an issue or pull request.

---

