# Audio Transcription with Whisper

This is a Python application that uses the Whisper model from OpenAI to transcribe audio files. It uses the Hugging Face Transformers library to load and run the Whisper model.

## Features

- Transcribes audio files using Whisper
- Supports various Whisper model sizes
- Provides timestamps for transcription chunks
- Automatically uses GPU if available
- Command-line interface for easy use

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python transcriber.py path/to/your/audio/file.mp3
```

Specify a different Whisper model:
```bash
python transcriber.py path/to/your/audio/file.mp3 --model openai/whisper-medium
```

Available Whisper models:
- openai/whisper-tiny
- openai/whisper-base
- openai/whisper-small
- openai/whisper-medium
- openai/whisper-large
- openai/whisper-large-v2

## Output

The application will output:
1. The complete transcription text
2. Timestamped chunks of the transcription

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster processing)
- Sufficient RAM (at least 8GB recommended)

## Notes

- The first run will download the specified Whisper model, which can be several GB in size
- Processing time depends on the audio length and your hardware
- The application will automatically use GPU if available, falling back to CPU if not 