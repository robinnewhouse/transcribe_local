import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Optional, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioTranscriber:
    def __init__(self, model_name: str = "openai/whisper-large-v2"):
        """
        Initialize the AudioTranscriber with a specific Whisper model.
        
        Args:
            model_name (str): Name of the Whisper model to use
        """
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Using dtype: {self.torch_dtype}")
        
        # Load model and processor
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        self.model.to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Create pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
    
    def transcribe_file(self, audio_path: Union[str, Path]) -> dict:
        """
        Transcribe an audio file.
        
        Args:
            audio_path (Union[str, Path]): Path to the audio file
            
        Returns:
            dict: Transcription result with text and timestamps
        """
        try:
            # Load audio file
            audio_path = Path(audio_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            logger.info(f"Transcribing file: {audio_path}")
            
            # Transcribe
            result = self.pipe(str(audio_path))
            
            return result
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            raise

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Transcribe audio files using Whisper")
    parser.add_argument("audio_file", help="Path to the audio file to transcribe")
    parser.add_argument("--model", default="openai/whisper-large-v2", 
                       help="Whisper model to use (default: openai/whisper-large-v2)")
    
    args = parser.parse_args()
    
    try:
        transcriber = AudioTranscriber(model_name=args.model)
        result = transcriber.transcribe_file(args.audio_file)
        
        print("\nTranscription Result:")
        print("=" * 50)
        print(result["text"])
        
        if "chunks" in result:
            print("\nTimestamps:")
            print("=" * 50)
            for chunk in result["chunks"]:
                print(f"[{chunk['timestamp'][0]:.2f} - {chunk['timestamp'][1]:.2f}] {chunk['text']}")
                
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 