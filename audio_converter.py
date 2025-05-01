import librosa
from pydub import AudioSegment
from pathlib import Path
from typing import Optional, Union
import logging
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioConverter:
    @staticmethod
    def convert_to_mp3(audio_path: Union[str, Path], output_dir: Optional[Union[str, Path]] = None) -> Path:
        """
        Convert an audio file to MP3 format using pydub.
        
        Args:
            audio_path (Union[str, Path]): Path to the input audio file
            output_dir (Optional[Union[str, Path]]): Directory to save converted file. If None, uses same directory as input.
            
        Returns:
            Path: Path to the converted MP3 file
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Determine output path
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{audio_path.stem}.mp3"
        else:
            output_path = audio_path.parent / f"{audio_path.stem}.mp3"
        
        try:
            # Load the audio file with pydub
            audio = AudioSegment.from_file(str(audio_path))
            
            # Export as MP3
            audio.export(output_path, format="mp3")
            logger.info(f"Successfully converted {audio_path} to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error during conversion: {str(e)}")
            raise RuntimeError(f"Failed to convert audio file: {str(e)}")
    
    @staticmethod
    def cleanup_file(file_path: Union[str, Path]) -> None:
        """
        Delete a converted audio file.
        
        Args:
            file_path (Union[str, Path]): Path to the file to delete
        """
        file_path = Path(file_path)
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Cleaned up converted file: {file_path}") 