import torch
import soundfile as sf
import logging
import sys
import os

# # Optional: ignore missing triton module if not installed
# try:
#     import triton
# except ImportError:
#     pass

# Append the project root so that the cli module can be found.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)

from cli.SparkTTS import SparkTTS

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Predefined sample text for TTS
    sample_text = "Hello world, this is a sample text to speech test using Spark TTS."
    
    # Compute the absolute path to the pretrained model directory.
    model_dir = os.path.join(project_root, "pretrained_models", "Spark-TTS-0.5B")
    config_path = os.path.join(model_dir, "config.yaml")
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at {config_path}. Please verify the model path!")
        sys.exit(1)
    
    # Choose device: use CUDA if available, otherwise CPU.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Instantiate the SparkTTS model using the absolute model path.
    tts_model = SparkTTS(model_dir, device)
    
    logger.info("Starting TTS inference in controllable (text-only) mode...")
    
    # Use controllable TTS mode by providing voice control parameters.
    gender = "female"       # Options: "male" or "female"
    pitch = "moderate"      # Options: "very_low", "low", "moderate", "high", "very_high"
    speed = "moderate"      # Options: "very_low", "low", "moderate", "high", "very_high"
    
    # Run inference using the provided sample text.
    with torch.no_grad():
        wav_tensor = tts_model.inference(
            sample_text,
            prompt_speech_path=None,
            gender=gender,
            pitch=pitch,
            speed=speed
        )
    
    # Check if the returned object is a torch.Tensor or a numpy array.
    if hasattr(wav_tensor, "cpu"):
        wav = wav_tensor.cpu().numpy()
    else:
        wav = wav_tensor
    
    # Save the generated waveform as a WAV file.
    output_path = os.path.join(current_dir, "output.wav")
    sf.write(output_path, wav, samplerate=16000, format="WAV")
    logger.info(f"Audio saved at {output_path}")

if __name__ == "__main__":
    main()
