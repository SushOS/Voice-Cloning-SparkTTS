import torch
import soundfile as sf
import logging
import sys
import os
import time
import subprocess

# Append the project root to sys.path so that the cli module can be found.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)

from cli.SparkTTS import SparkTTS

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_wav(input_file, output_file=None):
    """Convert audio/video file to WAV format."""
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + ".wav"
    
    try:
        subprocess.run([
            'ffmpeg', '-i', input_file, 
            '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            output_file
        ], check=True)
        return output_file
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error("FFmpeg not found. Please install FFmpeg.")
        sys.exit(1)

def generate_speech_with_voice_cloning(
    text_response, # The text to convert to speech.
    model_dir=None, # Path to the SparkTTS model directory.
    prompt_speech_path=None, # Path to reference audio for voice cloning.
    prompt_text=None, # Transcript of the prompt audio.
    output_path=None, # Path to save the generated audio.
    temperature=0.8,  # Controls randomness: lower is more deterministic
    top_k=50,         # Limits token choices to top k options
    top_p=0.95        # Nucleus sampling probability threshold
):
    """
    Generate speech using voice cloning.
    
    Args:
        text_response (str): The text to convert to speech.
        model_dir (str): Path to the SparkTTS model directory.
            Defaults to the project's pretrained_models/Spark-TTS-0.5B.
        prompt_speech_path (str): Path to reference audio for voice cloning.
        prompt_text (str, optional): Transcript of the prompt audio.
        output_path (str): Path to save the generated audio.
        temperature (float): Sampling temperature.
        top_k (int): Top-k sampling parameter.
        top_p (float): Nucleus (top-p) sampling parameter.
        
    Returns:
        str: The path where the generated audio is saved.
    """
    # Set default model_dir if not provided.
    if model_dir is None:
        model_dir = os.path.join(project_root, "pretrained_models", "Spark-TTS-0.5B")
    config_path = os.path.join(model_dir, "config.yaml")
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at {config_path}. Please verify the model path!")
        sys.exit(1)
    
    # Set default prompt_speech_path if not provided.
    if prompt_speech_path is None:
        # Provide a default path for the prompt audio file.
        prompt_speech_path = os.path.join(current_dir, "prompt_audio.wav")
        if not os.path.exists(prompt_speech_path):
            logger.error(f"Prompt audio file not found at {prompt_speech_path}.")
            sys.exit(1)
    
    # Set a default output path if not provided.
    if output_path is None:
        output_path = os.path.join(current_dir, "output_cloned.wav")
    
    # Select device: use CUDA if available, otherwise CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Instantiate the SparkTTS model.
    model = SparkTTS(model_dir, device)
    
    logger.info("Generating speech using voice cloning...")
    with torch.no_grad():
        wav_tensor = model.inference(
            text=text_response,
            prompt_speech_path=prompt_speech_path,
            prompt_text=prompt_text,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    
    # The inference method may return a torch.Tensor or a NumPy array.
    if hasattr(wav_tensor, "cpu"):
        wav = wav_tensor.cpu().numpy()
    else:
        wav = wav_tensor
    
    # Save the generated waveform as a WAV file.
    sf.write(output_path, wav, samplerate=16000, format="WAV")
    logger.info(f"Speech generated and saved to {output_path}")
    
    return output_path

def main():
    sample_text = "Hello Satish Sanpal, I see you have an eye for luxury. Your dream property awaits!"
    prompt_text = "Hello fans of the game, this is Chiteshwar Pujara here. I welcome you to Dafa news."
    
    # Original video file
    original_file = os.path.join(current_dir, "prompt_audios", "pujara_trimmed.mov")
    # Convert to WAV if not already a WAV file
    if not original_file.lower().endswith('.wav'):
        wav_file = os.path.splitext(original_file)[0] + ".wav"
        prompt_audio_path = convert_to_wav(original_file, wav_file)
    else:
        prompt_audio_path = original_file
    
    # Fix output path
    output_dir = os.path.join(current_dir, "output_audios")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"output_cloned_{int(time.time())}.wav")
    
    generate_speech_with_voice_cloning(
        text_response=sample_text,
        prompt_speech_path=prompt_audio_path,
        prompt_text=prompt_text,
        output_path=output_file
    )
if __name__ == "__main__":
    main()