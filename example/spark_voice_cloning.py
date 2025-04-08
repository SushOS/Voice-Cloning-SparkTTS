import torch
import soundfile as sf
import logging
import sys
import os
import time

# Append the project root to sys.path so that the cli module can be found.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)

from cli.SparkTTS import SparkTTS

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    # Define a sample text response for TTS.
    sample_text = "My name is Sushant."
    # Optionally, you can provide a transcript of the prompt audio.
    prompt_text = "Hello world, this is a sample text to speech test using Spark TTS."
    # Set the path to your prompt audio file.
    prompt_audio_path = os.path.join(current_dir, "/prompt_audios/prompt_audio1.wav")
    # Set the output path for the generated audio.
    output_file = os.path.join(current_dir, "/output_audios/output_cloned_" + str(time.time()) + ".wav")
    
    generate_speech_with_voice_cloning(
        text_response=sample_text,
        prompt_speech_path=prompt_audio_path,
        prompt_text=prompt_text,
        output_path=output_file
    )

if __name__ == "__main__":
    main()
