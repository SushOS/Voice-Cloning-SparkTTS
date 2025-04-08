from huggingface_hub import snapshot_download
import os

# Create the directory if it doesn't exist
os.makedirs("pretrained_models/Spark-TTS-0.5B", exist_ok=True)

# Download the model
print("Downloading Spark-TTS model...")
snapshot_download(
    "SparkAudio/Spark-TTS-0.5B",
    local_dir="pretrained_models/Spark-TTS-0.5B",
    local_dir_use_symlinks=False
)
print("Download complete!") 