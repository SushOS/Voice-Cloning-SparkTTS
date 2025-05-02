# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Description:
    This script contains a collection of functions designed to handle various
    audio processing.
"""

import random
import soxr
import soundfile
import torch
import torchaudio
import numpy as np

from pathlib import Path
from typing import Tuple
from numpy.lib.stride_tricks import sliding_window_view

# Add to the top of sparktts/utils/audio.py
import subprocess
import os
import tempfile
import logging

logger = logging.getLogger(__name__)

def convert_to_wav(input_file, output_file=None, sample_rate=16000):
    """Convert audio/video file to WAV format using FFmpeg.
    
    Args:
        input_file (str): Path to input audio/video file
        output_file (str, optional): Path for output WAV file. If None, creates a temporary file.
        sample_rate (int): Target sample rate for the audio
        
    Returns:
        str: Path to the converted WAV file
    """
    if output_file is None:
        # Create a temporary file with .wav extension
        fd, output_file = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
    
    try:
        subprocess.run([
            'ffmpeg', '-i', input_file, 
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            '-ar', str(sample_rate),  # Target sample rate
            '-ac', '1',               # Mono
            '-y',                     # Overwrite output file
            output_file
        ], check=True, capture_output=True)
        
        # Verify the file exists and has content
        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            raise IOError(f"FFmpeg failed to create a valid WAV file at {output_file}")
            
        return output_file
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr.decode() if e.stderr else str(e)}")
        raise
    except FileNotFoundError:
        logger.error("FFmpeg not found. Please install FFmpeg on your system.")
        raise

def audio_volume_normalize(audio: np.ndarray, coeff: float = 0.2) -> np.ndarray:
    """
    Normalize the volume of an audio signal.

    Parameters:
        audio (numpy array): Input audio signal array.
        coeff (float): Target coefficient for normalization, default is 0.2.

    Returns:
        numpy array: The volume-normalized audio signal.
    """
    # Sort the absolute values of the audio signal
    temp = np.sort(np.abs(audio))

    # If the maximum value is less than 0.1, scale the array to have a maximum of 0.1
    if temp[-1] < 0.1:
        scaling_factor = max(
            temp[-1], 1e-3
        )  # Prevent division by zero with a small constant
        audio = audio / scaling_factor * 0.1

    # Filter out values less than 0.01 from temp
    temp = temp[temp > 0.01]
    L = temp.shape[0]  # Length of the filtered array

    # If there are fewer than or equal to 10 significant values, return the audio without further processing
    if L <= 10:
        return audio

    # Compute the average of the top 10% to 1% of values in temp
    volume = np.mean(temp[int(0.9 * L) : int(0.99 * L)])

    # Normalize the audio to the target coefficient level, clamping the scale factor between 0.1 and 10
    audio = audio * np.clip(coeff / volume, a_min=0.1, a_max=10)

    # Ensure the maximum absolute value in the audio does not exceed 1
    max_value = np.max(np.abs(audio))
    if max_value > 1:
        audio = audio / max_value

    return audio


def load_audio(
    adfile: Path,
    sampling_rate: int = None,
    length: int = None,
    volume_normalize: bool = False,
    segment_duration: int = None,
    min_length: int = 160  # Minimum audio length (10ms at 16kHz)
) -> np.ndarray:
    """Load audio file with target sampling rate and length.
    Supports automatic conversion of non-WAV formats using FFmpeg.
    
    Args:
        adfile (Path): Path to audio file.
        sampling_rate (int, optional): Target sampling rate. Defaults to None.
        length (int, optional): Target audio length. Defaults to None.
        volume_normalize (bool, optional): Whether to perform volume normalization. Defaults to False.
        segment_duration (int): Random select a segment with duration. Defaults to None.
        min_length (int): Minimum audio length (in samples) to prevent zero-size inputs
        
    Returns:
        audio (np.ndarray): Audio array
    """
    # Check if file is WAV, if not, convert it
    adfile_str = str(adfile)
    temp_file = None
    
    if not adfile_str.lower().endswith('.wav'):
        try:
            temp_file = convert_to_wav(adfile_str, sample_rate=sampling_rate or 16000)
            adfile_str = temp_file
        except Exception as e:
            logger.error(f"Error converting audio: {e}")
            raise
    
    try:
        audio, sr = soundfile.read(adfile_str)
        if len(audio.shape) > 1:
            audio = audio[:, 0]  # Convert stereo to mono by taking first channel

        if sampling_rate is not None and sr != sampling_rate:
            audio = soxr.resample(audio, sr, sampling_rate, quality="VHQ")
            sr = sampling_rate

        if segment_duration is not None:
            seg_length = int(sr * segment_duration)
            audio = random_select_audio_segment(audio, seg_length)

        # Audio volume normalize
        if volume_normalize:
            audio = audio_volume_normalize(audio)
        
        # Ensure minimum audio length to prevent zero-size errors in models
        if len(audio) < min_length:
            audio = np.pad(audio, (0, min_length - len(audio)))
        
        # Check the audio length
        if length is not None:
            assert abs(audio.shape[0] - length) < 1000
            if audio.shape[0] > length:
                audio = audio[:length]
            else:
                audio = np.pad(audio, (0, int(length - audio.shape[0])))
                
        # Detect and handle if audio is completely silent
        if np.all(np.abs(audio) < 1e-6):
            logger.warning("Audio contains only silence, adding small noise")
            audio = audio + np.random.normal(0, 1e-6, audio.shape)
            
        return audio
    
    finally:
        # Clean up temporary file if created
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass



# def load_audio(
#     adfile: Path,
#     sampling_rate: int = None,
#     length: int = None,
#     volume_normalize: bool = False,
#     segment_duration: int = None,
# ) -> np.ndarray:
#     r"""Load audio file with target sampling rate and lsength

#     Args:
#         adfile (Path): path to audio file.
#         sampling_rate (int, optional): target sampling rate. Defaults to None.
#         length (int, optional): target audio length. Defaults to None.
#         volume_normalize (bool, optional): whether perform volume normalization. Defaults to False.
#         segment_duration (int): random select a segment with duration of {segment_duration}s.
#                                 Defualt to None which means the whole audio will be used.

#     Returns:
#         audio (np.ndarray): audio
#     """

#     audio, sr = soundfile.read(adfile)
#     if len(audio.shape) > 1:
#         audio = audio[:, 0]

#     if sampling_rate is not None and sr != sampling_rate:
#         audio = soxr.resample(audio, sr, sampling_rate, quality="VHQ")
#         sr = sampling_rate

#     if segment_duration is not None:
#         seg_length = int(sr * segment_duration)
#         audio = random_select_audio_segment(audio, seg_length)

#     # Audio volume normalize
#     if volume_normalize:
#         audio = audio_volume_normalize(audio)
#     # check the audio length
#     if length is not None:
#         assert abs(audio.shape[0] - length) < 1000
#         if audio.shape[0] > length:
#             audio = audio[:length]
#         else:
#             audio = np.pad(audio, (0, int(length - audio.shape[0])))
#     return audio


def random_select_audio_segment(audio: np.ndarray, length: int) -> np.ndarray:
    """get an audio segment given the length

    Args:
        audio (np.ndarray):
        length (int): audio length = sampling_rate * duration
    """
    if audio.shape[0] < length:
        audio = np.pad(audio, (0, int(length - audio.shape[0])))
    start_index = random.randint(0, audio.shape[0] - length)
    end_index = int(start_index + length)

    return audio[start_index:end_index]


def audio_highpass_filter(audio, sample_rate, highpass_cutoff_freq):
    """apply highpass fileter to audio

    Args:
        audio (np.ndarray):
        sample_rate (ind):
        highpass_cutoff_freq (int):
    """

    audio = torchaudio.functional.highpass_biquad(
        torch.from_numpy(audio), sample_rate, cutoff_freq=highpass_cutoff_freq
    )
    return audio.numpy()


def stft(
    x: torch.Tensor,
    fft_size: int,
    hop_size: int,
    win_length: int,
    window: str,
    use_complex: bool = False,
) -> torch.Tensor:
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """

    x_stft = torch.stft(
        x, fft_size, hop_size, win_length, window.to(x.device), return_complex=True
    )

    # clamp is needed to avoid nan or inf
    if not use_complex:
        return torch.sqrt(
            torch.clamp(x_stft.real**2 + x_stft.imag**2, min=1e-7, max=1e3)
        ).transpose(2, 1)
    else:
        res = torch.cat([x_stft.real.unsqueeze(1), x_stft.imag.unsqueeze(1)], dim=1)
        res = res.transpose(2, 3)  # [B, 2, T, F]
        return res


def detect_speech_boundaries(
    wav: np.ndarray,
    sample_rate: int,
    window_duration: float = 0.1,
    energy_threshold: float = 0.01,
    margin_factor: int = 2
) -> Tuple[int, int]:
    """Detect the start and end points of speech in an audio signal using RMS energy.
    
    Args:
        wav: Input audio signal array with values in [-1, 1]
        sample_rate: Audio sample rate in Hz
        window_duration: Duration of detection window in seconds
        energy_threshold: RMS energy threshold for speech detection
        margin_factor: Factor to determine extra margin around detected boundaries
        
    Returns:
        tuple: (start_index, end_index) of speech segment
        
    Raises:
        ValueError: If the audio contains only silence
    """
    window_size = int(window_duration * sample_rate)
    margin = margin_factor * window_size
    step_size = window_size // 10
    
    # Create sliding windows using stride tricks to avoid loops
    windows = sliding_window_view(wav, window_size)[::step_size]
    
    # Calculate RMS energy for each window
    energy = np.sqrt(np.mean(windows ** 2, axis=1))
    speech_mask = energy >= energy_threshold
    
    if not np.any(speech_mask):
        raise ValueError("No speech detected in audio (only silence)")
    
    start = max(0, np.argmax(speech_mask) * step_size - margin)
    end = min(len(wav), (len(speech_mask) - 1 - np.argmax(speech_mask[::-1])) * step_size + margin)
    
    return start, end


def remove_silence_on_both_ends(
    wav: np.ndarray,
    sample_rate: int,
    window_duration: float = 0.1,
    volume_threshold: float = 0.01
) -> np.ndarray:
    """Remove silence from both ends of an audio signal.
    
    Args:
        wav: Input audio signal array
        sample_rate: Audio sample rate in Hz
        window_duration: Duration of detection window in seconds
        volume_threshold: Amplitude threshold for silence detection
        
    Returns:
        np.ndarray: Audio signal with silence removed from both ends
        
    Raises:
        ValueError: If the audio contains only silence
    """
    start, end = detect_speech_boundaries(
        wav,
        sample_rate,
        window_duration,
        volume_threshold
    )
    return wav[start:end]



def hertz_to_mel(pitch: float) -> float:
    """
    Converts a frequency from the Hertz scale to the Mel scale.

    Parameters:
    - pitch: float or ndarray
        Frequency in Hertz.

    Returns:
    - mel: float or ndarray
        Frequency in Mel scale.
    """
    mel = 2595 * np.log10(1 + pitch / 700)
    return mel