import numpy as np
import wave
from IPython.display import Audio


def save_and_play_signal(
    signal, duration, export_path=None, sr=44100, original_sample_count=None
):
    """
    Resample a given signal, perform inverse FFT (for demonstration),
    and optionally save to WAV and return an Audio player.

    Parameters:
    -----------
    signal : np.ndarray
        Time-domain signal samples.
    duration : float
        Duration of the signal in seconds.
    export_path : str or None
        Path to export the WAV file. If None, no file is saved.
    sr : int
        Target sample rate for the audio file/playback.
    original_sample_count : int or None
        Original number of samples (for resampling interpolation).
        If None, uses len(signal).

    Returns:
    --------
    Audio object for inline playback in Jupyter.
    """
    if original_sample_count is None:
        original_sample_count = len(signal)

    # Create original time axis
    t = np.linspace(0, duration, original_sample_count)

    # FFT and inverse FFT
    spectrum = np.fft.fft(signal)
    reconstructed_signal = np.fft.ifft(spectrum).real

    # Resample to match target sample rate
    resampled_t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    resampled_signal = np.interp(resampled_t, t, reconstructed_signal)

    # Normalize to 16-bit
    resampled_signal_int16 = np.int16(
        resampled_signal / np.max(np.abs(resampled_signal)) * 32767
    )

    # Save as WAV if export_path is given
    if export_path:
        with wave.open(export_path, "w") as f:
            f.setnchannels(1)  # mono
            f.setsampwidth(2)  # 16 bits
            f.setframerate(sr)
            f.writeframes(resampled_signal_int16.tobytes())

    # Return audio player from NumPy array
    return Audio(resampled_signal_int16, rate=sr)
