import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from IPython.display import Audio, display
import librosa
import librosa.display


def record_audio_and_plot(sr=22050, duration=4.0, outfile=None, n_fft=2048, hop_length=512, n_mels=128, n_mfcc=13):
    """
    Record audio, save it, and generate waveform, FFT, STFT, Mel spectrogram, and MFCC visualizations.

    Parameters:
    -----------
    sr : int
        Sample rate for recording (Hz).
    duration : float
        Duration of the recording in seconds.
    outfile : str
        Path to save the recorded audio file.
    n_fft : int
        Number of FFT components for STFT.
    hop_length : int
        Number of samples between successive frames.
    n_mels : int
        Number of Mel bands for the Mel spectrogram.
    n_mfcc : int
        Number of MFCCs to compute.

    Returns:
    --------
    None
    """
    print(f"Recording {duration} seconds at {sr} Hz...")

    # Record from default microphone
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # wait until recording is finished
    recording = recording.flatten()  # make 1D

    # Save to WAV
    if outfile:
        sf.write(outfile, recording, sr)
        print(f"Saved recording to {outfile}")

    # Play audio (in notebooks this will display a player)
    try:
        print("Audio player (notebooks will show a widget)...")
        display(Audio(data=recording, rate=sr))
    except Exception:
        print("If running outside a notebook, use an audio player to open", outfile)

    # Time-domain waveform
    t = np.linspace(0, duration, len(recording), endpoint=False)
    plt.figure(figsize=(10, 3))
    plt.plot(t, recording, linewidth=0.6)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Waveform (time domain)")
    plt.tight_layout()
    plt.show()

    # Fourier Transform (overall)
    N = len(recording)
    fft_vals = np.fft.fft(recording)
    fft_freqs = np.fft.fftfreq(N, 1.0/sr)
    half = N // 2

    mag = np.abs(fft_vals[:half]) / N   # normalized magnitude
    freqs = fft_freqs[:half]

    plt.figure(figsize=(10, 3))
    plt.plot(freqs, mag, linewidth=0.6)
    plt.xlim(0, min(8000, sr/2))  # show up to 8 kHz (or sr/2)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.title("FFT magnitude (overall frequencies)")
    plt.tight_layout()
    plt.show()

    # Short-Time Fourier Transform (STFT)
    D = librosa.stft(recording, n_fft=n_fft, hop_length=hop_length)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D_db, sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='linear')
    plt.colorbar(format="%+2.0f dB")
    plt.title("STFT (dB)")
    plt.tight_layout()
    plt.show()

    # Mel spectrogram
    S = librosa.feature.melspectrogram(y=recording, sr=sr, n_fft=n_fft,
                                       hop_length=hop_length, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='mel')
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel spectrogram (dB)")
    plt.tight_layout()
    plt.show()

    # MFCC
    mfccs = librosa.feature.mfcc(y=recording, sr=sr, n_mfcc=n_mfcc,
                                 n_fft=n_fft, hop_length=hop_length)
    # mfccs shape: (n_mfcc, n_frames)
    print("MFCC shape:", mfccs.shape)

    plt.figure(figsize=(10, 3))
    librosa.display.specshow(mfccs, sr=sr, hop_length=hop_length, x_axis='time')
    plt.colorbar()
    plt.title(f"MFCC (n_mfcc={n_mfcc})")
    plt.tight_layout()
    plt.show()

    print("Done. You can re-run with different sr / n_fft / hop_length / duration to see how things change.")
