# Audio Processing Repository

This repository contains a collection of scripts, utilities, and a Jupyter Notebook for exploring and processing audio signals. It is designed for educational purposes and provides a hands-on approach to understanding audio signal processing concepts such as sine waves, Fourier Transform, Short-Time Fourier Transform (STFT), and Mel Frequency Cepstral Coefficients (MFCC).

---

## Repository Structure

### Files and Directories

- **`audioprocessing.ipynb`**: A comprehensive Jupyter Notebook that walks through various audio processing techniques, including:
  - Generating sine waves.
  - Visualizing waveforms.
  - Applying Fourier Transform and STFT.
  - Extracting MFCCs.
- **`audiofiles/`**: Contains sample audio files for analysis and experimentation.
  - `notes/`: Includes `.wav` files such as `C2_C4.wav` and `G6_A6.wav` for pitch and frequency analysis.
- **`images/`**: Contains visual aids and diagrams used in the notebook, such as:
  - Fourier Transform illustrations.
  - Mel Scale and MFCC process diagrams.
- **`utils/`**: A collection of Python utility scripts for audio processing:
  - `plot_waveform.py`: Functions to plot waveforms.
  - `read_audio_and_plot.py`: Reads audio files and plots their waveforms.
  - `record_audio_and_plot.py`: Records audio and visualizes it.
  - `save_and_play_signal.py`: Saves and plays generated audio signals.
- **`pyproject.toml`**: Defines the project dependencies and metadata.
- **`README.md`**: This file, providing an overview of the repository.
- **`uv.lock`**: Dependency lock file.

---

## Features

1. **Audio Signal Generation**:
   - Generate sine and cosine waves with customizable frequencies, amplitudes, and phases.
   - Combine multiple signals to create complex waveforms.

2. **Visualization**:
   - Plot time-domain signals.
   - Visualize frequency-domain representations using FFT and STFT.

3. **Audio Analysis**:
   - Extract and analyze Mel Frequency Cepstral Coefficients (MFCC).
   - Understand the sensitivity of human hearing to different frequencies.

4. **Interactive Learning**:
   - Step-by-step explanations and visualizations in the Jupyter Notebook.
   - Links to external resources for deeper understanding.

---

## Installation

### Prerequisites

- Python 3.13 or higher.
- Recommended: A virtual environment to manage dependencies.

### Steps

1. Install libraries using uv

    ```bash
    uv sync
    ```

---

## Usage

### Running the Jupyter Notebook

1. Open `audioprocessing.ipynb` and follow the cells step-by-step.

---

## Concepts Covered

1. **Sine Waves**:
   - Mathematical representation:
    $$
    s(t) = A \cdot \sin(2\pi f t + \phi)
    $$

1. **Fourier Transform**:
   - Convert time-domain signals to frequency-domain.

2. **Short-Time Fourier Transform (STFT)**:
   - Analyze how frequencies change over time.

3. **Mel Frequency Cepstral Coefficients (MFCC)**:
   - Extract features for audio classification and speech recognition.

---

## Resources

### Libraries

- [Librosa](https://librosa.org/): Audio and music signal analysis.
- [SoundFile](https://pypi.org/project/soundfile/): Reading and writing sound files.

### Tutorials

- [Mel-Spectrogram and MFCCs | Lecture 72 (Part 1) | Applied Deep Learning](https://youtu.be/hF72sY70_IQ?si=u7XEST_Js1mL13a4)
- [Mel Frequency Cepstral Coefficients (MFCC) Explained](https://youtu.be/SJo7vPgRlBQ?si=KjGdRm-52k98_AWt)

### Research Papers

- [McFee, Brian, et al. “librosa: Audio and music signal analysis in python.”](https://brianmcfee.net/papers/scipy2015_librosa.pdf)

---

