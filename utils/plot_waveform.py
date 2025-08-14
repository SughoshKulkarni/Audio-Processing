
import matplotlib.pyplot as plt


def plot_waveform(t, signal):
    """
    Plot the waveform of a signal.

    Parameters
    ----------
    t : numpy.ndarray
        The time axis values.
    signal : numpy.ndarray
        The audio signal values.
    """

    # Plot the signal
    plt.figure(figsize=(10, 4))
    plt.plot(t, signal)
    plt.title("Sample Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()