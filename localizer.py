import librosa
import soundfile as sf
import numpy as np


SR = 22050
EPSILON = 1e-10


def logarithmic_scale(x, k=9):
    """
    Apply a logarithmic scaling to a normalized value x in [0,1].
    
    Parameters
    ----------
    x : float or np.array
        Normalized input(s) between 0 and 1.
    k : float
        Curvature parameter. Higher values mean more logarization.
    
    Returns
    -------
    out : float or np.array 
        Scaled value(s) still in [0,1].
    """
    return np.log(1 + k * x) / np.log(1 + k)


def rms(S: np.ndarray) -> np.ndarray:
    """
    Calculates the root-mean-square (RMS) amplitude of a stereo signal.
    
    Parameters
    ----------
    S : np.ndarray
        The magnitude spectrum of a stereo signal.
    
    Returns
    -------
    out : np.ndarray
        Each entry in the array is the amplitude of a frame in the 
        signal, where 0 is silence, and 1 is the loudest possible 
        signal.
    """
    rms = np.sqrt(np.mean(S ** 2, axis=1))
    return np.sqrt(rms[0] ** 2 + rms[1] ** 2) / np.sqrt(2)


def ild(S: np.ndarray) -> np.ndarray:
    """
    Calculates the interaural level difference (ILD) of a stereo signal.
    
    Parameters
    ----------
    S : np.ndarray
        The magnitude spectrum of a stereo signal.
    
    Returns
    -------
    out : np.ndarray
        Each entry in the array is the ILD of a frame in the signal, 
        where 0 is audio only in the left channel, 1 is audio only in
        the right channel, and 0.5 is equal level in both channels.
    """
    rms = np.sqrt(np.mean(S ** 2, axis=1))

    # "Add" the two channel amplitudes
    total_amp = np.sqrt(rms[0] ** 2 + rms[1] ** 2)

    # Initailize our output array with a default value of 0.5
    ilds = np.full_like(total_amp, 0.5)

    # Calculate the ILD for frames where the total amplitude is greater than 0
    valid = total_amp > 0
    ilds[valid] = np.arccos(rms[0, valid] / total_amp[valid]) / np.pi * 2
    ilds = np.arccos(rms[0] / total_amp) / np.pi * 2

    return ilds


def itd(D: np.ndarray) -> np.ndarray:
    """
    Estimates the inter-channel time difference (ITD) of a stereo signal.

    Uses the Generalized Cross-Correlation with Phase Transform 
    (GCC-PHAT) algorithm.

    Parameters
    ----------
    D : np.ndarray
        The STFT of the stereo signal to be analyzed.
    
    Returns
    -------
    out : np.ndarray
        Each entry in the array is the estimated ITD of a frame in 
        the signal in milliseconds. Possible range is (-10, 10)
    """
    max_lag_ms = 10
    max_lag_samples = (int)(max_lag_ms / 1000 * SR)

    R = D[0] * np.conj(D[1])
    R_mag = np.abs(R)
    R_mag[R_mag == 0] = EPSILON
    R /= R_mag

    r = np.fft.ifft(R, axis=0)
    r = np.real(r)

    r = np.fft.fftshift(r, axes=0)

    num_samples, num_frames = r.shape
    midpoint = num_samples // 2
    r = r[midpoint - max_lag_samples : midpoint + max_lag_samples, :]

    lag_samples = np.argmax(r, axis=0) - max_lag_samples
    lag_ms = lag_samples * 1000 / SR * 2

    return lag_ms



def msw(y: np.ndarray) -> np.ndarray:
    """
    Estimates the stereo width of a signal based on mid-side processing.
    
    Parameters
    ----------
    y : np.ndarray
        The samples of a stereo signal.
    
    Returns
    -------
    out : np.ndarray
        Each entry in the array is the stereo width index of a frame in 
        the signal, where 0 is the narrowest and 1 is the widest.
    """
    rms = np.sqrt(np.mean(y ** 2, axis=1, keepdims=True))
    y_norm = y / (rms + EPSILON)

    mid = (y_norm[0] + y_norm[1]) / np.sqrt(2)
    side = (y_norm[0] - y_norm[1]) / np.sqrt(2)

    mid_rms = np.sqrt(np.mean(mid ** 2, axis=0))
    side_rms = np.sqrt(np.mean(side ** 2, axis=0))

    msw = side_rms / (mid_rms + side_rms + EPSILON)
    return msw


def centroid(S: np.ndarray) -> np.ndarray:
    """
    Calculates the spectral centroid of a signal.
    
    Parameters
    ----------
    S : np.ndarray
        The windowed magnitude spectrum of the signal, from an STFT.

    Returns
    -------
    out : np.ndarray
        The spectral centroid of each frame in the signal.
    """
    centroids = librosa.feature.spectral_centroid(S=S, sr=SR)
    centroids = centroids.sum(axis=0) / 2 / (SR / 2)
    centroids = logarithmic_scale(centroids)
    return centroids


def bandwidth(S: np.ndarray) -> np.ndarray:
    """
    Calculates the spectral bandwidth of a signal.
    
    Parameters
    ----------
    S : np.ndarray
        The windowed magnitude spectrum of the signal, from an STFT.

    Returns
    -------
    out : np.ndarray
        The spectral bandwidth of each frame in the signal.
    """
    bandwidths = librosa.feature.spectral_bandwidth(S=S, sr=SR)
    bandwidths = bandwidths.sum(axis=0) / 2 / (SR / 2)
    bandwidths = logarithmic_scale(bandwidths)
    return bandwidths


def main():
    # file_path = "Audio/beep.125.wav"
    # file_path = "Audio/beep.5.wav"
    # file_path = "Audio/beep.875.wav"
    # file_path = "Audio/beep.wide.wav"
    file_path = "Audio/beep2ms.wav" 

    frame_length = 8192
    hop_length = 4096  

    # print("Loading file...")
    y, _ = librosa.load(file_path, sr=SR, duration=10, mono=False)

    # print(f"y shape: {y.shape}")

    # print("Calculating the STFTs...")
    D = librosa.stft(
        y,
        n_fft=frame_length, 
        hop_length=hop_length, 
        center=False)

    S = np.abs(D)

    y = librosa.util.frame(
        y, 
        frame_length=frame_length, 
        hop_length=hop_length)
    y = y * np.hanning(frame_length)[None, :, None]

    # print("Calculating the ILDs...")
    ilds = ild(S)
    print("ILDs:")
    print(ilds)

    # print("Calculating the ITDs...")
    itds = itd(D)
    print("ITDS:")
    print(itds)
    
    # print("Calculating the MSW...")
    msws = msw(y)
    print("MSWs:")
    print(msws)

    # print("Calculating the centroid...")
    centroids = centroid(S)
    print("Centroids:")
    print(centroids)

    # print("Calculating the bandwidth...")
    bandwidths = bandwidth(S)
    print("Bandwidths:")
    print(bandwidths)


if __name__ == "__main__":
    main()
