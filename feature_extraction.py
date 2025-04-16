import librosa
import numpy as np


EPSILON = 1e-10


def logarithmic_scale(x, k=9):
    """
    Apply a logarithmic scaling to a normalized value x in [0, 1].
    
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


def rms_volume(y: np.ndarray) -> np.ndarray:
    """
    Estimates the perceived volume of a stereo signal.

    This function works by calculating the root mean square (RMS) of the
    signal, and then scaling the RMS value to range of [0, 1] with a 
    logarithmic (decibel) scaling.
    
    Parameters
    ----------
    y : np.ndarray
        The samples of a (framed) stereo signal. This should have shape
        (2, num_samples, num_frames).
    
    Returns
    -------
    out : np.ndarray
        Each entry in the array is the amplitude of a frame in the 
        signal, where 0 is silence, and 1 is the loudest possible 
        signal.
    """
    min_db = -60
    rms = np.sqrt(np.mean(y ** 2, axis=(0, 1)))
    rms = np.maximum(rms, EPSILON)

    rms_db = 20 * np.log10(rms)
    volume = (rms_db - min_db) / (0 - min_db)
    volume = np.clip(volume, 0, 1)
    return volume


def ild(y: np.ndarray) -> np.ndarray:
    """
    Calculates the interaural level difference (ILD) of a stereo signal.
    
    Parameters
    ----------
    y : np.ndarray
        The samples of a (framed) stereo signal. This should have shape
        (2, num_samples, num_frames).
    
    Returns
    -------
    out : np.ndarray
        Each entry in the array is the ILD of a frame in the signal, 
        where 0 is audio only in the left channel, 1 is audio only in
        the right channel, and 0.5 is equal level in both channels.
    """
    # Calculate the RMS amplitude of each channel
    rms = np.sqrt(np.mean(y ** 2, axis=1))

    # Combine the two channel amplitudes
    combined_rms = np.sqrt(rms[0] ** 2 + rms[1] ** 2)

    # Initailize our output array with a default value of 0
    ilds = np.full_like(combined_rms, 0.5)

    # Calculate the ILD for frames where the total amplitude is greater than 0
    valid = combined_rms > 0
    ilds[valid] = np.arccos(rms[0, valid] / combined_rms[valid]) / np.pi * 2

    return ilds


def itd(D: np.ndarray, sr: int) -> np.ndarray:
    """
    Estimates the inter-channel time difference (ITD) of a stereo signal.

    Uses the Generalized Cross-Correlation with Phase Transform 
    (GCC-PHAT) algorithm. Positive values indicate the right channel 
    signal arrives earlier, and negative values indicate the left 
    channel signal arrives earlier. The output range is (-10, 10).

    Parameters
    ----------
    D : np.ndarray
        The STFT of the stereo signal to be analyzed. This should have 
        shape (2, num_samples, num_frames). 
    sr : int
        The sample rate.
    
    Returns
    -------
    out : np.ndarray
        Each entry in the array is the estimated ITD of a frame in 
        the signal in milliseconds.
    """
    max_lag_ms = 10
    max_lag_samples = (int)(max_lag_ms / 1000 * sr)

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
    lag_ms = lag_samples * 1000 / sr * 2

    return lag_ms


def msw(y: np.ndarray) -> np.ndarray:
    """
    Uses mid-side processing to provide a measure of stereo width.
    
    Parameters
    ----------
    y : np.ndarray
        The samples of a (framed) stereo signal. This should have shape
        (2, num_samples, num_frames).
    
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


def centroid(S: np.ndarray, sr: int) -> np.ndarray:
    """
    Calculates the spectral centroid of a signal. 
    
    Parameters
    ----------
    S : np.ndarray
        The windowed magnitude spectrum of the signal, from an STFT. 
        This should have shape (2, num_bins, num_frames).
    sr: int
        The sample rate.

    Returns
    -------
    out : np.ndarray
        The spectral centroid of each frame in the signal.
    """
    centroids = librosa.feature.spectral_centroid(S=S, sr=sr)
    centroids = centroids.sum(axis=0) / 2 / (sr / 2)
    centroids = logarithmic_scale(centroids)
    return np.squeeze(centroids)


def bandwidth(S: np.ndarray, sr: int) -> np.ndarray:
    """
    Calculates the spectral bandwidth of a signal.
    
    Parameters
    ----------
    S : np.ndarray
        The windowed magnitude spectrum of the signal, from an STFT.
        This should have shape (2, num_bins, num_frames).
    sr: int
        The sample rate.

    Returns
    -------
    out : np.ndarray
        The spectral bandwidth of each frame in the signal.
    """
    bandwidths = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    bandwidths = bandwidths.sum(axis=0) / 2 / (sr / 2)
    bandwidths = logarithmic_scale(bandwidths)
    return np.squeeze(bandwidths)


"""
The following functions are for computing the attributes of the shape to 
be generated by Touchdesigner.
"""


def strength(y: np.ndarray) -> np.ndarray:
    """
    Calculates the strength of the visualization of a sound source.
    
    This implementation uses the RMS amplitude of the signal. If the
    rms amplitude is below a certain threshold, the strength is 0.
    Otherwise, the rms value is is scaled logarithmically.
    
    Parameters
    ----------
    y : np.ndarray
        The samples of a (framed) stereo signal. This should have shape
        (2, num_samples, num_frames).
    
    Returns
    -------
    out : np.ndarray
        Each entry in the array is the strength of a frame in the
        signal (from 0 to 1).
    """
    strength = rms_volume(y)
    strength = strength * 1.25 - 0.2
    strength = np.clip(strength, 0, 1)
    strength = logarithmic_scale(strength)
    return strength


def x_pos(y: np.ndarray, D: np.ndarray, sr: int) -> np.ndarray:
    """
    Calculates the x position of the visualization of a sound source.

    This function originally used the ILD and ITD of the signal, 
    weighted based on an experimentally determined model, but it was not
    rubust to bleed and artifacts from source separation.

    This implementation uses the only ILD of the signal.

    Parameters
    ----------
    y : np.ndarray
        The samples of a stereo signal.
    D : np.ndarray
        The STFT of the stereo signal.
    sr : int
        The sample rate.

    Returns
    -------
    out : np.ndarray
        Each entry in the array is the estimated x position of a frame 
        in the signal, where 0 is the farthest left, and 1 is the 
        furthest right.
    """
    # ilds = ild(y) * 2 - 1
    # itds = np.clip(itd(D, sr), -6., 6.)

    # ild_sign = np.sign(ilds)
    # itd_sign = np.sign(itds)

    # ilds = np.abs(ilds)
    # itds = np.abs(itds)

    # ild_relavent = ilds > 0.2
    # itd_relavent = itds > 1

    # # Choose ild_sign if ild_relavent is True, otherwise choose itd_sign
    # sign = np.where(ild_relavent, ild_sign, itd_sign)

    # k1 = 5.
    # k2 = 0.12
    # k3 = 10.

    # g = (k1 ** ilds - 1) / (k1 - 1)
    # h = k2 * itds * np.log10(k3 * (1 - ilds) + 1) / np.log10(k3 + 1)
    # x_pos = sign * np.clip(g + h, 0, 1)

    # return x_pos / 2 + 0.5

    return ild(y)


def y_pos(S: np.ndarray, sr: int) -> np.ndarray:
    """
    Calculates the y position of the visualization of a sound source.

    This implementation uses the centroid of the signal, scaled (extra)
    logarithmically.

    Parameters
    ----------
    S : np.ndarray
        The windowed magnitude spectrum of the signal, from an STFT.
        This should have shape (2, num_bins, num_frames).
    sr: int
        The sample rate.

    Returns
    -------
    out : np.ndarray
        Each entry in the array is the estimated y position of a frame 
        in the signal, where 0 is the bottom, and 1 is the top.
    """
    centroids = centroid(S, sr)
    y_pos = logarithmic_scale(centroids)
    return y_pos
    

def x_width(y: np.ndarray) -> np.ndarray:
    """
    Calculates the x width of the visualization of a sound source.

    This function uses the mid-side width defined above and scales it
    to a range of [0.05, 0.25], which was determined by experimenting
    with the Touchdesigner program.

    Parameters
    ----------
    y : np.ndarray
        The samples of a stereo signal.

    Returns
    -------
    out : np.ndarray
        Each entry in the array is the estimated x width of a frame in 
        the signal in the range [0.05, 0.25].
    """
    msws = msw(y)
    x_width = logarithmic_scale(msws)
    x_width = 0.05 + 0.2 * x_width
    return x_width


def y_width(S: np.ndarray, sr: int) -> np.ndarray:
    """
    Calculates the y width of the visualization of a sound source.

    This function uses the spectral bandwidth and scales it to a range 
    of [0.05, 0.15], which was determined by experimenting with the 
    Touchdesigner program.

    Parameters
    ----------
    y : np.ndarray
        The samples of a stereo signal.

    Returns
    -------
    out : np.ndarray
        Each entry in the array is the estimated y width of a frame in 
        the signal in the range [0.05, 0.25].
    """
    bandwidths = bandwidth(S, sr)
    y_width = logarithmic_scale(bandwidths)
    y_width = 0.05 + 0.1 * y_width
    return y_width