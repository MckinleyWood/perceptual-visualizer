import librosa
import soundfile as sf
import numpy as np

SR = 22050
frame_length = 8192
hop_length = 4096

def ild(S: np.ndarray) -> np.ndarray:
    """
    Calculates the Interaural Level Difference (ILD) of a stereo signal.
    
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
    print("Calculating the RMS channel amplitudes...")
    rms = np.sqrt(np.mean(S ** 2, axis=1))

    print("Calculating the ILDs...")
    # "Add" the two channel amplitudes
    total_amp = np.sqrt(rms[0] ** 2 + rms[1] ** 2)

    # Initailize our output array with a default value of 0.5
    ilds = np.full_like(total_amp, 0.5)

    # Calculate the ILD for frames where the total amplitude is greater than 0
    valid = total_amp > 0
    ilds[valid] = np.arccos(rms[0, valid] / total_amp[valid]) / np.pi * 2
    ilds = np.arccos(rms[0] / total_amp) / np.pi * 2

    return ilds


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
        the signal.
    """
    mid = (y[0] + y[1]) / np.sqrt(2)
    side = (y[0] - y[1]) / np.sqrt(2)

    mid_rms = librosa.feature.rms(y=mid)
    side_rms = librosa.feature.rms(y=side)

    return side_rms / mid_rms

def main():
    # file_path = "Audio/beep.125.wav"
    file_path = "Audio/beep.5.wav"
    # file_path = "Audio/beep.875.wav"

    print("Loading file...")
    y, _ = librosa.load(file_path, sr=SR, duration=10, mono=False)

    print("Calculating the STFTs...")
    D = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
    S = np.abs(D)

    y = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    y = y * np.hanning(frame_length)[None, :, None]

    print("Calculating the MSW...")
    msws = msw(y)

    print("MSWs:")
    print(msws)

    print("Calculating the ILDs...")
    ilds = ild(S)

    print("ILDs:")
    print(ilds)

if __name__ == "__main__":
    main()
