
import nussl
import librosa
import numpy as np
import feature_extraction as fe

def timbral_clustering(y, num_sources, n_components, window_size=None, 
                       sr=44100,mask_type='soft'):
    """
    Perform timbre clustering on the input audio signal.
    
    Parameters
    ----------
    y : nussl.AudioSignal
        The input audio signal.
    num_sources : int
        The number of sources to separate.
    n_components : int
        The number of components for the clustering algorithm.
    window_size : int | None
        The size of the window in seconds (if None, the entire signal is used).
    sr : int
        The sample rate of the audio signal.
        
    Returns
    -------
    list[nussl.AudioSignal]
        The {num_sources} separated audio signals.
    """
    if window_size is None:
        separator = nussl.separation.primitive.TimbreClustering(y, num_sources, 
                        n_components=n_components, mask_type=mask_type)
        other_split = separator()
        return other_split

    # Split audio into non-overlapping windows since nussl's timbre 
    # clustering isn't working for files longer than ~1 min
    sr = y.sample_rate
    num_windows = y.audio_data.shape[1] // (window_size * sr) + 1
    remainder = y.audio_data.shape[1] % (window_size * sr) / sr
    windows = []

    for i in range(num_windows-1):
        start = i * window_size * sr
        end = (i + 1) * window_size * sr
        window = y.audio_data[:, start:end]
        windows.append(window)

    remainder_window = y.audio_data[:, -int(remainder * sr):]
    windows.append(remainder_window)

    # Initialize processed windows
    clustered_windows = [None] * num_windows

    # Perform clustering on each window
    for j, window in enumerate(windows):
        signal = nussl.AudioSignal(audio_data_array=window, sample_rate=sr)
        separator = nussl.separation.primitive.TimbreClustering(signal, 
                num_sources, n_components=n_components, mask_type=mask_type)

        other_split = separator() 

        # The problem with clustering each window separately is that the 
        # clusters created for each window are all unique because the 
        # clustering algorithm is receiving a different part of the 
        # audio each time. This also means that even when you get very 
        # similar clusters, the order of the clusters is not always 
        # consistent across windows. To fix this, we sort the splits by 
        # their mean centroid and reorder the splits accordingly.

        # Initialize
        mean_centroids = np.zeros(num_sources)
        other_split_sorted = [None] * num_sources
        
        # Get audio data of the clusters of each window and compute ild means
        for i in range(num_sources):
            other_split[i] = other_split[i].audio_data

            # Compute STFT
            D = librosa.stft(other_split[i], n_fft=4096, hop_length=2048, 
                     center=False)
    
            # Compute magnitude spectrum
            S = np.abs(D)

            # Mean centroid
            mean_centroids[i] = (np.mean(fe.centroid(S, sr=sr)))

        # Sort the clusters by mean centroid
        sort_idx = np.argsort(mean_centroids)
        # print(f"Window {j+1} sort indices: {sort_idx}")
        other_split_sorted = [other_split[i] for i in sort_idx]

        # Assemble list of processed windows
        clustered_windows[j] = other_split_sorted

        print(f"Window {j+1}/{len(windows)} split successfully")


    # clustered_windows is now the whole collection of data. Each window 
    # (e.g. clustered_windows[0]) contains a list of clusters, with each 
    # cluster window_size seconds long. clustered_windows[1] contains 
    # the next window_size seconds of each cluster. Each cluster in each 
    # window (e.g. clustered_windows[0][0]) contains the processed / 
    # clustered audio data and has 2 channels.

    # Concatenate windows
    tracks = [np.concatenate([window[i] for window in clustered_windows], axis=1)
            for i in range(num_sources)]
    
    tracks = [nussl.AudioSignal(audio_data_array=track, sample_rate=sr) 
              for track in tracks]

    return tracks

def spatial_clustering(y, num_sources, clustering_type='KMeans', mask_type='soft'):
    """
    Perform spatial clustering on the input audio signal.
    
    Parameters
    ----------
    y : nussl.AudioSignal
        The input audio signal.
    num_sources : int
        The number of sources to separate.
        
    Returns
    -------
    list[nussl.AudioSignal]
        The {num_sources} separated audio signals.
    """
    # Separate
    separator = nussl.separation.spatial.SpatialClustering(
        y, num_sources, clustering_type=clustering_type, mask_type=mask_type)
    other_split = separator() 
    return other_split
