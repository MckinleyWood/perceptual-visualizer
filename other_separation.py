
import nussl
import librosa
import soundfile as sf
import os
import numpy as np
import feature_extraction as fe
import time


def timbral_clustering(input_file_path, output_folder_path, num_sources=2, 
                       window_size_s=30):

    print("ignore convergence warning^")

    #load audio
    x,sr = librosa.load(input_file_path, sr=None, mono=False)

    #split audio into non-overlapping windows since nussl's timbre clustering isn't working for 
    #files longer than ~1 min
    num_windows = x.shape[1] // (window_size_s * sr) + 1
    remainder = x.shape[1] % (window_size_s * sr) / sr
    windows = []

    for i in range(num_windows-1):
        start = i * window_size_s * sr
        end = (i + 1) * window_size_s * sr
        window = x[:, start:end]
        windows.append(window)

    remainder_window = x[:, -int(remainder * sr):]
    windows.append(remainder_window)

    #initialize processed windows
    clustered_windows = [None] * num_windows

    #perform clustering on each window
    for j, window in enumerate(windows):
        signal = nussl.AudioSignal(audio_data_array=window, sample_rate=sr)
        separator = nussl.separation.primitive.TimbreClustering(signal, num_sources, 50, mask_type='binary')

        other_split = separator() 

        #The problem with clustering each window separately is that the clusters
        #created for each window are all unique because the clustering algorithm
        #is receiving a different part of the audio each time. This also means that
        #even when you get very similar clusters, the order of the clusters is not
        #always consistent across windows. To fix this, we sort the splits by their
        #mean centroid and reorder the splits accordingly.

        #initialize
        mean_centroids = np.zeros(num_sources)
        other_split_sorted = [None] * num_sources
        
        #get audio data of the clusters of each window and compute ild means
        for i in range(num_sources):
            other_split[i] = other_split[i].audio_data

            # Compute STFT
            D = librosa.stft(other_split[i], n_fft=4096, hop_length=2048, 
                     center=False)
    
            # Compute magnitude spectrum
            S = np.abs(D)

            # Mean centroid
            mean_centroids[i] = (np.mean(fe.centroid(S,sr=sr)))

        #sort the clusters by mean ild
        sort_idx = np.argsort(mean_centroids)
        #print(f"Window {j+1} sort indices: {sort_idx}")
        other_split_sorted = [other_split[i] for i in sort_idx]

        #assemble list of processed windows
        clustered_windows[j] = other_split_sorted

        print(f"Window {j+1}/{len(windows)} split successfully")


    #clustered_windows is now the whole collection of data
    #each window (e.g. clustered_windows[0]) contains a list of clusters, with each cluster window_size_s seconds long.
    #clustered_windows[1] contains the next window_size_s seconds of each cluster
    #each cluster in each window (e.g. clustered_windows[0][0]) contains the processed/clustered audio data and has 2 channels

    #mainly wrote this out for my own sake^

    #Concatenate windows
    tracks = [np.concatenate([window[i] for window in clustered_windows], axis=1)
            for i in range(num_sources)]

    #Save as .wav files
    for i in range(num_sources):
        sf.write(output_folder_path + f"other_timbral_{i}.wav", tracks[i].T, sr)


# def spatial_clustering(input_file_path, output_folder_path, num_sources):
    
#     # Load audio
#     x,sr = librosa.load(input_file_path, sr=None, mono=False)
#     signal = nussl.AudioSignal(audio_data_array=x, sample_rate=sr)

#     # Separate
#     separator = nussl.separation.spatial.SpatialClustering(signal,num_sources, mask_type='binary')
#     other_split = separator() 

#     # Save audio data as .wav files
#     for i in range(num_sources):
#         other_split[i] = other_split[i].audio_data
#         sf.write(output_folder_path + f"other_spatial_{i}.wav", other_split[i].T, sr)


def spatial_clustering(y, num_sources, clustering_type='KMeans'):
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
        y, num_sources, clustering_type=clustering_type)
    other_split = separator() 
    return other_split


def main():
    # Test here

    input_file_path = "/Users/owenohlson/Documents/GitHub/perceptual-visualizer/output/Parquet Courts - Tenderness/demucs/other.wav"
    output_folder_path = "/Users/owenohlson/Documents/GitHub/perceptual-visualizer/output/Parquet Courts - Tenderness/nussl/"

    start_time = time.time()
    spatial_clustering(input_file_path, output_folder_path, num_sources=2)
    spatial_time = time.time()
    print(f"Spatial clustering time taken: {spatial_time - start_time} seconds")

    timbral_clustering(input_file_path, output_folder_path, num_sources=2, window_size_s=30, clustering='timbral')
    timbral_time = time.time()
    print(f"Timbral clustering time taken: {timbral_time - spatial_time} seconds")


if __name__ == "__main__":
    main()