
import nussl
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from sklearn.decomposition import NMF
import os
import numpy as np
import feature_extraction as fe

def other_clustering(input_file_path, output_folder_path, num_sources, window_size_s, clustering="spatial"):

    #load audio
    x,sr = librosa.load(input_file_path, sr=None, mono=False)

    #split audio into non-overlapping windows since nussl isn't working for 
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

        if clustering == "spatial":
            separator = nussl.separation.spatial.SpatialClustering(signal,num_sources)
        elif clustering == "timbre":
            separator = nussl.separation.primitive.TimbreClustering(signal, num_sources, 50)

        #POTENTIALLY ADD PROJET CLUSTERING

        else:
            raise ValueError(f"Invalid clustering method: {clustering}")

        other_split = separator() 

        #The problem with clustering each window separately is that the clusters
        #created for each window are all unique because the clustering algorithm
        #is receiving a different part of the audio each time. This also means that
        #even when you get very similar clusters, the order of the clusters is not
        #always consistent across windows. To fix this, we sort the splits by their
        #mean ild (interaural level difference) and reorder the splits accordingly.
        #This actually works surprisingly well since the mean ild is a good indicator 
        # of the source location in the stereo field.

        ###Might want to use a different feature for timbre clustering, like spectral centroid###

        #initialize
        mean_ilds = np.zeros(num_sources)
        other_split_sorted = [None] * num_sources
        
        #get audio data of the clusters of each window and compute ild means
        for i in range(num_sources):
            other_split[i] = other_split[i].audio_data
            mean_ilds[i] = (np.mean(fe.ild(other_split[i])))

        #sort the clusters by mean ild
        sort_idx = np.argsort(mean_ilds)
        # print(f"Window {j+1} sort indices: {sort_idx}")
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
        sf.write(output_folder_path + f"other_{i}.wav", tracks[i].T, sr)

#Test here
input_file_path = "/Users/owenohlson/Documents/GitHub/perceptual-visualizer/output/Parquet Courts - Tenderness/demucs/other.wav"
output_folder_path = "/Users/owenohlson/Documents/GitHub/perceptual-visualizer/output/Parquet Courts - Tenderness/nussl/"
other_clustering(input_file_path, output_folder_path, num_sources=2, window_size_s=10)
