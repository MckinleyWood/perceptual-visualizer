import argparse
from dataclasses import dataclass, asdict
import csv
import glob
import os

import demucs.separate
import librosa
import numpy as np
import soundfile as sf

# Since nussl is deprecated, we need to patch some things for it to work
import scipy.signal.windows as sw
import scipy.signal
np.float_ = np.float64
scipy.signal.hamming = sw.hamming
scipy.signal.hann = sw.hann
scipy.signal.blackman = sw.blackman

import nussl

import feature_extraction as fe


@dataclass
class DataFrame:
    x_position: float
    y_position: float
    x_width: float
    y_width: float
    strength: float


def extract_features(
        y: np.ndarray,
        duration: int, 
        frame_length: int, 
        hop_length: int, 
        sr: int,
        output_filename: str) -> None:
    """
    """
    # y will be the input signal to analyze.
    # y, _ = librosa.load(filename, sr=sr, duration=duration, mono=False)
    
    # D will be the (complex) STFT of y.
    D = librosa.stft(y, n_fft=frame_length, hop_length=hop_length, 
                     center=False)
    
    # S will be the magnitude spectrum of y.
    S = np.abs(D)

    # We frame the signal y to match the frames of the STFT.
    y = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    y = y * np.hanning(frame_length)[None, :, None]

    # Calculate the features.
    volumes = fe.rms_volume(y)
    x_pos = fe.x_pos(y, D, sr)
    msws = fe.msw(y)
    centroids = fe.centroid(S, sr)
    bandwidths = fe.bandwidth(S, sr)

    # Create a CSV file with the features.
    data_frames = [
        DataFrame(xp, yp, xw, yw, s)
        for xp, yp, xw, yw, s 
        in zip(x_pos, centroids, msws, bandwidths, volumes)
    ]
    
    data_dicts = [asdict(frame) for frame in data_frames]
    output_filename = "output/" + output_filename + '.csv'
    fieldnames = data_dicts[0].keys()

    with open(output_filename, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_dicts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--frame_length", type=int, default=8192)
    parser.add_argument("--hop_length", type=int, default=4096)
    parser.add_argument("--sr", type=int, default=22050)
    args = parser.parse_args()

    # Separate vocals, drums, bass, and other stems using Demucs.
    # This will create a directory structure like:
    # separated/htdemucs/filename/
    # ├── vocals.wav
    # ├── drums.wav
    # ├── bass.wav
    # └── other.wav
    print(f"\nRunning demucs on {args.input_path}...\n")
    demucs.separate.main([args.input_path])

    # Load the separated audio files for further separation with nussl.
    base = os.path.splitext(os.path.basename(args.input_path))[0]
    vocals = nussl.AudioSignal(
        os.path.join("separated", "htdemucs", base, "vocals.wav"))
    drums = nussl.AudioSignal(
        os.path.join("separated", "htdemucs", base, "drums.wav"))
    bass = nussl.AudioSignal(
        os.path.join("separated", "htdemucs", base, "bass.wav"))
    other = nussl.AudioSignal(
        os.path.join("separated", "htdemucs", base, "other.wav"))
        
    # Separate further using nussl...
    print("\nRunning second-level separation...\n")
    timbre_separator = nussl.separation.primitive.TimbreClustering(
        other, 2, 50)
    other_split = timbre_separator()

    # Create a list of the separated audio signals.
    sources = [
        vocals.audio_data, 
        drums.audio_data, 
        bass.audio_data, 
        other_split[0].audio_data,
        other_split[1].audio_data
    ]

    # Resample to the rate we want for feature extraction and file writing
    sources = [librosa.resample(y, orig_sr=44100, target_sr=args.sr) 
               for y in sources]

    # Create the output directory if it doesn't exist.
    if not os.path.exists("output"):
        os.makedirs("output")
    
    # Process each file and write the output files.
    print("\nExtracting features and writing files...")
    for i, y in enumerate(sources):
        # sf.write("output/" + (str)(i) + ".wav", y.T, samplerate=args.sr)
        extract_features(y, args.duration, args.frame_length, args.hop_length, 
                         args.sr, (str)(i))
        
    print("\nDone!\n")
 

if __name__ == "__main__":
    main()
