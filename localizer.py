import argparse
from dataclasses import dataclass, asdict
import csv
import glob
import librosa
import numpy as np
import os

import feature_extraction as fe


@dataclass
class DataFrame:
    x_position: float
    y_position: float
    x_width: float
    y_width: float
    strength: float


def process_file(
        filename: str, 
        duration: int, 
        frame_length: int, 
        hop_length, 
        sr: int) -> None:
    """
    Processes a file and extracts features."
    """
       # y will be the input signal to analyze.
    y, _ = librosa.load(filename, sr=sr, duration=duration, mono=False)
    
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
    base = os.path.splitext(os.path.basename(filename))[0]
    output_filename = "output/" + base + '.csv'
    fieldnames = data_dicts[0].keys()

    with open(output_filename, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_dicts)

    print(f"Processed '{filename}' -> '{output_filename}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--frame_length", type=int, default=8192)
    parser.add_argument("--hop_length", type=int, default=4096)
    parser.add_argument("--sr", type=int, default=22050)
    args = parser.parse_args()

    # Create the output directory if it doesn't exist.
    if not os.path.exists("output"):
        os.makedirs("output")

    if os.path.isdir(args.input_path):
        # If input_path is a folder, process all .wav files in it.
        file_list = glob.glob(os.path.join(args.input_path, "*.wav"))
    else:
        # Otherwise, assume it's a single file.
        file_list = [args.input_path]
    
    # Process each file.
    for filename in file_list:
        process_file(filename, args.duration, args.frame_length, 
                     args.hop_length, args.sr)
 

if __name__ == "__main__":
    main()
