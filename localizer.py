import argparse
from dataclasses import dataclass, asdict
import json
import librosa
import numpy as np

import feature_extraction as fe


@dataclass
class DataFrame:
    x_position: float
    y_position: float
    x_width: float
    y_width: float
    strength: float


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--frame_length", type=int, default=8192)
    parser.add_argument("--hop_length", type=int, default=4096)
    parser.add_argument("--sr", type=int, default=22050)
    args = parser.parse_args()

    y, _ = librosa.load(args.filename, 
        sr=args.sr, 
        duration=args.duration, 
        mono=False)
    D = librosa.stft(
        y,
        n_fft=args.frame_length,
        hop_length=args.hop_length, 
        center=False)
    S = np.abs(D)

    y = librosa.util.frame(
        y, 
        frame_length=args.frame_length, 
        hop_length=args.hop_length)
    y = y * np.hanning(args.frame_length)[None, :, None]

    volumes = fe.rms_volume(y)
    ilds = fe.ild(y)
    itds = fe.itd(D, args.sr)
    msws = fe.msw(y)
    centroids = fe.centroid(S, args.sr)
    bandwidths = fe.bandwidth(S, args.sr)

    data_frames = [
        DataFrame(xp, yp, xw, yw, s)
        for xp, yp, xw, yw, s 
        in zip(ilds, centroids, msws, bandwidths, volumes)
    ]
    
    data_dicts = [asdict(frame) for frame in data_frames]
    with open("output.json", "w") as f:
        json.dump(data_dicts, f, indent=2)


if __name__ == "__main__":
    main()
