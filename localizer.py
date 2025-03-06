import argparse
from dataclasses import dataclass, asdict
import librosa
import numpy as np

import feature_extraction as fe


@dataclass
class SourceData:
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

    amps = fe.rms(y)
    ilds = fe.ild(S)
    itds = fe.itd(D, args.sr)
    msws = fe.msw(y)
    centroids = fe.centroid(S, args.sr)
    bandwidths = fe.bandwidth(S, args.sr)

    print(amps)


if __name__ == "__main__":
    main()
