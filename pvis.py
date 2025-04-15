import argparse
from dataclasses import dataclass, asdict
import csv
import os
import shutil
import sys

# We are not using MusDB and this keeps nussl happy.
sys.modules['musdb'] = __import__('musdb_dummy')

import demucs.separate
import larsnet.separate
import librosa
import numpy as np
import soundfile as sf

# Since nussl is deprecated, we need to patch some things for it to work.
import scipy.signal.windows as sw
import scipy.signal

np.float_ = np.float64
scipy.signal.hamming = sw.hamming
scipy.signal.hann = sw.hann
scipy.signal.blackman = sw.blackman

import nussl

import feature_extraction as fe
import other_separation as sep


MAX_OTHER_SOURCES = 5


@dataclass
class DataFrame:
    x_position: float
    y_position: float
    x_width: float
    y_width: float
    strength: float


def extract_features(
        output_filename: str,
        y: np.ndarray,
        frame_length: int, 
        hop_length: int, 
        sr: int) -> None:
    """
    Extracts features from an audio signal and saves them to a CSV file.
 
    Parameters
    ----------
    output_filename : str
        The name of the output CSV file.
    y : np.ndarray
        The input audio signal.
    frame_length : int
        The length of the frames for STFT.
    hop_length : int
        The hop length for STFT.
    sr : int
        The sample rate of the audio signal.
    """
    # y will be the input signal to analyze.
    
    # D will be the (complex) STFT of y.
    D = librosa.stft(y, n_fft=frame_length, hop_length=hop_length, 
                     center=False)
    
    # S will be the magnitude spectrum of y.
    S = np.abs(D)

    # We frame the signal y to match the frames of the STFT.
    y = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    y = y * np.hanning(frame_length)[None, :, None]

    # Calculate the features.
    x_pos = fe.x_pos(y, D, sr)
    y_pos = fe.y_pos(S, sr)
    x_width = fe.x_width(y)
    y_width = fe.y_width(S, sr)
    strengths = fe.strength(y)

    # Create a CSV file with the features.
    data_frames = [
        DataFrame(xp, yp, xw, yw, s)
        for xp, yp, xw, yw, s 
        in zip(x_pos, y_pos, x_width, y_width, strengths)
    ]
    
    data_dicts = [asdict(frame) for frame in data_frames]
    fieldnames = data_dicts[0].keys()

    with open(output_filename, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_dicts)


def create_output_dirs(base: str) -> None:
    """
    Create the output directories if they don't exist.

    The directory structure will be:
    output/<base>
    ├── demucs/
    ├── larsnet/
    ├── nussl/
    └── features/
    
    Parameters
    ----------
    base : str
        The name of the input file without the extension.
    """
    if not os.path.exists("output"):
        os.makedirs("output")
    if not os.path.exists(os.path.join("output", base)):
        os.makedirs(os.path.join("output", base))
    if not os.path.exists(os.path.join("output", base, "demucs")):
        os.makedirs(os.path.join("output", base, "demucs"))
    if not os.path.exists(os.path.join("output", base, "larsnet")):
        os.makedirs(os.path.join("output", base, "larsnet"))
    if not os.path.exists(os.path.join("output", base, "nussl")):
        os.makedirs(os.path.join("output", base, "nussl"))
    if not os.path.exists(os.path.join("output", base, "features")):
        os.makedirs(os.path.join("output", base, "features"))
    if not os.path.exists(os.path.join("output", base, "larsnet", "input")):
        os.makedirs(os.path.join("output", base, "larsnet", "input"))


def move_demucs_files(base: str) -> None:
    """
    Move the Demucs files to the appropriate directory.

    The directory structure will be:
    output/<base>/demucs/
    ├── vocals.wav
    ├── drums.wav
    ├── bass.wav
    └── other.wav

    Parameters
    ----------
    base : str
        The name of the input file without the extension.
    """
    shutil.move(
        os.path.join("separated", "htdemucs", base, "vocals.wav"),
        os.path.join("output", base, "demucs", "vocals.wav"))
    shutil.move(
        os.path.join("separated", "htdemucs", base, "drums.wav"),
        os.path.join("output", base, "demucs", "drums.wav"))
    shutil.move(
        os.path.join("separated", "htdemucs", base, "bass.wav"),
        os.path.join("output", base, "demucs", "bass.wav"))
    shutil.move(
        os.path.join("separated", "htdemucs", base, "other.wav"),
        os.path.join("output", base, "demucs", "other.wav"))
    
    shutil.rmtree("separated")


def move_larsnet_files(larsnet_path: str) -> None:
    """
    Move the LarsNet files to the appropriate directory.

    The directory structure will be:
    output/<base>/larsnet/
    ├── kick.wav
    ├── snare.wav
    ├── toms.wav
    ├── hihat.wav
    └── cymbals.wav

    Parameters
    ----------
    larsnet_path : str
        The path to the LarsNet output directory.
    """
    shutil.move(
        os.path.join(larsnet_path, "kick", "drums.wav"),
        os.path.join(larsnet_path, "kick.wav"))
    shutil.move(
        os.path.join(larsnet_path, "snare", "drums.wav"),
        os.path.join(larsnet_path, "snare.wav"))
    shutil.move(
        os.path.join(larsnet_path, "toms", "drums.wav"),
        os.path.join(larsnet_path, "toms.wav"))
    shutil.move(
        os.path.join(larsnet_path, "hihat", "drums.wav"),
        os.path.join(larsnet_path, "hihat.wav"))
    shutil.move(
        os.path.join(larsnet_path, "cymbals", "drums.wav"),
        os.path.join(larsnet_path, "cymbals.wav"))

    shutil.rmtree(os.path.join(larsnet_path, "input"))
    shutil.rmtree(os.path.join(larsnet_path, "kick"))
    shutil.rmtree(os.path.join(larsnet_path, "snare"))
    shutil.rmtree(os.path.join(larsnet_path, "toms"))
    shutil.rmtree(os.path.join(larsnet_path, "hihat"))
    shutil.rmtree(os.path.join(larsnet_path, "cymbals"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame_length", type=int, default=4096)
    parser.add_argument("--hop_length", type=int, default=2048)
    parser.add_argument("--clustering", type=str, default="spatial")
    parser.add_argument("--tc_window", type=str, default="None")
    parser.add_argument("--num_sources", type=int, default=2)
    args = parser.parse_args()

    # Create the output directories if they don't exist.
    base = os.path.splitext(os.path.basename(args.input_path))[0]
    larsnet_path = os.path.join("output", base, "larsnet")
    demucs_path = os.path.join("output", base, "demucs")
    create_output_dirs(base)

    # Separate vocals, drums, bass, and other stems using Demucs
    # You can comment this out if you have already run Demucs.
    print(f"\nRunning demucs on {args.input_path}...\n")
    demucs.separate.main([args.input_path])
    move_demucs_files(base)

    # Separate the drums further using larsnet
    print("\nRunning larsnet drum separation...\n")
    shutil.copy(os.path.join(demucs_path, "drums.wav"),
                os.path.join(larsnet_path, "input", "drums.wav"))
    larsnet.separate.separate(os.path.join(larsnet_path, "input"), 
                              larsnet_path, wiener_exponent=None, device='cpu')
    move_larsnet_files(larsnet_path)
    
    # Load the separated audio files for further separation with nussl
    # and feature extraction.
    vocals = nussl.AudioSignal(os.path.join(demucs_path, "vocals.wav"))
    kick = nussl.AudioSignal(os.path.join(larsnet_path, "kick.wav"))
    snare = nussl.AudioSignal(os.path.join(larsnet_path, "snare.wav"))
    toms = nussl.AudioSignal(os.path.join(larsnet_path, "toms.wav"))
    hihat = nussl.AudioSignal(os.path.join(larsnet_path, "hihat.wav"))
    cymbals = nussl.AudioSignal(os.path.join(larsnet_path, "cymbals.wav"))
    bass = nussl.AudioSignal(os.path.join(demucs_path, "bass.wav"))
    other = nussl.AudioSignal(os.path.join(demucs_path, "other.wav"))
    
    # Separate further using nussl...
    print("\nRunning nussl \"other\" separation...")

    if args.clustering == "spatial":
        other_split = sep.spatial_clustering(other, args.num_sources, 
                        clustering_type="GaussianMixture")
    elif args.clustering == "timbral":
        if args.tc_window == "None":
            window_size = None
        else:
            window_size = int(args.tc_window)
        other_split = sep.timbral_clustering(other, args.num_sources, 
                        args.num_sources * 2, window_size=window_size)
    else:
        raise ValueError(f"Invalid clustering method: {args.clustering}")
    
    # Create a list of the separated audio signals and resample them.
    sources = [
        vocals,
        kick,
        snare,
        toms,
        hihat,
        cymbals, 
        bass
    ]
    for i in range(args.num_sources):
        sources.append(other_split[i])

    feature_sr = args.fps * args.hop_length 
    sources = [s.audio_data for s in sources]
    sources = [librosa.resample(y, orig_sr=44100, target_sr=feature_sr) 
               for y in sources]
    
    # Process each file and write the output files.
    print("\nExtracting features and writing files...")
    for i, y in enumerate(sources):
        sf.write(os.path.join("output", base, "nussl", f"{i}.wav"), 
                 y.T, samplerate=feature_sr)
        extract_features(os.path.join("output", base, "features", f"{i}.csv"),
                         y, args.frame_length, args.hop_length, feature_sr)
    
    # Write dummy files so we have the same number every time
    total_num_files = 7 + MAX_OTHER_SOURCES
    empty_file = np.zeros_like(sources[0])
    num_frames = empty_file.shape[1] // args.hop_length
    empty_frames = [DataFrame(0, 0, 0, 0, 0) for _ in range(num_frames)]
    empty_dicts = [asdict(frame) for frame in empty_frames]
    fieldnames = empty_dicts[0].keys()

    for i in range(7 + args.num_sources, total_num_files):
        sf.write(os.path.join("output", base, "nussl", f"{i}.wav"),
                 empty_file.T, samplerate=feature_sr)
        csv_filename = os.path.join("output", base, "features", f"{i}.csv")
        with open(csv_filename, mode='w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(empty_dicts)

    shutil.copyfile(args.input_path, os.path.join("output", base, f"{base}.wav"))

    print("\nDone!\n")
 

if __name__ == "__main__":
    main()
