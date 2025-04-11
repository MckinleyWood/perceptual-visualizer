import argparse
from dataclasses import dataclass, asdict
import csv
import os
import shutil

import demucs.separate
import larsnet
import librosa
import numpy as np
import soundfile as sf

# Since nussl is deprecated, we need to patch some things for it to work
import scipy.signal.windows as sw
import scipy.signal

import larsnet.separate
np.float_ = np.float64
scipy.signal.hamming = sw.hamming
scipy.signal.hann = sw.hann
scipy.signal.blackman = sw.blackman

import nussl

import feature_extraction as fe
import other_separation as sep

# LarsNet also requires external download of pretrained models found here:
# https://drive.usercontent.google.com/download?id=1U8-5924B1ii1cjv9p0MTPzayb00P4qoL&export=download&authuser=0

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
    # x_pos = fe.x_pos(y, D, sr)
    x_pos = fe.ild(y)
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
    parser.add_argument("--separator", type=str, default="spatial")
    parser.add_argument("--num_sources", type=int, default=2)
    args = parser.parse_args()

    # Create the output directories if they don't exist.
    base = os.path.splitext(os.path.basename(args.input_path))[0]
    larsnet_path = os.path.join("output", base, "larsnet")
    demucs_path = os.path.join("output", base, "demucs")
    create_output_dirs(base)

    # Separate vocals, drums, bass, and other stems using Demucs
    # You can comment this out if you have already run Demucs.
    # print(f"\nRunning demucs on {args.input_path}...\n")
    # demucs.separate.main([args.input_path])
    # move_demucs_files(base)    

    # Separate the drums further using larsnet
    # print("\nRunning second-level \"drums\" separation with larsnet...\n")
    # shutil.copy(os.path.join(demucs_path, "drums.wav"),
    #             os.path.join(larsnet_path, "input", "drums.wav"))
    # larsnet.separate.separate(os.path.join(larsnet_path, "input"), 
    #                           larsnet_path, wiener_exponent=None, device='cpu')
    # move_larsnet_files(larsnet_path)
    
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
    print("\nRunning second-level \"other\" separation with nussl...\n")
    
    other_path = os.path.join(demucs_path, "other.wav")
    nussl_output_path = os.path.join("output", base, "nussl")

    if args.clustering == "spatial":
        sep.spatial_clustering(other_path, nussl_output_path, args.num_sources)
    elif args.clustering == "timbral":
        sep.timbral_clustering(other_path, nussl_output_path, args.num_sources)
    else:
        raise ValueError(f"Invalid clustering method: {args.clustering}")
    
    # sc_separator = nussl.separation.spatial.SpatialClustering(
    #     other, 2, clustering_type="KMeans")
    # sc_separator_gmm = nussl.separation.spatial.SpatialClustering(
    #     other, 2, clustering_type="GaussianMixture") 
    # tc_separator = nussl.separation.primitive.TimbreClustering(
    #     other, 2, 6)
    # ec_separator = nussl.separation.composite.EnsembleClustering(
    #     other, 2, [sc_separator, sc_separator_gmm, tc_separator], num_cascades=2)
    # other_split = sc_separator()

    # Create a list of the separated audio signals and resample them.
    sources = [
        vocals,
        kick,
        snare,
        toms,
        hihat,
        cymbals, 
        bass, 
        ]
    for i in range(args.num_sources):
        sources.append(nussl.AudioSignal(os.path.join(nussl_output_path, f"{i}.wav")))

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
    
    print("\nDone!\n")
 

if __name__ == "__main__":
    main()
