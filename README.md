# Perceptual Visualizer

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.
The goal of the Perceptual Visualizer is to produce a visualization of a stereo mix where distinct sources are represented based on their percieved spatial location. We achieve this by first source-separating the input using a combination of deep-learning libraries (Demucs and LarsNet) and unsupervised, primitive separation algorithms provided by the nussl library, and then running another script that extimates the spatial positions of the separated sources, and finally a TouchDesigner program that produces an artful visualization.

## Visuals
Coming soon...

## Installation and Usage
This project strings together a number of tools and programs, not all of which were designed to be run in series like this. As a result, after cloning this repository, getting it up and running takes a few more steps. The following is one way to set things up that will probably work, although you may need to do a few more things depending on your system and configuration.

### Step 1: Set up your environment
First, clone this repository:
```
git clone https://github.com/MckinleyWood/perceptual-visualizer.git
```
This project uses older libraries that do not work with the latest python versions. If you do not have Python 3.10 installed, you will have to do that first. Navigate to the root directory and create a new virtual environment:
```
python3.10 -m venv venv
```
Finally, activate your new vitual environment:
```
source venv/bin/activate
```
On Windows, the commands will look more like this:
```
git clone https://github.com/MckinleyWood/perceptual-visualizer.git
cd perceptual-visualizer
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\activate
```

### Step 2: Install Libraries
This project uses [Demucs](https://github.com/adefossez/demucs) and [nussl](https://github.com/nussl/nussl) to do sound source separation. You can install them with `pip`:
```
pip install demucs nussl
```
You will also need a few non-Python tools, namely [SoX](https://sourceforge.net/projects/sox/) and [FFmpeg](https://ffmpeg.org/download.html), so make sure you have those installed as well.

### Step 3: Install LarsNet
This project uses [LarsNet](https://github.com/polimi-ispl/larsnet) for its individual drum separation. LarsNet is not really set up to work as a Python library, so we have made a few minor changes to its code for nicer integration. To install, navigate to `larsnet_src/` and run 
```
pip install .
```
You will then have to download the pretrained models from the [LarsNet page](https://drive.usercontent.google.com/download?id=1U8-5924B1ii1cjv9p0MTPzayb00P4qoL&export=download&authuser=0), unzip them, and copy the folder into the larsnet folder at `venv/lib/python3.10/site-packages/larsnet/` (or `venv\Lib\site-packages\larsnet\` on Windows).

At this point, you should be able run the main script (I would reccomend testing with a short file because it can take quite a while to run):
```
python pvis.py path/to/audio/file.wav
```
This should produce a folder called `output` with the same format as `example_output`.

### Step 4: TouchDesigner
This project uses [TouchDesigner](https://derivative.ca/showcase) to generate the visuals. The free version (which is more than fine for this application) can be downloaded [here](https://derivative.ca/download). After installing TouchDesigner, all you need to do is open the `visual_generator.toe` file, click on the `masta` node, and then type the (exact!) name of the file you have just run `pvis.py` on, minus the extension. You can then press `f1` to fullscreen the output. More experienced TouchDesigner users can go in and change things like the colours of the shapes or make it output a video file, but we currently have not implemeted a super easy way to do that.

### Step 5: Command-line Arguments
Additional command-line arguments may be provided to the program to change certain behaviours. Below are a few examples:
```
python pvis.py path/to/audio/file.wav --num_sources=3
```
By default, the program separates Demucs' "other" stem into two separate sources. However, it may well be the case that your song has more than two spatially distinct elements in the "other" category and you would like them to be separated. In that case, you can provide an int in the range (1, 3) to specify the number of sources in the "other" category.
```
python pvis.py path/to/audio/file.wav --fps=60 
```
By default, the program outputs 30 rows of csv values (frames) per second of audio. This can be changed to whatever you want.
```
python pvis.py path/to/audio/file.wav --clustering="timbral" --tc_window=30
```
By default, the program uses nussl's spatial clustering algorithm as that is what we have had the most success with. You can change this to timbral clustering with `--clustering="timbral"` if you wish. This algorithm was causing some problems for us (never finishing on longer input) so we have added a feature whereby you can window your input into smaller chunks that the algorithm has an easier time with. The window length in seconds is set with `--tc_window`.

## Support
If you have any issues, or just want to tell me you used this and thought it was cool, hit me up at mwood@dhdev.ca.

## Authors
The Perceptual Visualizer was created by Kian Dunn, Owen Ohlson, and Mckinley Wood.

## Acnowledgement
Thank you to all of the people behind the tools and libraries that made this project possible, including Demucs, LarsNet, nussl, and all of the stuff that made those projects possible, too! Also thank you to George Tzanetakis for the awesome course that gave us the knowledge and the opportunity to do this.

## Project status
This project was completed for a course in music information retrieval at the University of Victoria. We have now finished the course, so it is likely that no more work will be done on this project.