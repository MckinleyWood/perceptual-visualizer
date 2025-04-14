# perceptual-visualizer

to do:
* properly integrate other_separation.py
* python makes {max_num_sources} other files every time, populates the extra ones with 0s



## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation and Usage
This project strings together a number of tools and programs, not all of which were designed to be run in series like this. As a result after cloning this repository, getting it up and running takes a few more steps. The following is one way to set things up that will probably work, although you may need to do a few more things depending on your system and configuration:

### Step 1: Set up your environment
First clone this repository:
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
On Windows, the commands will look like
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
There may be a few dependencies that are not automatically installed, such as [SoX](https://sourceforge.net/projects/sox/) and [FFmpeg](https://ffmpeg.org/download.html), so watch out for that.

### Step 3: Install LarsNet
This project uses [LarsNet](https://github.com/polimi-ispl/larsnet) for its individual drum separation. LarsNet is not really set up to work as a python library, so we have made a few minor changes to its code for nicer integration. To install, navigate to `larsnet_src/` and run 
```
pip install .
```
You will then have to download the pretrained models from the [LarsNet page](https://drive.usercontent.google.com/download?id=1U8-5924B1ii1cjv9p0MTPzayb00P4qoL&export=download&authuser=0), unzip them, and copy the folder into the larsnet folder at `venv/lib/python3.10/site-packages/larsnet/` (or `venv\Lib\site-packages\larsnet\` on Windows).

At this point, you should be able run the main script (ideally with a short file because it can take quite a while to run):
```
python localizer.py path/to/audio/file.wav
```
This should produce a folder called `output` with the same format as `example_output`.

### Step 4: TouchDesigner
(Insert some stuff about where to get touchdesigner and how to use the toe here)

### Step 5: Command-line Arguments
Write about command line argument

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.