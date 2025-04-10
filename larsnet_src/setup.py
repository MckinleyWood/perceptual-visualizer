from setuptools import setup, find_packages

setup(
    name="larsnet",
    version="0.1",
    description="Deep Drum Source Separation with LarsNet",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "cffi>=1.16.0",
        "filelock>=3.13.1",
        "jinja2>=3.1.2",
        "networkx>=3.1",
        "numpy>=1.26.2",
        "mpmath>=1.3.0",
        "pycparser>=2.21",
        "pyyaml>=6.0.1",
        "soundfile>=0.12.1",
        "sympy>=1.12",
        "torch>=2.1.2",
        "torchaudio>=2.1.2",
        # "torchtriton>=2.1.0",
        "tqdm>=4.66.1",
        # "triton>=2.1.0",
        "typing-extensions>=4.7.1",
        "markupsafe>=2.1.1",
        # Optionally include gmpy2 if it's required and installs smoothly:
        # "gmpy2>=2.1.2",
    ],
)