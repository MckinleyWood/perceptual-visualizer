from setuptools import setup, find_packages

setup(
    name="larsnet",
    version="0.1",
    description="Deep Drum Source Separation with LarsNet",
    author="Alessandro Ilic Mezza and Riccardo Giampiccolo and Alberto Bernardini and Augusto Sarti",
    packages=find_packages(),
    include_package_data=True,
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
        "tqdm>=4.66.1",
        "typing-extensions>=4.7.1",
        "markupsafe>=2.1.1",
    ],
)