from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ga-decoys",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Genetic algorithm for decoy molecule generation using RDKit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ga_decoys",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "rdkit>=2022.9.1",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
    ],
    entry_points={
        "console_scripts": [
            "ga-decoys=ga_decoys.main:main",
        ],
    },
    package_data={
        "ga_decoys": [
            "config/*.json",
            "data/*.npy",
            "input_smiles/*.smi",
            "molecule/*.json",
            "filtering/*.csv",
            "filtering/*.pkl.gz",
        ],
    },
    include_package_data=True,
)
