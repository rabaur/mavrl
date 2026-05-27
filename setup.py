from pathlib import Path

from setuptools import setup, find_packages

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="mavrl",
    version="0.1",
    description="Unified Multi-modal Feedback using Amortized Variational Inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Raphaël Baur",
    author_email="raphaelbaur1@gmail.com",
    license="MIT",
    url="https://github.com/rabaur/mavrl",
    project_urls={
        "Source": "https://github.com/rabaur/mavrl",
    },
    packages=find_packages(),
    # configs/ are not Python subpackages (loaded by file path via
    # mavrl_experiments.config_loader), but they must ship with the package
    # so the loader can find them when mavrl_experiments is pip-installed.
    package_data={
        "mavrl_experiments": [
            "configs/experiments/*.py",
            "configs/optuna/*.py",
        ],
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "gymnasium>=0.26.0",
        "wandb>=0.13.0",
        "tabulate>=0.9.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.4.0",
        "Pillow>=8.0.0",
        "joblib>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
        "nontabular": [
            "stable-baselines3>=2.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
