from setuptools import setup, find_packages

setup(
    name="polymarket-ml-analyzer",
    version="0.1.0",
    description="ML Trading Signal Analyzer for Polymarket prediction markets",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "requests>=2.31",
        "sentence-transformers>=2.2",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "plotly>=5.15",
        "pyyaml>=6.0",
        "tqdm>=4.65",
    ],
)
