"""
Setup script for the Text Summarization project
"""

from setuptools import setup, find_packages

# Read the contents of README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="text-summarization-ai",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered text summarization using LSTM and Attention mechanisms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/text-summarization-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "visualization": [
            "graphviz>=0.20",
            "pydot>=1.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "textsum-train=train:main",
            "textsum-test=test_models:main",
            "textsum-visualize=visualize:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/text-summarization-ai/issues",
        "Source": "https://github.com/yourusername/text-summarization-ai",
        "Documentation": "https://github.com/yourusername/text-summarization-ai/wiki",
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.txt", "*.md"],
    },
    keywords="text summarization, nlp, machine learning, deep learning, lstm, attention, tensorflow",
    zip_safe=False,
)