from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="logicif",
    version="0.1.0",
    author="LogicIF Team",
    author_email="",
    description="A framework for generating algorithmic instructions from code functions and evaluating language model performance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/logicif",
    packages=["logicif"],
    package_dir={"logicif": "."},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tqdm>=4.60.0",
        "requests>=2.25.0",
        "openai>=1.0.0",
    ],
    extras_require={
        "full": [
            "transformers>=4.20.0",
            "vllm>=0.2.0",
            "numpy>=1.20.0",
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
        ],
    },
    include_package_data=True,
    package_data={
        "logicif": ["benchmark/*.jsonl", "*.json", "*.jsonl"],
    },
) 