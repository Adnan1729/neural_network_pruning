from setuptools import setup, find_packages

setup(
    name="neural-network-pruning",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "tensorflow",
        "matplotlib",
        "jupyter",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A project demonstrating neural network pruning techniques",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/neural-network-pruning",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
