from setuptools import find_packages, setup


with open("README.md", "r") as fh:
    long_description = fh.read()

version = "0.0.1"

setup(
    name="flowley",
    version=version,
    description="Flowley: Synchronized Video-to-Audio Synthesis with Flow Matching.",
    long_description=long_description,
    packages=find_packages(),
)
