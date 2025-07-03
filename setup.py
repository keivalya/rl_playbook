# setup.py

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="rl_playbook",
    py_modules=["rl_playbook"],
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "numpy",
    ],
    author="Keivalya Pandya",
    author_email="keivalyapandya2001@gmail.com",
    description="A library for reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/keivalya/rl_playbook",
)