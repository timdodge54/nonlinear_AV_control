"""setup.py install python code."""
import os

from setuptools import find_packages, setup

thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + "/requirements.txt"

install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setup(
    name="nonlinear_av_control",
    # install_requires=install_requires,
    packages=find_packages(),
)