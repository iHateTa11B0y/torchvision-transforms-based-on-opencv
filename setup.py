from setuptools import setup, find_packages
import sys, os

with open('README.md', 'rb') as f:
    readme = f.read().decode('utf-8')

install_requires = [
        'torch',
        'torchvision',
        'numpy',
        'opencv-python',
        ]

setup(
    name='cvtorch',
    version='0.0.6',
    description='vision tools based on opencv',
    long_description=readme,
    author='iHateTa11B0y',
    author_email='1187203155@qq.com',
    install_requires=install_requires,
    packages=["cvtorch"],

)
