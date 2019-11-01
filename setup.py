from setuptools import setup, find_packages
import sys, os

with open('README.md', 'rb') as f:
    readme = f.read().decode('utf-8')

install_requires = []

with open('requirements.txt', 'rb') as f:
    for req in f.readlines():
        install_requires.append(req.strip())

setup(
    name='cvtorch',
    version='0.0.1',
    description='vision tools based on opencv',
    long_description=readme,
    author='iHateTa11B0y',
    author_email='1187203155@qq.com',
    install_requires=[], #install_requires,
    packages=["cvtorch"],
)
