from setuptools import find_packages, setup


setup(
    name="text_classifier",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[l.strip() for l in open("requirements.txt")],
)
