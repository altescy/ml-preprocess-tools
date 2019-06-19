from setuptools import find_packages, setup


setup(
    name="ml_preprocess_tools",
    version="0.1.0",
    author="altescy",
    author_email="altescy@fastmail.com",
    description="Experiment manager for machilen learning.",
    url="https://github.com/altescy/ml-preprocess-tools",
    license='MIT License',
    install_requires=[
        "numpy>=1.11.0",
        "scikit-learn>=0.20.0",
        "spacy>=2.1.0",
        "nltk>=3.4.0",
        "mecab-python3>=0.996.0",
        "janome>=0.3.0"],
    packages=find_packages(),
)
