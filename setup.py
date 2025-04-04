from setuptools import setup, find_packages

with open("pip_readme.md", "r") as fh:
    long_description = fh.read()

setup(
    name="riboclette",
    version="0.1.1",
    author="Vamsi Nallapareddy and Francesco Craighero",
    packages=find_packages(),
    package_data={
        'riboclette': ['checkpoints/XLNet-PLabelDH_S2/best_model/*'],
    },
    include_package_data=True,
    install_requires=[
        "numpy>=1.26.4",
        "transformers>=4.38.0",
        "captum>=0.8.0"
    ],
    extras_require={
        "cuda": ["torch==2.6.0+cu118"],
        "cpu": ["torch==2.6.0"],
    },
    long_description=long_description,
    long_description_content_type="text/markdown"
)
