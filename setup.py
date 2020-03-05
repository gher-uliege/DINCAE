import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DINCAE", # Replace with your own username
    version="1.0.0",
    author="Alexander Barth",
    description="DINCAE (Data-Interpolating Convolutional Auto-Encoder) is a neural network to reconstruct missing data in satellite observations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gher-ulg/DINCAE",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
