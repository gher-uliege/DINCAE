import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="DINCAE", # Replace with your own username
    version="1.1.0",
    author="Alexander Barth",
    description="DINCAE (Data-Interpolating Convolutional Auto-Encoder) is a neural network to reconstruct missing data in satellite observations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gher-ulg/DINCAE",
    packages=setuptools.find_packages(),
    install_requires=[
        "netCDF4>=1.4.2",
        "numpy>=1.15.4",
        "tensorflow==1.15.2",
    ],
    extras_require= {
        "test": [ "pytest-cov",
                  "codecov",
                  "pytest",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
