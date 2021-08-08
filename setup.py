import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pysmtb",
    version="0.0.24",
    author="Sebastian Merzbach",
    author_email="smerzbach@gmail.com",
    description="python toolbox of (mostly) image-related helper / visualization functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smerzbach/pysmtb",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'imageio',
        'matplotlib',
        'colour-science'
    ],
    python_requires='>=3.6',
)

