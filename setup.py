import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pysmtb",
    version="0.0.1",
    author="Sebastian Merzbach",
    author_email="merzbach@cs.uni-bonn.de",
    description="python toolbox of (mostly) image-related helper / visualization functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        "matplotlib",
        "visdom",
        "torch",
        "importlib",
        "PyQt5",
        "imageio",
        "colour-science"
    ],
    python_requires=">=3.6",
)