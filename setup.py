import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='InvertDeflectPy',
    version='0.0.1',
    author='Peter Heuer',
    author_email='pheu@lle.rochester.edu',
    description='Code for inverting deflectometry data',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/pheuer/InvertDeflectPy',
    project_urls = {
        "Bug Tracker": "https://github.com/pheuer/InvertDeflectPy/issues"
    },
    license='MIT',
    packages=['invertdeflectpy'],
    install_requires=['numpy', 'h5py', 'scipy', 'matplotlib', 'numba'],
)