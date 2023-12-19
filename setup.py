import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='DAMIAN Environment',
    version='0.1',
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Damiano Brunori',
    author_email='brunori@diag.uniroma1.it',
    url='https://github.com/DamianoBrunori/DAMIAN-Delay-Aware-MultI-Aerial-Navigation-DRL-based-environment-',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
