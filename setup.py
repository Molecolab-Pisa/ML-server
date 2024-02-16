from setuptools import find_packages, setup

setup(
    name="sander_orca",
    version="0.0.0",
    url="https://molimen1.dcci.unipi.it/p.mazzeo/sander-orca.git",
    author="Patrizia Mazzeo, Edoardo Cignoni"
    author_email="mazzeo.patrizia.1998@gmail.com, edoardo.cignoni96@gmail.com"
    description=open("README.md").read(),
    packages=find_packages(),
    install_requires=["numpy"],
)
