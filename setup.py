from setuptools import find_packages, setup

entry_points = {
    "console_scripts": [
        "ml_server=sander_orca.cli:server",
        "orca=sander_orca.cli:orca_client",
    ]
}

setup(
    name="sander_orca",
    version="0.0.0",
    url="https://molimen1.dcci.unipi.it/p.mazzeo/sander_orca.git",
    author="Patrizia Mazzeo, Edoardo Cignoni",
    author_email="mazzeo.patrizia.1998@gmail.com, edoardo.cignoni96@gmail.com",
    description=open("README.md").read(),
    entry_points=entry_points,
    packages=find_packages(),
    install_requires=["numpy"],
)
