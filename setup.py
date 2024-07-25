from setuptools import find_packages, setup

entry_points = {
    "console_scripts": [
        "ml-server=ml_server.cli:server",
        "orca=ml_server.cli:orca_client",
        "ml-stop=ml_server.cli:stop_server",
    ]
}

setup(
    name="ml_server",
    version="0.1.0",
    url="https://github.com/Molecolab-Pisa/ML-server",
    author="Patrizia Mazzeo and Edoardo Cignoni and Lorenzo Cupellini and Benedetta Mennucci",
    author_email="mazzeo.patrizia.1998@gmail.com, edoardo.cignoni96@gmail.com",
    description=open("README.md").read(),
    entry_points=entry_points,
    packages=find_packages(),
    install_requires=["numpy"],
)
