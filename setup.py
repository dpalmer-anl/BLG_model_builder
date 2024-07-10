from setuptools import setup, find_packages

setup(
    name="BLG_model_builder",
    version="0.1",
    author="Daniel Palmer, and Harley T. Johnson",
    author_email="dpalmer3@illinois.edu",
    packages=find_packages(),
    install_requires=["joblib","dask","ase","h5py","pandas"],
    include_package_data=True,
)

