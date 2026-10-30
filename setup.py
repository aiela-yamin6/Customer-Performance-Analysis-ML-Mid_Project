"""Install the customer-ml-lab source package."""
from setuptools import setup, find_packages

setup(name="customer-ml-lab", version="1.0.0", packages=find_packages(), install_requires=["pandas", "numpy", "scikit-learn", "imbalanced-learn", "joblib", "fastapi", "uvicorn"])
