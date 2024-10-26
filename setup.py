from setuptools import find_packages, setup

setup(
    name='ml_project',
    packages=find_packages(),
    version='0.1.0',
    description="Example of ml project",
    author="Abdullaev Gadzhimurad",
    install_requires=[
        "click==7.1.2",
        "python-dotenv>=0.5.1",
        "scikit-learn==1.5.2",
        "dataclasses==0.6",
        "pyyaml==3.11",
        "marshmallow-dataclass==8.3.0",
        "pandas==2.2.3"
    ],
    license="MIT",
)