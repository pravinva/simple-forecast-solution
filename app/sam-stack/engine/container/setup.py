import setuptools

setuptools.setup(
    name="enginemap",
    version="0.0.1",
    description="enginemap",
    author="author",
    packages=setuptools.find_packages(where="enginemap"),
    install_requires=[
        "aws-cdk.core==1.97.0",
    ],
    python_requires=">=3.6"
)
