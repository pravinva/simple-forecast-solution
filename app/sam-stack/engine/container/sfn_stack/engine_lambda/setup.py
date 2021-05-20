import setuptools

setuptools.setup(
    name="voyager",
    version="1.1",
    packages=setuptools.find_packages(where="voyager"),

    install_requires=[
        "awswrangler",
        "cloudpickle==1.6.0",
        "s3fs",
        "scipy",
        "tqdm",
        "requests",
        "toolz"
    ],

    python_requires=">=3.7",

    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8"
    ],
)
