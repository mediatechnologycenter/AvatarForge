#!/usr/bin/env python
from distutils.core import setup

setup(
    name="avatar_backend_api",
    description="The Avatar backend API",
    version="0.1.0",
    url="https://gitlab.ethz.ch/mtc/video-synthesis/mtc-avatar-api",
    author="Media Technology Center (ETH Zürich)",
    author_email="mtc@ethz.ch",
    package_data={
        "avatar_backend_api": ["py.typed"],
    },
    packages=[
        "avatar_backend_api",
        "avatar_backend_api.clients",
        "avatar_backend_api.models",
        "avatar_backend_api.background_tools",
    ],
    install_requires=[
        "mtc_api_utils @ git+https://github.com/mediatechnologycenter/api-utils.git@0.1.0",
        "tinydb~=4.7.0",
        "python-multipart~=0.0.5",
        "httpx~=0.23.1",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ]
)
