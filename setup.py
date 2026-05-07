from setuptools import setup, find_packages
import os

long_desc = open("README.md").read() if os.path.exists("README.md") else ""

setup(
    name="s4-unified-slicer",
    version="1.0.0",
    description="Single-step non-planar 4-axis slicer (STL → 4-axis G-code) with web UI and ~100x faster backend.",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    license="GPL-3.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "s4_slicer.web": ["templates/*.html", "static/*.js", "static/*.css"],
        "s4_slicer":     ["configs/*", "examples/*"],
    },
    install_requires=[
        "numpy", "scipy", "networkx", "pyvista", "tetgen",
        "pygcode", "trimesh", "numba",
        "fastapi", "uvicorn[standard]", "python-multipart",
    ],
    entry_points={
        "console_scripts": [
            "s4slicer=s4_slicer.cli:main",
            "s4slicer-web=s4_slicer.web.app:main",
        ],
    },
    python_requires=">=3.9",
)
