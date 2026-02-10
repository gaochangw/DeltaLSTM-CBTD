#!/usr/bin/env python3
from pathlib import Path
from setuptools import setup, find_packages
import re

ROOT = Path(__file__).parent
PKG  = "torchstat2"

init_text = (ROOT / PKG / "__init__.py").read_text(encoding="utf-8")
meta = {}
for key in ("__version__", "__author__"):
    m = re.search(rf'{key}\s*=\s*[\'"]([^\'"]+)[\'"]', init_text)
    if m: meta[key] = m.group(1)

req = ROOT / "requirements.txt"
install_requires = [l.strip() for l in req.read_text(encoding="utf-8").splitlines()
                    if l.strip() and not l.startswith("#")] if req.exists() else []

readme = ROOT / "README.md"
long_desc = readme.read_text(encoding="utf-8") if readme.exists() else ""

setup(
    name=PKG,
    version=meta.get("__version__", "0.0.0"),
    description="A lightweight neural network analyzer based on PyTorch",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    author=meta.get("__author__", "Phoenix"),
    author_email="fengzhe8215@outlook.com",
    packages=find_packages(exclude=("tests*", "examples*", "docs*")),
    install_requires=install_requires,
    python_requires=">=3.7",
    entry_points={"console_scripts": ["torchstat2=torchstat2.__main__:main"]},
)
