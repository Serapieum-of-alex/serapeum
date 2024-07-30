from setuptools import find_packages, setup

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [line.strip() for line in open("requirements.txt").readlines()]
requirements_dev = [line.strip() for line in open("requirements-dev.txt").readlines()]

setup(
    name="serapeum",
    version="0.1.0",
    description="llm utility package",
    author="Mostafa Farrag",
    author_email="moah.farag@gmail.com",
    url="https://github.com/Serapieum-of-alex/serapeum",
    keywords=["llm", "generativeai", "chatbot", "rag", "natural language processing"],
    long_description=readme + "\n\n" + history,
    repository="https://github.com/MAfarrag/serapeum",
    documentation="https://serapeum.readthedocs.io/",
    long_description_content_type="text/markdown",
    license="GNU General Public License v3",
    zip_safe=False,
    packages=find_packages(include=["serapeum", "serapeum.*"]),
    install_requires=requirements,
    extras_require={
        "dev": requirements_dev,
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: AI",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
)
