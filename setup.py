from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import os
import pybind11


class BuildExt(build_ext):
    """Custom build to set platform-specific compiler flags"""
    c_opts = {
        "msvc": ["/O2", "/std:c++17"],
        "unix": ["-O3", "-std=c++17"],
    }

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


ext_modules = [
    Extension(
        "execution_cpp",
        sources=["cpp/execution_engine.cpp", "cpp/bindings.cpp"],
        include_dirs=[pybind11.get_include(), "cpp"],
        language="c++",
    ),
    Extension(
        "backtester_cpp",
        sources=["cpp/backtester.cpp", "cpp/bindings.cpp"],
        include_dirs=[pybind11.get_include(), "cpp"],
        language="c++",
    ),
]

setup(
    name="refactored_trading_bot",
    version="2.0.0",
    description="Advanced AI + RL based trading bot with risk management and C++ acceleration",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "transformers",
        "PyYAML",
        "python-dotenv",
        "matplotlib",
        "plotly",
        "streamlit",
        "kiteconnect",
        "requests",
        "tqdm",
        "joblib",
        "pybind11",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
