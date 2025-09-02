from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import os
import pybind11  # ✅ import pybind11 to get include path

class BuildExt(build_ext):
    """Custom build to set platform-specific compiler flags"""
    c_opts = {
        "msvc": ["/O2", "/std:c++17"],   # ✅ force C++17 for MSVC
        "unix": ["-O3", "-std=c++17"],
    }

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == "unix":
            opts.append("-fPIC")
            opts.append("-Wall")
        elif ct == "msvc":
            opts.append("/EHsc")  # exception handling, safe C++ on MSVC

        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


# Collect all cpp files in cpp/
cpp_dir = os.path.join(os.path.dirname(__file__), "cpp")
cpp_sources = [os.path.join(cpp_dir, f) for f in os.listdir(cpp_dir) if f.endswith(".cpp")]

boost_include = os.environ.get("BOOST_ROOT", "C:/local/boost_1_84_0")  # adjust path if needed

execution_module = Extension(
    "core.execution_engine",
    sources=cpp_sources,
    include_dirs=[
        pybind11.get_include(),
        "cpp",
        boost_include  # ✅ add boost headers
    ],
    language="c++",
)

setup(
    name="refactored_trading_bot",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[execution_module],
    cmdclass={"build_ext": BuildExt},
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "stable-baselines3",
        "gym==0.21.0",
        "pyyaml",
        "streamlit",
        "kiteconnect",
        "pybind11",  # ✅ ensure pybind11 is installed
    ],
    entry_points={
        "console_scripts": [
            "trading-bot=main:main",
        ]
    },
)
