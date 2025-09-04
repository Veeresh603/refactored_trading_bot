from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools import Command
import pybind11
import shutil
import os


class BuildExt(build_ext):
    """Custom build to set platform-specific compiler flags and output dir"""

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
        self._copy_extensions()

    def _copy_extensions(self):
        for ext in self.extensions:
            filename = self.get_ext_filename(ext.name)
            build_path = os.path.join(self.build_lib, filename)

            # Copy to core/
            core_dir = os.path.abspath("core")
            os.makedirs(core_dir, exist_ok=True)
            shutil.copy2(build_path, core_dir)
            print(f"📦 Copied {filename} → core/")

            # Copy to cpp/
            cpp_dir = os.path.abspath("cpp")
            os.makedirs(cpp_dir, exist_ok=True)
            shutil.copy2(build_path, cpp_dir)
            print(f"📦 Copied {filename} → cpp/")


class CleanAll(Command):
    """Custom clean command to remove build artifacts"""

    user_options = []

    def initialize_options(self): pass
    def finalize_options(self): pass

    def run(self):
        if os.path.exists("build"):
            shutil.rmtree("build")
            print("🧹 Removed build/")

        for folder in ["core", "cpp"]:
            if os.path.exists(folder):
                for fname in os.listdir(folder):
                    if fname.endswith((".pyd", ".so", ".dll", ".obj", ".o", ".exp", ".lib")):
                        os.remove(os.path.join(folder, fname))
                        print(f"🧹 Removed {folder}/{fname}")


ext_modules = [
    Extension(
        "execution_engine_cpp",  # 🔥 renamed module
        sources=[
            "cpp/execution_engine.cpp",
            "cpp/bindings.cpp",
            "cpp/risk_manager.cpp",
            "cpp/indicators.cpp",
            "cpp/backtester.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            "cpp",
        ],
        language="c++",
    )
]

setup(
    name="trading_bot",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": BuildExt,
        "clean": CleanAll,
    },
    install_requires=[
        "pybind11",
        "numpy",
        "pandas",
        "pyyaml",
        "python-dotenv",
        "matplotlib",
    ],
)
