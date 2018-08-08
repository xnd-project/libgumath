#
# BSD 3-Clause License
#
# Copyright (c) 2017-2018, plures
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import sys, os

if "bdist_wheel" in sys.argv:
    from setuptools import setup, Extension
else:
    from distutils.core import setup, Extension

from distutils.command.build_ext import build_ext
from distutils.sysconfig import get_python_lib
from glob import glob
import platform
import subprocess
import shutil
import warnings


DESCRIPTION = """\
Extensible array functions that operate on xnd containers.\
"""

LONG_DESCRIPTION = """\
"""

warnings.simplefilter("ignore", UserWarning)

if sys.platform == "darwin":
    LIBNAME = "libgumath.dylib"
    LIBSONAME = "libgumath.0.dylib"
    LIBSHARED = "libgumath.0.2.0dev3.dylib"
else:
    LIBNAME = "libgumath.so"
    LIBSONAME = "libgumath.so.0"
    LIBSHARED = "libgumath.so.0.2.0dev3"
    LIBNDTYPES = "libndtypes.so.0.2.0dev3"
    LIBXND = "libxnd.so.0.2.0dev3"

if "install" in sys.argv or "bdist_wheel" in sys.argv:
    CONFIGURE_INCLUDES = ["%s/ndtypes" % get_python_lib(),
                          "%s/xnd" % get_python_lib()]
    CONFIGURE_LIBS = CONFIGURE_INCLUDES
    INCLUDES = LIBS = CONFIGURE_INCLUDES
    LIBGUMATHDIR = "%s/gumath" % get_python_lib()
    INSTALL_LIBS = True
elif "conda_install" in sys.argv:
    site = ["%s/ndtypes" % get_python_lib(), "%s/xnd" % get_python_lib()]
    sys_includes = [os.path.join(os.environ['PREFIX'], "include")]
    libdir = "Library/bin" if sys.platform == "win32" else "lib"
    sys_libs = os.path.join(os.environ['PREFIX'], libdir)
    INCLUDES = CONFIGURE_INCLUDES = sys_includes + site
    LIBS = [sys_libs] + site
    LIBGUMATHDIR = "%s/gumath" % get_python_lib()
    INSTALL_LIBS = False
else:
    CONFIGURE_INCLUDES = ["../python/ndtypes", "../python/xnd"]
    CONFIGURE_LIBS = CONFIGURE_INCLUDES
    INCLUDES = LIBS = CONFIGURE_INCLUDES
    LIBGUMATHDIR = "../python/gumath"
    INSTALL_LIBS = False



PY_MAJOR = sys.version_info[0]
PY_MINOR = sys.version_info[1]
ARCH = platform.architecture()[0]
BUILD_ALL = \
    "build" in sys.argv or "install" in sys.argv or "bdist_wheel" in sys.argv


if PY_MAJOR < 3:
    raise NotImplementedError(
        "python2 support is not implemented")


def get_module_path():
    pathlist = glob("build/lib.*/")
    if pathlist:
        return pathlist[0]
    raise RuntimeError("cannot find xnd module in build directory")

def copy_ext():
    if sys.platform == "win32":
        pathlist = glob("build/lib.*/gumath/*.pyd")
    else:
        pathlist = glob("build/lib.*/gumath/*.so")
    for f in pathlist:
        shutil.copy2(f, "python/gumath")

def make_symlinks():
    os.chdir(LIBGUMATHDIR)
    os.chmod(LIBSHARED, 0o755)
    os.system("ln -sf %s %s" % (LIBSHARED, LIBSONAME))
    os.system("ln -sf %s %s" % (LIBSHARED, LIBNAME))


if len(sys.argv) == 3 and sys.argv[1] == "install" and \
    sys.argv[2].startswith("--local"):
    localdir = sys.argv[2].split("=")[1]
    sys.argv = sys.argv[:2] + [
        "--install-base=" + localdir,
        "--install-purelib=" + localdir,
        "--install-platlib=" + localdir,
        "--install-scripts=" + localdir,
        "--install-data=" + localdir,
        "--install-headers=" + localdir]

    CONFIGURE_INCLUDES = ["%s/ndtypes" % localdir, "%s/xnd" % localdir]
    INCLUDES = LIBS = CONFIGURE_LIBS = CONFIGURE_INCLUDES
    LIBGUMATHDIR = "%s/gumath" % localdir
    INSTALL_LIBS = True

    if sys.platform == "darwin": # homebrew bug
        sys.argv.append("--prefix=")

if len(sys.argv) == 2:
    if sys.argv[1] == 'module':
       sys.argv[1] = 'build'
    if sys.argv[1] == 'module_install' or sys.argv[1] == 'conda_install':
       sys.argv[1] = 'install'
    if sys.argv[1] == 'test':
        module_path = get_module_path()
        python_path = os.getenv('PYTHONPATH')
        path = module_path + ':' + python_path if python_path else module_path
        env = os.environ.copy()
        env['PYTHONPATH'] = path
        ret = subprocess.call([sys.executable, "python/test_gumath.py"], env=env)
        sys.exit(ret)
    elif sys.argv[1] == 'clean':
        shutil.rmtree("build", ignore_errors=True)
        os.chdir("python/gumath")
        shutil.rmtree("__pycache__", ignore_errors=True)
        for f in glob("_gumath*.so"):
            os.remove(f)
        sys.exit(0)
    elif sys.argv[1] == 'distclean':
        if sys.platform == "win32":
            os.chdir("vcbuild")
            os.system("vcdistclean.bat")
        else:
            os.system("make distclean")
        sys.exit(0)
    else:
        pass


def gumath_extensions():
    add_include_dirs = [".", "libgumath", "ndtypes/python/ndtypes", "xnd/python/xnd"] + INCLUDES
    add_library_dirs = ["libgumath", "ndtypes/libndtypes", "xnd/libxnd"] + LIBS
    add_depends = []

    if sys.platform == "win32":
        add_libraries = ["libndtypes-0.2.0dev3.dll", "libxnd-0.2.0dev3.dll", "libgumath-0.2.0dev3.dll"]
        add_extra_compile_args = ["/DNDT_IMPORT", "/DXND_IMPORT", "/DGM_IMPORT"]
        add_extra_link_args = []
        add_runtime_library_dirs = []

        if BUILD_ALL:
            from distutils.msvc9compiler import MSVCCompiler
            MSVCCompiler().initialize()
            os.chdir("vcbuild")
            os.environ['LIBNDTYPESINCLUDE'] = os.path.normpath(CONFIGURE_INCLUDES[0])
            os.environ['LIBNDTYPESDIR'] = os.path.normpath(CONFIGURE_LIBS[0])
            os.environ['LIBXNDINCLUDE'] = os.path.normpath(CONFIGURE_INCLUDES[1])
            os.environ['LIBXNDDIR'] = os.path.normpath(CONFIGURE_LIBS[1])
            if ARCH == "64bit":
                 os.system("vcbuild64.bat")
            else:
                 os.system("vcbuild32.bat")
            os.chdir("..")
    else:
        add_extra_compile_args = ["-Wextra", "-Wno-missing-field-initializers", "-std=c11"]
        if sys.platform == "darwin":
            add_libraries = ["ndtypes", "xnd", "gumath"]
            add_extra_link_args = ["-Wl,-rpath,@loader_path"]
            add_runtime_library_dirs = []
        else:
            add_libraries = [":%s" % LIBNDTYPES, ":%s" % LIBXND, ":%s" % LIBSHARED]
            add_extra_link_args = []
            add_runtime_library_dirs = ["$ORIGIN"]

        if BUILD_ALL:
            cflags = '"-I%s -I%s"' % tuple(CONFIGURE_INCLUDES)
            ldflags = '"-L%s -L%s"' % tuple(CONFIGURE_LIBS)
            os.system("./configure CFLAGS=%s LDFLAGS=%s && make" % (cflags, ldflags))

    def gumath_ext():
        sources = ["python/gumath/_gumath.c"]

        return Extension (
            "gumath._gumath",
            include_dirs = add_include_dirs,
            library_dirs = add_library_dirs,
            depends = add_depends,
            sources = sources,
            libraries = add_libraries,
            extra_compile_args = add_extra_compile_args,
            extra_link_args = add_extra_link_args,
            runtime_library_dirs = add_runtime_library_dirs
        )

    def functions_ext():
        sources = ["python/gumath/functions.c"]

        return Extension (
            "gumath.functions",
            include_dirs = add_include_dirs,
            library_dirs = add_library_dirs,
            depends = add_depends,
            sources = sources,
            libraries = add_libraries,
            extra_compile_args = add_extra_compile_args,
            extra_link_args = add_extra_link_args,
            runtime_library_dirs = add_runtime_library_dirs
        )

    def examples_ext():
        sources = ["python/gumath/examples.c"]

        return Extension (
            "gumath.examples",
            include_dirs = add_include_dirs,
            library_dirs = add_library_dirs,
            depends = add_depends,
            sources = sources,
            libraries = add_libraries,
            extra_compile_args = add_extra_compile_args,
            extra_link_args = add_extra_link_args,
            runtime_library_dirs = add_runtime_library_dirs
        )

    return [gumath_ext(), functions_ext(), examples_ext()]

setup (
    name = "gumath",
    version = "0.2.0dev3",
    description = DESCRIPTION,
    long_description = LONG_DESCRIPTION,
    url = "https://github.com/plures/gumath",
    author = 'Stefan Krah',
    author_email = 'skrah@bytereef.org',
    license = "BSD License",
    keywords = ["gufuncs", "array computing", "vectorization"],
    platforms = ["Many"],
    classifiers = [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: C",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development"
    ],
    install_requires = ["ndtypes == v0.2.0dev3", "xnd == v0.2.0dev3"],
    package_dir = {"": "python"},
    packages = ["gumath"],
    package_data = {"gumath": ["libgumath*", "gumath.h", "pygumath.h"]
                              if INSTALL_LIBS else ["pygumath.h"]},
    ext_modules = gumath_extensions(),
)

copy_ext()

if INSTALL_LIBS and sys.platform != "win32" and not "bdist_wheel" in sys.argv:
    make_symlinks()
