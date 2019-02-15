
AC_PREREQ([2.67])
AC_INIT(gumath, 0.2.0dev3, skrah@bytereef.org, gumath, https://github.com/plures/)
AC_CONFIG_HEADER(config.h)
AC_CONFIG_FILES([Makefile libgumath/Makefile])

# System and machine type (only used for Darwin):
AC_MSG_CHECKING(system as reported by uname -s)
ac_sys_system=`uname -s`
AC_MSG_RESULT($ac_sys_system)

LIBSTATIC=libgumath.a
case $ac_sys_system in
    darwin*|Darwin*)
        LIBRARY_PATH="DYLD_LIBRARY_PATH"
        LIBNAME="libgumath.dylib"
        LIBSONAME="libgumath.0.dylib"
        LIBSHARED="libgumath.0.2.0dev3.dylib"
        CONFIGURE_LDFLAGS="-dynamiclib -install_name @rpath/$LIBSONAME -undefined dynamic_lookup -compatibility_version 0.2 -current_version 0.2.0"
        ;;
    *)
        LIBRARY_PATH="LD_LIBRARY_PATH"
        LIBNAME="libgumath.so"
        LIBSONAME="libgumath.so.0"
        LIBSHARED="libgumath.so.0.2.0dev3"
        CONFIGURE_LDFLAGS="-shared -Wl,-soname,$LIBSONAME"
        ;;
esac

AC_SUBST(LIBSTATIC)
AC_SUBST(LIBRARY_PATH)
AC_SUBST(LIBNAME)
AC_SUBST(LIBSONAME)
AC_SUBST(LIBSHARED)


# Apparently purely informational for this particular build:
AC_CANONICAL_HOST
AC_SUBST(build)
AC_SUBST(host)

# Language and compiler:
AC_LANG_C
saved_cflags="$CFLAGS"
saved_cxxflags="$CXXFLAGS"
AC_PROG_CC
AC_PROG_CXX
CFLAGS="$saved_cflags"
CXXFLAGS="$saved_cxxflags"

# ar and ranlib:
AC_CHECK_TOOL(AR, ar, ar)
AC_PROG_RANLIB
AC_SUBST(RANLIB)


# Checks for header files:
AC_HEADER_STDC
AC_CHECK_HEADER([pthread.h],
  [AC_DEFINE([HAVE_PTHREAD_H], 1, [Define to 1 if you have the <pthread.h> header file.])],
  [])

# Install program:
AC_PROG_INSTALL
AC_SUBST(INSTALL)

# Cuda compiler:
AC_MSG_CHECKING(for nvcc)
saved_cc="$CC"
saved_cflags="$CFLAGS"
saved_cxxflags="$CXXFLAGS"
CC=nvcc
AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
void
f(void)
{
    return;
}
]])],
[have_nvcc=yes],
[have_nvcc=no],
[have_nvcc=undefined])
CC="$saved_cc"
CFLAGS="$saved_cflags"
CXXFLAGS="$saved_cxxflags"
AC_MSG_RESULT($have_nvcc)

CUDA_CXX=
CONFIGURE_CUDA_CXXFLAGS=
if test "$have_nvcc" = yes; then
    if nvcc -o tools/detect_cuda_arch tools/detect_cuda_arch.cc > /dev/null 2>&1; then
        CUDA_CXX="nvcc"
        CUDA_ARCH=`./tools/detect_cuda_arch`
        CONFIGURE_CUDA_CXXFLAGS="-std=c++11 -arch=compute_$CUDA_ARCH -code=sm_$CUDA_ARCH"
        AC_DEFINE(HAVE_CUDA, 1, [Define to 1 if you have the nvcc cuda compiler.])
    fi
fi
AC_SUBST(CUDA_CXX)
AC_SUBST(CONFIGURE_CUDA_CXXFLAGS)

# Add an explicit include directory.
AC_MSG_CHECKING(for --with-includes)
AC_ARG_WITH(includes,
            AS_HELP_STRING([--with-includes="/usr/local/include"],
            [absolute path to an include directory]),
            [],
            [with_includes="none"])

AC_MSG_RESULT($with_includes)

if test "$with_includes" = "none"; then
    CONFIGURE_INCLUDES=../ndtypes/libndtypes
    CONFIGURE_INCLUDES_TEST=../../ndtypes/libndtypes
else
    CONFIGURE_INCLUDES=$with_includes
    CONFIGURE_INCLUDES_TEST=$with_includes
fi

AC_SUBST(CONFIGURE_INCLUDES)
AC_SUBST(CONFIGURE_INCLUDES_TEST)

# Add an explicit library path.
AC_MSG_CHECKING(for --with-libs)
AC_ARG_WITH(libs,
            AS_HELP_STRING([--with-libs="/usr/local/lib"],
            [absolute path to a library directory]),
            [],
            [with_libs="none"])

AC_MSG_RESULT($with_libs)

if test "$with_libs" = "none"; then
    CONFIGURE_LIBS=../ndtypes/libndtypes
    CONFIGURE_LIBS_TEST=../../ndtypes/libndtypes
else
    CONFIGURE_LIBS=$with_libs
    CONFIGURE_LIBS_TEST=$with_libs
fi

AC_SUBST(CONFIGURE_LIBS)
AC_SUBST(CONFIGURE_LIBS_TEST)

# Without documentation:
AC_MSG_CHECKING([for --with-docs])
AC_ARG_WITH([docs],
  AS_HELP_STRING([--with-docs], [install documentation - enabled by default]),,
  with_docs=yes)
AC_MSG_RESULT([$with_docs])
if test "$with_docs" = yes; then
   NDT_INSTALL_DOCS="install_docs"
else
   NDT_INSTALL_DOCS=""
fi

AC_SUBST(NDT_INSTALL_DOCS)

# Compiler dependent settings:
GM_WARN=
GM_OPT="-O2"
case $CC in
    *gcc*)
        GM_WARN="-Wall -Wextra -std=c11 -pedantic"
        GM_OPT="-O3"
        ;;
    *icc*)
        AR=xiar
        GM_WARN="-Wall"
        GM_OPT="-O3"
        ;;
    *clang*)
        GM_WARN="-Wall -Wextra -std=c11 -pedantic"
        GM_OPT="-O3"
        ;;
esac



# Substitute variables and generate output:
if test -z "$LD"; then
    LD="$CXX"
fi
AC_SUBST(LD)
AC_SUBST(AR)
AC_SUBST(GM_WARN)
AC_SUBST(GM_OPT)


if test -z "$CFLAGS"; then
    CONFIGURE_CFLAGS="$GM_INCLUDE $GM_WARN $GM_CONFIG $GM_OPT"
else
    CONFIGURE_CFLAGS="$GM_INCLUDE $GM_WARN $GM_CONFIG $GM_OPT $CFLAGS"
fi

if test -z "$CXXFLAGS"; then
    CONFIGURE_CXXFLAGS="$XND_INCLUDE -Wall -Wextra -std=c++11 $GM_OPT -g"
else
    CONFIGURE_CXXFLAGS="$XND_INCLUDE -Wall -Wextra -std=c++11 $GM_OPT -g $CXXFLAGS"
fi

if test -z "$LDFLAGS"; then
    CONFIGURE_LDFLAGS="$GM_LINK $CONFIGURE_LDFLAGS"
else
    CONFIGURE_LDFLAGS="$GM_LINK $CONFIGURE_LDFLAGS $LDFLAGS"
fi

AC_SUBST(CONFIGURE_CFLAGS)
AC_SUBST(CONFIGURE_CXXFLAGS)
AC_SUBST(CONFIGURE_LDFLAGS)

AC_OUTPUT

