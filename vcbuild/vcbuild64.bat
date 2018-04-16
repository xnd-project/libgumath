@ECHO off

if not exist dist64 mkdir dist64
if exist dist64\* del /q dist64\*

cd ..\libgumath
copy /y Makefile.vc Makefile

nmake /nologo clean
nmake /nologo

copy /y libgumath-0.2.0dev3.lib ..\vcbuild\dist64
copy /y libgumath-0.2.0dev3.dll ..\vcbuild\dist64
copy /y libgumath-0.2.0dev3.dll.lib ..\vcbuild\dist64
copy /y libgumath-0.2.0dev3.dll.exp ..\vcbuild\dist64
copy /y gumath.h ..\vcbuild\dist64

cd ..\vcbuild



