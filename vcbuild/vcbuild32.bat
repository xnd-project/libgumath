@ECHO off

if not exist dist32 mkdir dist32
if exist dist32\* del /q dist32\*

cd ..\libgumath
copy /y Makefile.vc Makefile

nmake /nologo clean
nmake /nologo

copy /y libgumath-0.2.0dev3.lib ..\vcbuild\dist32
copy /y libgumath-0.2.0dev3.dll ..\vcbuild\dist32
copy /y libgumath-0.2.0dev3.dll.lib ..\vcbuild\dist32
copy /y libgumath-0.2.0dev3.dll.exp ..\vcbuild\dist32
copy /y gumath.h ..\vcbuild\dist32

cd ..\vcbuild



