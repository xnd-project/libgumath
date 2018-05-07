set "LIBXNDDIR=%PREFIX%\Library\bin"
set "LIBXNDINCLUDE=%PREFIX%\Library\include"
cd "%RECIPE_DIR%\..\..\vcbuild" || exit 1
call vcbuild64.bat || exit 1
copy /y dist64\lib* "%PREFIX%\Library\bin\"
copy /y dist64\gumath.h "%PREFIX%\Library\include\"
