set HOSTNAME=localhost
set IOCNAME=weatherbnl
set TOP=../../
set PATH=C:\Python310;C:\Windows
set PYTHONPATH=%TOP%devsupApp\src;%TOP%weatherApp
call dllPath.bat
..\..\bin\windows-x64\softIocPy310.exe st.cmd
