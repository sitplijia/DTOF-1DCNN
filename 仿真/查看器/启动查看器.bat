@echo off
setlocal
if exist "D:\Software\anaconda\envs\1dcnn\pythonw.exe" (
    start "" "D:\Software\anaconda\envs\1dcnn\pythonw.exe" "%~dp0histogram_viewer.py" %*
    exit /b
)
python "%~dp0histogram_viewer.py" %*
if errorlevel 1 pause
