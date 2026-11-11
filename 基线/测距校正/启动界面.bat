@echo off
set "PYTHON_EXE=D:\Software\anaconda\envs\1dcnn\python.exe"
if exist "%PYTHON_EXE%" (
  "%PYTHON_EXE%" -X utf8 "%~dp0distance_calibration_gui.py"
) else (
  py -3 -X utf8 "%~dp0distance_calibration_gui.py"
)
