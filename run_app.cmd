@echo off
chcp 65001 >nul
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Khong tim thay Python. Hay cai Python 3.10+ hoac build ban .exe bang build_exe.cmd.
  pause
  exit /b 1
)
python -c "import cv2, numpy, PIL" >nul 2>nul
if errorlevel 1 (
  echo Dang cai thu vien can thiet - can mang Internet...
  python -m pip install -r requirements.txt
)
python main.py %*
if errorlevel 1 pause
