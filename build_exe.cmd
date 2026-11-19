@echo off
rem Build ban chay ngay (khong can cai Python) vao dist\HSV_Segmentation\
chcp 65001 >nul
cd /d "%~dp0"
python -m pip install -r requirements.txt pyinstaller || goto :loi
python -m PyInstaller --noconfirm --clean --windowed --onedir --name HSV_Segmentation ^
  --workpath build --distpath dist --specpath build ^
  --exclude-module matplotlib --exclude-module scipy --exclude-module skimage ^
  --exclude-module pandas --exclude-module pytest --exclude-module IPython ^
  main.py || goto :loi
rem Du lieu dat canh file .exe de nguoi dung thay va app ghi duoc outputs\
xcopy data   dist\HSV_Segmentation\data   /E /I /Y /Q >nul
xcopy config dist\HSV_Segmentation\config /E /I /Y /Q >nul
if not exist dist\HSV_Segmentation\outputs mkdir dist\HSV_Segmentation\outputs
rmdir /s /q build
echo.
echo Xong: dist\HSV_Segmentation\HSV_Segmentation.exe
exit /b 0
:loi
echo Build that bai.
pause
exit /b 1
