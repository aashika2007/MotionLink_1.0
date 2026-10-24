@echo off
REM MotionInk: install requirements, start the server, open the dashboard.
cd /d "%~dp0"
if not exist app.py (
  echo Run this from the folder that contains app.py.
  pause
  exit /b 1
)
python -m pip install -r requirements.txt || goto :fail
start "" http://127.0.0.1:5000
python app.py
exit /b 0
:fail
echo.
echo Installing failed. Copy the messages above and share them.
pause
exit /b 1
