@echo off
rem Starts the SNR + Invariant Residual (SNR_R) prototype and opens it in the browser (http://127.0.0.1:8765).
rem Close this window to stop it.
cd /d "%~dp0..\..\.."
if not exist ".venv\Scripts\python.exe" (
  echo The project environment .venv is missing. See code\scheduling\prototype\README.md, "Set up once".
  pause
  exit /b 1
)
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" "code\scheduling\prototype\server.py" %*
pause
