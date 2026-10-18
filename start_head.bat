@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-head\Scripts\python.exe" (
  echo Please run setup_dense_head.ps1 first.
  pause
  exit /b 1
)
echo Open http://127.0.0.1:6007/app/#/head
echo This standalone server provides head reconstruction only.
.venv-head\Scripts\python.exe head_server.py
