@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-head\Scripts\python.exe" (
  if not exist ".venv\Scripts\python.exe" (
    echo Please run setup_env.bat first to install Python.
    exit /b 1
  )
  .venv\Scripts\python.exe -m venv .venv-head
  if errorlevel 1 exit /b 1
)
.venv-head\Scripts\python.exe -m pip install -r head_reconstruction\requirements.txt
if errorlevel 1 exit /b 1
.venv-head\Scripts\python.exe -m head_reconstruction.setup_assets
if errorlevel 1 exit /b 1
.venv-head\Scripts\python.exe -m head_reconstruction.worker --template --output head_reconstruction/assets/neutral_head.glb
if errorlevel 1 exit /b 1
echo Head reconstruction ready. Start the main service, or run start_head.bat.
