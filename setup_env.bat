@echo off
setlocal enabledelayedexpansion

REM ================================================================
REM  AIHB - One-click environment setup (Windows + NVIDIA GPU)
REM  Backend : Python 3.10 venv + GPU deps + model weights
REM  Frontend: Node.js + npm install (Vue 3 + Vite)
REM ================================================================

cd /d "%~dp0"

REM ---------- 1. Locate or auto-install uv (Python version manager) ----------
set "UV_EXE="
where uv >nul 2>nul && set "UV_EXE=uv"
if not defined UV_EXE if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"
if not defined UV_EXE if exist "%LOCALAPPDATA%\Programs\uv\uv.exe" set "UV_EXE=%LOCALAPPDATA%\Programs\uv\uv.exe"
if not defined UV_EXE if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\uv.exe" set "UV_EXE=%LOCALAPPDATA%\Microsoft\WinGet\Links\uv.exe"

if not defined UV_EXE (
    echo [setup] uv not found, installing automatically ...
    echo   trying official installer ^(astral.sh^) ...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"
)

if not defined UV_EXE (
    echo   official installer failed, trying winget ...
    winget install --id astral-sh.uv -e --accept-source-agreements --accept-package-agreements
    if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"
    if not defined UV_EXE if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\uv.exe" set "UV_EXE=%LOCALAPPDATA%\Microsoft\WinGet\Links\uv.exe"
    if not defined UV_EXE (
        where uv >nul 2>nul && set "UV_EXE=uv"
    )
)

if not defined UV_EXE (
    echo   winget failed, trying pip ...
    pip install uv >nul 2>nul
    if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"
    if not defined UV_EXE (
        where uv >nul 2>nul && set "UV_EXE=uv"
    )
)

if not defined UV_EXE (
    echo [ERROR] Failed to auto-install uv. Please install manually:
    echo    powershell -c "irm https://astral.sh/uv/install.ps1 ^| iex"
    pause
    exit /b 1
)
echo [OK] uv: %UV_EXE%

REM ---------- 2. Install Python 3.10.11 ----------
echo [1/7] Installing Python 3.10.11 (required by TensorFlow 2.13 / basicsr) ...
"%UV_EXE%" python install 3.10.11
if errorlevel 1 (
    echo [ERROR] Failed to install Python 3.10.11.
    pause
    exit /b 1
)

REM ---------- 3. Create virtual environment ----------
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [2/7] Creating virtual environment .venv ...
    "%UV_EXE%" venv --python 3.10.11 --seed .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create .venv.
        pause
        exit /b 1
    )
) else (
    echo [2/7] .venv already exists, skip.
)

REM ---------- 4. Install Python dependencies ----------
echo [3/7] Installing Python dependencies (large download, please wait) ...
"%PY%" -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

REM ---------- 5. Fix pyamg (pkg_resources) ----------
echo [4/7] Pinning setuptools^<81 for pyamg compatibility ...
"%PY%" -m pip install "setuptools<81"

REM ---------- 6. Download missing model weights ----------
echo [5/7] Checking model weights ...
set "MIRROR=https://ghfast.top/https://github.com"

set "PARSING=%~dp0.venv\Lib\site-packages\facexlib\weights\parsing_parsenet.pth"
if not exist "%PARSING%" (
    echo   downloading parsing_parsenet.pth ...
    curl.exe -L --connect-timeout 20 --max-time 900 -o "%PARSING%" "%MIRROR%/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth"
    if errorlevel 1 echo   [WARN] download failed, see README for manual steps.
) else (
    echo   parsing_parsenet.pth OK
)

set "DETECT=%~dp0.venv\Lib\site-packages\facexlib\weights\detection_Resnet50_Final.pth"
if not exist "%DETECT%" (
    echo   downloading detection_Resnet50_Final.pth ...
    curl.exe -L --connect-timeout 20 --max-time 900 -o "%DETECT%" "%MIRROR%/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth"
    if errorlevel 1 echo   [WARN] download failed, see README for manual steps.
) else (
    echo   detection_Resnet50_Final.pth OK
)

set "RESRGAN=%~dp0.venv\Lib\site-packages\weights\RealESRGAN_x2plus.pth"
if not exist "%RESRGAN%" (
    echo   downloading RealESRGAN_x2plus.pth ...
    curl.exe -L --connect-timeout 20 --max-time 900 -o "%RESRGAN%" "%MIRROR%/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    if errorlevel 1 echo   [WARN] download failed, see README for manual steps.
) else (
    echo   RealESRGAN_x2plus.pth OK
)

REM ---------- 7. Frontend dependencies (Vue 3 + Vite) ----------
echo [6/7] Setting up frontend (Vue 3 + Vite) ...
where node >nul 2>nul
if errorlevel 1 (
    echo   [WARN] Node.js not found, skipping frontend setup.
    echo          Install Node.js 18+ from https://nodejs.org/ then re-run this script,
    echo          or run "npm install" inside the frontend folder manually.
) else (
    for /f "delims=" %%v in ('node --version') do echo   Node.js %%v detected.
    if exist "%~dp0frontend\package.json" (
        pushd "%~dp0frontend"
        if exist "node_modules" (
            echo   frontend\node_modules already exists, skip install.
        ) else (
            echo   running npm install (this may take a few minutes) ...
            call npm install
            if errorlevel 1 echo   [WARN] npm install failed, run it manually in frontend\.
        )
        echo   building frontend (npm run build) ...
        call npm run build
        if errorlevel 1 echo   [WARN] npm run build failed, run it manually in frontend\.
        popd
    ) else (
        echo   [WARN] frontend\package.json not found, skip.
    )
)

REM ---------- 8. Verify ----------
echo [7/7] Verifying Python environment ...
"%PY%" -c "import sys,torch;print('  Python:',sys.version.split()[0]);print('  torch :',torch.__version__);print('  CUDA  :',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
if errorlevel 1 echo   [WARN] verification failed, please check the messages above.

echo.
echo ================================================================
echo  Setup complete!
echo  Start services:  start.bat
echo    Backend API      : http://127.0.0.1:6006
echo    Frontend (Vite)  : http://127.0.0.1:5173
echo    Frontend (Flask) : http://127.0.0.1:6006/app/
echo ================================================================
pause
endlocal
