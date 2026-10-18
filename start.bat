@echo off
setlocal

REM ================================================================
REM  AIHB - Start backend (Flask 6006) and frontend (Vite 5173)
REM ================================================================

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found: .venv
    echo Please run setup_env.bat first.
    pause
    exit /b 1
)

echo ================================================================
echo  AIHB services
echo    Backend API       : http://127.0.0.1:6006
echo    Frontend (Vite)   : http://127.0.0.1:5173
echo    Frontend (Flask)  : http://127.0.0.1:6006/app/    (needs frontend\dist)
echo    API list          : http://127.0.0.1:6006/app/#/api
echo ================================================================
echo.

start "AIHB Backend" cmd /k ".venv\Scripts\python.exe flask_server.py"

if not exist "frontend\node_modules" goto :no_frontend

start "AIHB Frontend" cmd /k "cd frontend && npm run dev"
echo Backend + frontend started in two windows.
echo Close those windows (or press Ctrl+C inside them) to stop the services.
echo.
echo Opening frontend (Vite dev) in browser ...
timeout /t 6 >nul
start "" "http://127.0.0.1:5173"
goto :end

:no_frontend
echo [WARN] frontend\node_modules not found.
echo        Please run setup_env.bat to install frontend dependencies.
echo        Backend only started. You can still use http://127.0.0.1:6006/app/
echo        if frontend\dist was built.

:end
echo.
pause
endlocal
