@echo off
rem One-click launcher for start_all.ps1
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_all.ps1"
echo.
echo Script finished. Press any key to close...
pause >nul
