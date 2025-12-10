@echo off
echo ==========================================
echo   Starting AI Librarian...
echo ==========================================

:: Switch to the project folder
cd /d "%~dp0"

:: Check for Docker
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Docker is NOT running!
    echo Please start Docker Desktop and try again.
    echo.
    pause
    exit
)

:: Run the Docker command with --build to pick up code changes
echo [INFO] Updating code and starting application...
docker-compose up --build

pause