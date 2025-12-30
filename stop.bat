@echo off
echo ===============================
echo Fully stopping Docker Desktop
echo ===============================

echo Killing Docker Desktop UI...
taskkill /IM "Docker Desktop.exe" /F >nul 2>&1
taskkill /IM "Docker Desktop Backend.exe" /F >nul 2>&1
taskkill /IM "com.docker.backend.exe" /F >nul 2>&1

timeout /t 2 >nul

echo Stopping Docker Windows service...
net stop com.docker.service >nul 2>&1

timeout /t 2 >nul

echo Shutting down WSL...
wsl --shutdown

echo ===============================
echo Docker and WSL fully stopped
echo ===============================
pause