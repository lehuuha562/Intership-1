@echo off
cd /d "%~dp0"
echo Stopping services...
docker-compose down
echo Done.
pause