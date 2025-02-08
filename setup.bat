@echo off
setlocal

REM Set the working directory to the directory of the current batch file
cd /d %~dp0

REM Build the docker image without cache and tag it as anitts-builder-v3
docker build --no-cache -t anitts-builder-v3 .

pause
