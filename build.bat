@echo off
rem llama.cpp build with this fork's web UI (Windows): see build.ps1. Arguments are passed on,
rem e.g.  build.bat -Clean
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build.ps1" %*
exit /b %ERRORLEVEL%
