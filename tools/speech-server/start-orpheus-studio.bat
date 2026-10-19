@echo off
REM Start the Orpheus 3B LoRA fine-tuning studio (Gradio). Windows.
REM
REM The GUI is a thin Gradio app that shells out to the speech-server torch venv
REM for dataset building, LoRA training, GGUF conversion and synthesis, so it
REM runs in a gradio-only environment (no torch) and never fights the
REM torch-nightly-vs-gradio segfault.
REM
REM   ORPHEUS_GUI_ENV : gradio env (default: a local gui-env, created on first run)
REM   ORPHEUS_TTS_VENV: torch venv (default: .\venv)
REM   ORPHEUS_PORT    : GUI port (default 7861)
setlocal
set "HERE=%~dp0"
set "HERE=%HERE:~0,-1%"
if "%ORPHEUS_TTS_VENV%"=="" set "ORPHEUS_TTS_VENV=%HERE%\venv"
if "%ORPHEUS_LLAMA_CPP%"=="" set "ORPHEUS_LLAMA_CPP=%HERE%\..\.."
if "%ORPHEUS_GUI_ENV%"=="" set "ORPHEUS_GUI_ENV=%HERE%\gui-env"
if "%ORPHEUS_HOST%"=="" set "ORPHEUS_HOST=127.0.0.1"
if "%ORPHEUS_PORT%"=="" set "ORPHEUS_PORT=7861"
set "GRADIO_ANALYTICS_ENABLED=False"

if not exist "%ORPHEUS_GUI_ENV%\Scripts\python.exe" (
  echo First start: creating a gradio env in %ORPHEUS_GUI_ENV% ...
  set "PATH=%USERPROFILE%\.local\bin;%PATH%"
  uv venv --allow-existing --python 3.11 "%ORPHEUS_GUI_ENV%"
  uv pip install --python "%ORPHEUS_GUI_ENV%\Scripts\python.exe" "gradio>=5,^<6"
)
if not exist "%ORPHEUS_TTS_VENV%\Scripts\python.exe" (
  echo torch venv missing: %ORPHEUS_TTS_VENV% (run run-tts-setup.bat)
  pause
  exit /b 1
)

echo Orpheus LoRA studio: http://%ORPHEUS_HOST%:%ORPHEUS_PORT%
echo   GUI env : %ORPHEUS_GUI_ENV%
echo   torch   : %ORPHEUS_TTS_VENV%
"%ORPHEUS_GUI_ENV%\Scripts\python.exe" "%HERE%\orpheus\app.py"
pause
