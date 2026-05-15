@echo off
setlocal

cd /d "%~dp0"

set "GUI_SCRIPT=%~dp0run_gui.py"
set "VENV_PYTHONW=%~dp0.venv\Scripts\pythonw.exe"
set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"

if "%AUDIO_TRANSCRIPTION_DRY_RUN%"=="1" (
    if exist "%VENV_PYTHONW%" (
        echo "%VENV_PYTHONW%" "%GUI_SCRIPT%"
        exit /b 0
    )
    if exist "%VENV_PYTHON%" (
        echo "%VENV_PYTHON%" "%GUI_SCRIPT%"
        exit /b 0
    )
    where pythonw >nul 2>nul && (
        echo pythonw "%GUI_SCRIPT%"
        exit /b 0
    )
    where python >nul 2>nul && (
        echo python "%GUI_SCRIPT%"
        exit /b 0
    )
    echo PYTHON_NOT_FOUND
    exit /b 1
)

if exist "%VENV_PYTHONW%" (
    start "" "%VENV_PYTHONW%" "%GUI_SCRIPT%"
    exit /b 0
)

if exist "%VENV_PYTHON%" (
    start "" "%VENV_PYTHON%" "%GUI_SCRIPT%"
    exit /b 0
)

where pythonw >nul 2>nul && (
    start "" pythonw "%GUI_SCRIPT%"
    exit /b 0
)

where python >nul 2>nul && (
    start "" python "%GUI_SCRIPT%"
    exit /b 0
)

echo Python was not found. Run the setup script first.
pause
exit /b 1
