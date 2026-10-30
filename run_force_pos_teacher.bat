@echo off
setlocal
title B601 MIT Arm + FORCE_POS I-P Gripper
cd /d "%~dp0"
if errorlevel 1 (
    echo Cannot open the project directory.
    pause
    exit /b 1
)

where uv >nul 2>&1
if errorlevel 1 (
    echo uv was not found. Install uv or add it to PATH, then try again.
    pause
    exit /b 1
)

rem Load native six-joint MIT defaults and saved gripper settings into the panel.
rem Device connection and motion require explicit UI actions.
set "PYTHONPATH=%~dp0src;%~dp0"
uv run --no-sync python "%~dp0examples\force_teacher\b601_desktop.py" %*
set "FORCEPOS_EXIT_CODE=%ERRORLEVEL%"
if not "%FORCEPOS_EXIT_CODE%"=="0" (
    echo.
    echo FORCE_POS Teacher failed to start or exited with an error.
    pause
)
exit /b %FORCEPOS_EXIT_CODE%
