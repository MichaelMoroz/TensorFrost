@echo off
setlocal enableextensions

set "REQUESTED_VERSION=%~1"
if defined REQUESTED_VERSION (
    set "PYTHON_VERSION=%REQUESTED_VERSION%"
    set "VERSION_WAS_EXPLICIT=1"
) else (
    set "PYTHON_VERSION=3.12"
    set "VERSION_WAS_EXPLICIT=0"
)

set "SCRIPT_DIR=%~dp0"
if not defined SCRIPT_DIR set "SCRIPT_DIR=.\"
for %%I in ("%SCRIPT_DIR%.") do set "REPO_ROOT=%%~fI"

set "VENV_DIR=%REPO_ROOT%\.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

set "CREATE_VENV_CMD=py -%PYTHON_VERSION% -m venv"
py -%PYTHON_VERSION% -c "import sys" >nul 2>&1
if errorlevel 1 (
    if "%VERSION_WAS_EXPLICIT%"=="1" (
        echo [TensorFrost] Python %PYTHON_VERSION% is not available via the launcher. Install it or pass a different version.
        exit /b 1
    ) else (
        echo [TensorFrost] Python %PYTHON_VERSION% not found; falling back to default interpreter for venv creation.
        set "CREATE_VENV_CMD=py -m venv"
    )
)

if exist "%VENV_PYTHON%" (
    echo [TensorFrost] Using existing virtual environment at "%VENV_DIR%"
) else (
    echo [TensorFrost] Creating virtual environment at "%VENV_DIR%"
    %CREATE_VENV_CMD% "%VENV_DIR%"
    if errorlevel 1 (
        echo [TensorFrost] Failed to create virtual environment.
        exit /b 1
    )
)

if not exist "%VENV_PYTHON%" (
    echo [TensorFrost] Could not find python interpreter in "%VENV_DIR%".
    exit /b 1
)

echo [TensorFrost] Upgrading pip inside the virtual environment...
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 (
    echo [TensorFrost] Failed to upgrade pip.
    exit /b 1
)

echo [TensorFrost] Installing TensorFrost in editable mode (verbose)...
"%VENV_PYTHON%" -m pip install -v -e "%REPO_ROOT%\Python"
if errorlevel 1 (
    echo [TensorFrost] Editable install failed.
    exit /b 1
)

echo [TensorFrost] TensorFrost development environment ready.
echo [TensorFrost] Activate it with "%VENV_DIR%\Scripts\activate.bat" or "pwsh %VENV_DIR%\Scripts\Activate.ps1".

exit /b 0
