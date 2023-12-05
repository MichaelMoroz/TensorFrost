@echo off
setlocal

REM Define the list of Python versions
set PYTHON_VERSIONS=3.7 3.8 3.9 3.10 3.11 3.12

for %%v in (%PYTHON_VERSIONS%) do (
    echo Building for Python %%v
    call clean_rebuild %%v
)

endlocal