#!/usr/bin/pwsh

Push-Location $PSScriptRoot
& "./configure.ps1"
cmake --build ../build -j
Pop-Location
