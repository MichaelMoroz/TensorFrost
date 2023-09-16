#!/usr/bin/pwsh

Push-Location $PSScriptRoot
cmake -S .. -B ../build
Pop-Location
