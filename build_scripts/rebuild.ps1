#!/usr/bin/pwsh

Push-Location $PSScriptRoot
& "./clean.ps1"
& "./build.ps1"
Pop-Location
