#!/usr/bin/pwsh

Push-Location $PSScriptRoot
& "./format.ps1"
& "./tidy.ps1"
Pop-Location
