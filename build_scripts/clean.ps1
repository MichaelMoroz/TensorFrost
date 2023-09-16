#!/usr/bin/pwsh

Push-Location $PSScriptRoot
Remove-Item -LiteralPath "../build" -Force -Recurse
Pop-Location