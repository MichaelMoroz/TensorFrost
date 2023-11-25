#!/usr/bin/pwsh

Push-Location $PSScriptRoot
Remove-Item -LiteralPath ../build_tid -Force -Recurse -ErrorAction SilentlyContinue
cmake -G "Ninja" -DCMAKE_CXX_CLANG_TIDY="clang-tidy" -DCMAKE_CXX_CLANG_TIDY_EXPORT_FIXES_DIR="$PSScriptRoot/../build_tidy/fixes" -S .. -B ../build_tidy
cmake --build ../build_tidy -j
clang-apply-replacements ../build_tidy/fixes
Remove-Item -LiteralPath ../build_tidy -Force -Recurse
Pop-Location
