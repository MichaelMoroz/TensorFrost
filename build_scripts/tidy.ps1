#!/usr/bin/pwsh

Push-Location $PSScriptRoot
cmake -G "Ninja" -DCMAKE_CXX_CLANG_TIDY="clang-tidy" -DCMAKE_CXX_CLANG_TIDY_EXPORT_FIXES_DIR="$PSScriptRoot/../build_tidy/fixes" -S .. -B ../build_tidy
cmake --build ../build_tidy -j
clang-apply-replacements ../build_tidy/fixes
Pop-Location
