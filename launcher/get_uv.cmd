@echo off
rem Download uv from PyPI with the curl.exe and tar.exe that ship with Windows 10 1803+, verify sha256, extract uv.exe.
rem Called by start.bat with RT set. Must match "uv" in launcher\runtime-manifest.json (tests\test_launcher.py checks).
rem PyPI is used instead of GitHub: the same uv.exe, but GitHub releases were about 100 times slower in testing.
if not defined RT exit /b 1
set "UVVER=0.12.15"
set "UVWHL=%RT%\downloads\uv-0.12.15-py3-none-win_amd64.whl"
set "UVSHA=39af7d437b6d1f962f11c0b77f891d302ea675a0ebbcfe9792f633d843740a25"
set "UVURL=https://files.pythonhosted.org/packages/ac/d8/0b4fdfb56c79a307237295f8a0f7c64a34d5d1e8c090fab0bc1201cab2a8/uv-0.12.15-py3-none-win_amd64.whl"
set "UVLOG=%RT%\logs\install.log"

if not exist "%UVWHL%" goto download
certutil -hashfile "%UVWHL%" SHA256 | findstr /i /x "%UVSHA%" >nul
if not errorlevel 1 goto extract

:download
echo [get_uv] downloading %UVURL% >>"%UVLOG%"
curl.exe -fL --retry 5 --retry-delay 3 --connect-timeout 30 -C - -o "%UVWHL%" "%UVURL%" >>"%UVLOG%" 2>&1
certutil -hashfile "%UVWHL%" SHA256 | findstr /i /x "%UVSHA%" >nul
if not errorlevel 1 goto extract
rem a broken partial file: start over once
echo [get_uv] sha256 mismatch, downloading again >>"%UVLOG%"
del /q "%UVWHL%" 2>nul
curl.exe -fL --retry 5 --retry-delay 3 --connect-timeout 30 -o "%UVWHL%" "%UVURL%" >>"%UVLOG%" 2>&1
certutil -hashfile "%UVWHL%" SHA256 | findstr /i /x "%UVSHA%" >nul
if errorlevel 1 goto bad

:extract
if exist "%RT%\downloads\uvwhl" rmdir /s /q "%RT%\downloads\uvwhl"
mkdir "%RT%\downloads\uvwhl"
tar.exe -xf "%UVWHL%" -C "%RT%\downloads\uvwhl" "uv-%UVVER%.data/scripts/uv.exe" >>"%UVLOG%" 2>&1
if errorlevel 1 goto bad
copy /y "%RT%\downloads\uvwhl\uv-%UVVER%.data\scripts\uv.exe" "%RT%\uv\uv.exe" >nul
if errorlevel 1 goto bad
rmdir /s /q "%RT%\downloads\uvwhl"
echo [get_uv] uv %UVVER% ready >>"%UVLOG%"
exit /b 0

:bad
echo [get_uv] failed >>"%UVLOG%"
if exist "%RT%\uv\uv.exe" del /q "%RT%\uv\uv.exe"
exit /b 2
