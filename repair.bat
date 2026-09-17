@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
set "ROOT=%CD%"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

rem Rebuild the Python environment: delete runtime\venv and runtime\install-state.json, then run start.bat.
rem Downloads already in runtime\cache are reused; torch (1.9 GB) is downloaded again.
rem ASCII-only and CRLF, like start.bat.

if exist "%ROOT%\.dev" goto dev
if not exist "%ROOT%\launcher\msg\repair_intro.txt" goto missing
type "%ROOT%\launcher\msg\repair_intro.txt"
pause

rem Nothing may still run from runtime (the server, llama-server, an install in another window):
rem setup_runtime.py checks that first and only then deletes (exit 2 = still in use, nothing deleted).
set "BASEPY=%ROOT%\runtime\python\cpython-3.12.14-windows-x86_64-none\python.exe"
if not exist "%BASEPY%" goto plain_delete
"%BASEPY%" -I -X utf8 "%ROOT%\launcher\setup_runtime.py" --repair-reset
if errorlevel 2 goto in_use
if errorlevel 1 goto failed
goto run_start

:plain_delete
rem Without runtime\python nothing can be running from runtime\venv.
if exist "%ROOT%\runtime\install-state.json" del /q "%ROOT%\runtime\install-state.json"
if exist "%ROOT%\runtime\venv" rmdir /s /q "%ROOT%\runtime\venv"

:run_start
set "VS_REPAIR=1"
"%ROOT%\start.bat" %*

:in_use
type "%ROOT%\launcher\msg\repair_in_use.txt"
pause
exit /b 1

:failed
echo.
pause
exit /b 1

:dev
type "%ROOT%\launcher\msg\repair_dev.txt"
pause
exit /b 1

:missing
echo.
echo 找不到 launcher 資料夾。請先把整個壓縮檔解壓縮到 D:\BilingualSubtitles 這類位置，再從那裡打開 repair.bat。
echo.
pause
exit /b 1
