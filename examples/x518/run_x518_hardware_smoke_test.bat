@echo off
setlocal
chcp 65001 >nul

rem Run from the repository root even when this file is launched by double-clicking.
pushd "%~dp0\..\.."

rem Arguments: host, port, unit id, sample count, expected unit.
set "X518_TEST_HOST=%~1"
set "X518_TEST_PORT=%~2"
set "X518_TEST_UNIT_ID=%~3"
set "X518_TEST_SAMPLES=%~4"
set "X518_TEST_EXPECTED_UNIT=%~5"

if not defined X518_TEST_HOST set "X518_TEST_HOST=192.168.1.100"
if not defined X518_TEST_PORT set "X518_TEST_PORT=502"
if not defined X518_TEST_UNIT_ID set "X518_TEST_UNIT_ID=1"
if not defined X518_TEST_SAMPLES set "X518_TEST_SAMPLES=10"
if not defined X518_TEST_EXPECTED_UNIT set "X518_TEST_EXPECTED_UNIT=kg"

echo ============================================================
echo X518 read-only hardware smoke test
echo Address: %X518_TEST_HOST%:%X518_TEST_PORT%
echo Unit ID: %X518_TEST_UNIT_ID%
echo Samples: %X518_TEST_SAMPLES%
echo Expected unit: %X518_TEST_EXPECTED_UNIT%
echo ============================================================

uv run python examples\x518\x518_hardware_smoke_test.py --host "%X518_TEST_HOST%" --port "%X518_TEST_PORT%" --unit-id "%X518_TEST_UNIT_ID%" --samples "%X518_TEST_SAMPLES%" --expected-unit "%X518_TEST_EXPECTED_UNIT%"

set "X518_TEST_EXIT_CODE=%ERRORLEVEL%"
popd

rem Keep the window open after a no-argument double-click run.
if "%~1"=="" pause
exit /b %X518_TEST_EXIT_CODE%
