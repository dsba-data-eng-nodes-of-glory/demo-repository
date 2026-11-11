@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Full Amprion replay for the October 2026 test set.
REM History seed ends 2026-09-30 23:45 UTC; first streamed row is 2026-10-01 00:00 UTC.
REM Kafka must already be running at localhost:9092.

set "ROOT=%~dp0"
if "%KAFKA_BIN%"=="" set "KAFKA_BIN=C:\kafka_2.13-4.3.1\bin\windows"
if "%ACTIVATE%"=="" set "ACTIVATE=C:\Users\hp\Anaconda3\Scripts\activate.bat"
set "SPARK_CHECKPOINT=%ROOT%checkpoints\spark_amprion_features_amprion"
set "RAW_TOPIC=amprion-raw"
set "FEATURED_TOPIC=amprion-featured"
set "STATION=amprion"
set "STREAM_CSV=data\training\amprion\demand_test.csv"
set "HISTORY_CSV=data\training\amprion\demand_train.csv"
set "OUTPUT_DIR=%ROOT%data\processed\stations\amprion"
set "PRODUCER_PROGRESS=%ROOT%data\state\amprion\producer_progress.json"
set "TOPIC_LIST=%TEMP%\amprion_kafka_topics_%RANDOM%.txt"
set "DELETE_JSON=%TEMP%\amprion_delete_records_%RANDOM%.json"

cd /d "%ROOT%"

echo ============================================================
echo AMPRION - OCTOBER 2026 HOLDOUT REPLAY
echo ============================================================
echo History: %HISTORY_CSV%
echo Stream : %STREAM_CSV%
echo Expected history end : 2026-09-30 23:45 UTC
echo Expected first stream: 2026-10-01 00:00 UTC
echo.
echo This script backs up and clears Amprion actual, demand forecast,
echo and renewable forecast outputs; clears Kafka records and the Spark
 echo checkpoint; then starts Spark, predictor, and the October holdout producer.
echo It does not delete model checkpoints or source datasets.
echo.

if not exist "%ACTIVATE%" (
  echo ERROR: Conda activation script not found: %ACTIVATE%
  goto :fail
)
if not exist "%KAFKA_BIN%\kafka-topics.bat" (
  echo ERROR: Kafka topic CLI not found: %KAFKA_BIN%\kafka-topics.bat
  goto :fail
)
if not exist "%KAFKA_BIN%\kafka-delete-records.bat" (
  echo ERROR: kafka-delete-records.bat not found in %KAFKA_BIN%
  goto :fail
)
if not exist "%ROOT%%STREAM_CSV%" (
  echo ERROR: October stream CSV not found: %STREAM_CSV%
  goto :fail
)
if not exist "%ROOT%%HISTORY_CSV%" (
  echo ERROR: Training history CSV not found: %HISTORY_CSV%
  goto :fail
)
if not exist "%ROOT%src\producers\csv_producer.py" goto :missingcode
if not exist "%ROOT%src\spark\streaming.py" goto :missingcode
if not exist "%ROOT%src\models\streaming_predictor.py" goto :missingcode

powershell -NoProfile -Command "$p=Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'src\.producers\.csv_producer|src\.spark\.streaming|src\.models\.streaming_predictor' }; if($p){$p | Select-Object ProcessId,CommandLine | Format-List; exit 1}"
if errorlevel 1 (
  echo ERROR: A producer, Spark streaming process, or predictor is still running.
  echo Stop the running processes before replaying.
  goto :fail
)

call "%KAFKA_BIN%\kafka-topics.bat" --bootstrap-server localhost:9092 --list > "%TOPIC_LIST%" 2>nul
if errorlevel 1 (
  echo ERROR: Kafka is not reachable at localhost:9092.
  goto :fail
)

choice /C YN /N /M "Reset Amprion outputs and replay from 1 October 2026? [Y/N] "
if errorlevel 2 goto :cancel

REM Back up and clear the station-isolated output files (including renewable forecasts).
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%I"
set "BACKUP_DIR=%ROOT%data\processed\replay_backups\amprion_%STAMP%"
for %%F in (actuals.csv forecasts.csv renewable_forecasts.csv) do (
  if exist "%OUTPUT_DIR%\%%F" (
    if not exist "%BACKUP_DIR%" mkdir "%BACKUP_DIR%"
    copy /Y "%OUTPUT_DIR%\%%F" "%BACKUP_DIR%\%%F" >nul
    if errorlevel 1 (
      echo ERROR: Could not back up %%F.
      goto :fail
    )
    del /Q "%OUTPUT_DIR%\%%F"
    if exist "%OUTPUT_DIR%\%%F" (
      echo ERROR: Could not clear %%F.
      goto :fail
    )
  )
)
echo Output backup: %BACKUP_DIR%

call :ENSURE_TOPIC "%RAW_TOPIC%"
if errorlevel 1 goto :fail
call :ENSURE_TOPIC "%FEATURED_TOPIC%"
if errorlevel 1 goto :fail

(
  echo {
  echo   "partitions": [
  echo     {"topic": "%RAW_TOPIC%", "partition": 0, "offset": -1},
  echo     {"topic": "%FEATURED_TOPIC%", "partition": 0, "offset": -1}
  echo   ],
  echo   "version": 1
  echo }
) > "%DELETE_JSON%"

call "%KAFKA_BIN%\kafka-delete-records.bat" --bootstrap-server localhost:9092 --offset-json-file "%DELETE_JSON%"
if errorlevel 1 (
  echo ERROR: Kafka could not clear old records.
  goto :fail
)
if exist "%DELETE_JSON%" del /Q "%DELETE_JSON%"
if exist "%TOPIC_LIST%" del /Q "%TOPIC_LIST%"

if exist "%SPARK_CHECKPOINT%" rmdir /S /Q "%SPARK_CHECKPOINT%"
if exist "%SPARK_CHECKPOINT%" (
  echo ERROR: Could not remove Spark checkpoint: %SPARK_CHECKPOINT%
  goto :fail
)

REM Kafka records and result files are now reset; discard the matching producer
REM cursor so the explicitly requested replay starts at the first holdout row.
if exist "%PRODUCER_PROGRESS%" del /Q "%PRODUCER_PROGRESS%"
if exist "%PRODUCER_PROGRESS%" (
  echo ERROR: Could not clear producer progress: %PRODUCER_PROGRESS%
  goto :fail
)

REM Start services in dependency order. The predictor seeds from the final 672
REM training rows and creates its initial 96-step forecasts before producer starts.
start "Amprion - Spark" cmd /k "cd /d ""%ROOT%"" && call ""%ACTIVATE%"" electricity-spark && python -u -m src.spark.streaming --station %STATION%"
echo Waiting 15 seconds for Spark startup...
timeout /T 15 /NOBREAK >nul

start "Amprion - Predictor" cmd /k "cd /d ""%ROOT%"" && call ""%ACTIVATE%"" electricity-spark && python -u -m src.models.streaming_predictor --station %STATION% --history-file %HISTORY_CSV% --replay"
echo Waiting 5 seconds for predictor subscription...
timeout /T 5 /NOBREAK >nul

start "Amprion - CSV Producer" cmd /k "cd /d ""%ROOT%"" && call ""%ACTIVATE%"" electricity-spark && python -u -m src.producers.csv_producer --station %STATION% --csv %STREAM_CSV% --history-file %HISTORY_CSV% --delay 0.3 --replay"

echo.
echo ============================================================
echo OCTOBER HOLDOUT REPLAY STARTED
echo ============================================================
echo Initial forecast origin: 2026-09-30 23:45 UTC
echo First target timestamp : 2026-10-01 00:00 UTC
echo Final supplied weather : 2026-10-19 23:45 UTC
echo Full 96-step forecasts require a complete future weather horizon.
echo Check logs\amprion_predictor.log for renewable forecast persist counts.
echo.
pause
exit /b 0

:ENSURE_TOPIC
set "TOPIC_TO_ENSURE=%~1"
findstr /X /C:"%TOPIC_TO_ENSURE%" "%TOPIC_LIST%" >nul
if errorlevel 1 (
  call "%KAFKA_BIN%\kafka-topics.bat" --bootstrap-server localhost:9092 --create --topic "%TOPIC_TO_ENSURE%" --partitions 1 --replication-factor 1
  if errorlevel 1 exit /b 1
) else (
  echo Topic %TOPIC_TO_ENSURE% exists.
)
exit /b 0

:missingcode
echo ERROR: A required Python module is missing under src\.
goto :fail

:cancel
echo Cancelled. No changes made.
if exist "%TOPIC_LIST%" del /Q "%TOPIC_LIST%"
exit /b 0

:fail
echo.
echo Replay aborted. Check the error above; source CSVs and trained models are preserved.
if exist "%TOPIC_LIST%" del /Q "%TOPIC_LIST%"
if exist "%DELETE_JSON%" del /Q "%DELETE_JSON%"
pause
exit /b 1
