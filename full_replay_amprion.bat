@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Full end-to-end Amprion replay from row 1 of data/raw/amprion.csv.
REM IMPORTANT: Kafka broker must already be running at localhost:9092.
REM This version DOES NOT delete/recreate Kafka topics. It uses kafka-delete-records
REM with offset -1 to advance each topic's log start to its current high watermark.

set "ROOT=%~dp0"
set "KAFKA_BIN=C:\kafka_2.13-4.3.1\bin\windows"
set "ACTIVATE=C:\Users\hp\Anaconda3\Scripts\activate.bat"
set "SPARK_CHECKPOINT=%ROOT%checkpoints\spark_amprion_features_amprion"
set "RAW_TOPIC=amprion-raw"
set "FEATURED_TOPIC=amprion-featured"
set "STATION=amprion"
set "TOPIC_LIST=%TEMP%\amprion_kafka_topics_%RANDOM%.txt"
set "DELETE_JSON=%TEMP%\amprion_delete_records_%RANDOM%.json"

cd /d "%ROOT%"

echo ============================================================
echo AMPRION - FULL END-TO-END REPLAY FROM CSV ROW 1
echo ============================================================
echo.
echo This version keeps the Kafka topics and broker running.
echo It clears existing topic records using kafka-delete-records,
echo resets the Amprion Spark checkpoint, and backs up output CSVs.
echo Raw CSVs, trained model, Kafka metadata and Kafka log folders are preserved.
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
if not exist "%ROOT%data\raw\amprion.csv" (
  echo ERROR: Stream CSV not found: data\raw\amprion.csv
  goto :fail
)
if not exist "%ROOT%data\raw\amprion_history.csv" (
  echo ERROR: History CSV not found: data\raw\amprion_history.csv
  goto :fail
)
if not exist "%ROOT%src\producers\csv_producer.py" goto :missingcode
if not exist "%ROOT%src\spark\streaming.py" goto :missingcode
if not exist "%ROOT%src\models\streaming_predictor.py" goto :missingcode

echo Checking that producer, Spark and predictor are stopped...
powershell -NoProfile -Command "$p=Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'src\.producers\.csv_producer|src\.spark\.streaming|src\.models\.streaming_predictor' }; if($p){$p | Select-Object ProcessId,CommandLine | Format-List; exit 1}"
if errorlevel 1 (
  echo ERROR: A producer, Spark streaming process, or predictor is still running.
  echo Stop those three processes with Ctrl+C, then rerun this script.
  goto :fail
)

call "%KAFKA_BIN%\kafka-topics.bat" --bootstrap-server localhost:9092 --list > "%TOPIC_LIST%" 2>nul
if errorlevel 1 (
  echo ERROR: Kafka is not reachable at localhost:9092.
  echo Start Kafka and confirm it stays running before retrying.
  goto :fail
)

choice /C YN /N /M "Start a full replay from CSV row 1? [Y/N] "
if errorlevel 2 goto :cancel

REM Back up and reset processed output CSVs.
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%I"
set "BACKUP_DIR=%ROOT%data\processed\replay_backups\%STAMP%"
if not exist "%BACKUP_DIR%" mkdir "%BACKUP_DIR%"
for %%F in (actuals.csv forecasts.csv) do (
  if exist "%ROOT%data\processed\%%F" (
    copy /Y "%ROOT%data\processed\%%F" "%BACKUP_DIR%\%%F" >nul
    if errorlevel 1 (
      echo ERROR: Could not back up %%F. Aborting.
      goto :fail
    )
    del /Q "%ROOT%data\processed\%%F"
    if exist "%ROOT%data\processed\%%F" (
      echo ERROR: Could not reset %%F.
      goto :fail
    )
  )
)
echo Output backups saved in: %BACKUP_DIR%

REM Ensure both topics exist. Do not delete or recreate them.
echo.
echo Checking Amprion Kafka topics...
call :ENSURE_TOPIC "%RAW_TOPIC%"
if errorlevel 1 goto :fail
call :ENSURE_TOPIC "%FEATURED_TOPIC%"
if errorlevel 1 goto :fail

REM Offset -1 means the current high watermark: delete all old records up to
REM the current end, without deleting topic metadata or renaming partition dirs.
(
  echo {
  echo   "partitions": [
  echo     {"topic": "%RAW_TOPIC%", "partition": 0, "offset": -1},
  echo     {"topic": "%FEATURED_TOPIC%", "partition": 0, "offset": -1}
  echo   ],
  echo   "version": 1
  echo }
) > "%DELETE_JSON%"

echo.
echo Clearing old records while retaining both Kafka topics...
call "%KAFKA_BIN%\kafka-delete-records.bat" --bootstrap-server localhost:9092 --offset-json-file "%DELETE_JSON%"
if errorlevel 1 (
  echo ERROR: Kafka could not clear the old records. Broker and topics were not intentionally deleted.
  goto :fail
)
if exist "%DELETE_JSON%" del /Q "%DELETE_JSON%"
if exist "%TOPIC_LIST%" del /Q "%TOPIC_LIST%"

echo.
echo Resetting Amprion Spark checkpoint:
echo %SPARK_CHECKPOINT%
if exist "%SPARK_CHECKPOINT%" rmdir /S /Q "%SPARK_CHECKPOINT%"
if exist "%SPARK_CHECKPOINT%" (
  echo ERROR: Could not remove the Spark checkpoint directory.
  goto :fail
)

REM Start services in dependency order. Producer is last and starts from CSV row 1.
echo.
echo Starting Spark streaming...
start "Amprion - Spark" cmd /k "cd /d ""%ROOT%"" && call ""%ACTIVATE%"" electricity-spark && python -u -m src.spark.streaming --station %STATION%"
echo Waiting 15 seconds for Spark to initialize...
timeout /T 15 /NOBREAK >nul

echo Starting LSTM predictor from the beginning of the newly-cleared featured topic...
start "Amprion - LSTM Predictor" cmd /k "cd /d ""%ROOT%"" && call ""%ACTIVATE%"" electricity-spark && python -u -m src.models.streaming_predictor --station %STATION% --history-file data/raw/amprion_history.csv --replay"
echo Waiting 5 seconds for predictor subscription...
timeout /T 5 /NOBREAK >nul

echo Starting CSV producer from the beginning of data/raw/amprion.csv...
start "Amprion - CSV Producer" cmd /k "cd /d ""%ROOT%"" && call ""%ACTIVATE%"" electricity-spark && python -u -m src.producers.csv_producer --station %STATION% --delay 0.3"

echo.
echo ============================================================
echo FULL END-TO-END REPLAY STARTED
echo ============================================================
echo Producer should begin at: 2026-01-08 00:00 UTC
echo History seed ends at:     2026-01-07 23:45 UTC
echo Kafka broker should remain running in its original window.
echo Three service windows should now be open: Spark, Predictor, Producer.
echo.
pause
exit /b 0

:ENSURE_TOPIC
set "TOPIC_TO_ENSURE=%~1"
findstr /X /C:"%TOPIC_TO_ENSURE%" "%TOPIC_LIST%" >nul
if errorlevel 1 (
  echo Topic %TOPIC_TO_ENSURE% is missing. Creating it...
  call "%KAFKA_BIN%\kafka-topics.bat" --bootstrap-server localhost:9092 --create --topic "%TOPIC_TO_ENSURE%" --partitions 1 --replication-factor 1
  if errorlevel 1 (
    echo ERROR: Could not create topic %TOPIC_TO_ENSURE%.
    exit /b 1
  )
) else (
  echo Topic %TOPIC_TO_ENSURE% exists.
)
exit /b 0

:missingcode
echo ERROR: One of the required Python module files is missing under src\.
goto :fail

:cancel
echo Cancelled. No changes made.
if exist "%TOPIC_LIST%" del /Q "%TOPIC_LIST%"
exit /b 0

:fail
echo.
echo Replay aborted. Kafka topic deletion was not requested by this script.
if exist "%TOPIC_LIST%" del /Q "%TOPIC_LIST%"
if exist "%DELETE_JSON%" del /Q "%DELETE_JSON%"
pause
exit /b 1
