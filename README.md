# Amprion Producer — Windows Checkpoint Save Fix

## Diagnosis
The Kafka producer acknowledges the record, then `save_producer_progress` writes a temporary JSON and atomically renames it over `data/state/amprion/producer_progress.json`. Windows can reject the replace (`WinError 5` or `WinError 32`) if a process temporarily holds the file or if the destination is read-only. The former implementation had no retry and ended the producer immediately.

This is a producer cursor persistence error, not a Demand/Solar/Wind/Residual Autoencoder model error.

## Changes
Only `src/producers/csv_producer.py` is modified:
1. Unique `.tmp` name for each write, avoiding collisions with abandoned `.tmp` files.
2. Up to 15 bounded retries for transient access/share failures; Kafka is not sent additional events during retries.
3. Clear error if the checkpoint is marked read-only; the script **does not** change file attributes automatically.
4. Persistent failure stops safely and preserves the last successfully saved cursor. The most recent acknowledged batch may be replayed once on restart (at-least-once semantics).
5. The state JSON schema, Kafka topics, forecast models, and pipeline structure are unchanged.

## Install on Windows
1. Stop the affected Amprion producer/pipeline using the dashboard. Make a backup of your existing `src/producers/csv_producer.py`.
2. From this ZIP, replace only `src/producers/csv_producer.py` under `C:\Users\hp\Documents\Nodes of Glory\Version 3`.
3. Keep `data/state/amprion/producer_progress.json` **unchanged**. Do not use `--replay` or reset Kafka. Old `.tmp` files can be left alone until the pipeline is stable.
4. From PowerShell in the project root, check the attribute and saved timestamp:

   ```powershell
   Get-Item '.\data\state\amprion\producer_progress.json' | Select-Object FullName, IsReadOnly, LastWriteTime
   Get-Content '.\data\state\amprion\producer_progress.json' | ConvertFrom-Json | Select-Object last_acked_timestamp
   ```

5. Only if `IsReadOnly` is True, remove the read-only flag on **that file only**:

   ```powershell
   attrib -R '.\data\state\amprion\producer_progress.json'
   ```

6. Restart the producer using your existing pipeline controls, and watch for `Producer cursor: resuming after last acknowledged source row`.

If it still raises `PermissionError`, inspect the file handle in Resource Monitor (CPU -> Associated Handles -> `producer_progress.json`), verify directory rights, and check whether a security/indexing process is repeatedly opening the file. Do not remove the progress file to resolve locking.

## Validation
- `tests/test_producer_resume.py` plus `tests/test_producer_progress_windows_retry.py`: **7 passed**.
- This was tested at source/unit level; live Windows file-locking is not simulated by Linux unit tests.

## Note about duplicate messages
The producer receives a broker acknowledgement *before* saving the cursor. If that final save fails, restarting from the older cursor may publish one or more already-acknowledged records again. This is expected **at-least-once** behavior; verify consumers/storage deduplicate by station and timestamp.
