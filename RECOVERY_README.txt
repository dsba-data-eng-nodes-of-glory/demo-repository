VERSION 4 RECOVERY — 2026-11-10

Source: The user's uploaded "Version 4.zip" in this conversation.
Merged: Cassandra_All_Models_V4_Patch.zip from the latest implementation.

IMPORTANT
- Original Version 4 archive uses an internal root folder named "Version 2".
  This is how the user's uploaded ZIP was packaged; it does not mean this is the
  previous Version 2 source. We preserved the original folder naming to avoid
  changing paths unexpectedly.
- config/storage.yaml remains set to backend: csv.
- No Cassandra service is required to run CSV mode.
- This archive is a recovery copy. It does not overwrite or recover files on
  your Windows computer automatically.
- Any unuploaded local changes made after the Version 4 ZIP was shared cannot
  be recovered from this archive.

RECOVERY
1. Make a separate backup of the current local project before extracting.
2. Extract this ZIP into a NEW directory; don't merge blindly with an unknown
   or partially undone folder.
3. If needed, rename its top-level "Version 2" directory to your preferred
   project directory name after extraction.
4. Open config/storage.yaml and confirm backend: csv.
5. Activate your usual Python environment and start FastAPI/Streamlit in the
   same way you previously did.
6. Do not initialize Cassandra or run CSV->Cassandra migration while in CSV mode.
7. Verify any paths in config/stations.yaml and disk availability before
   starting the producer, Kafka, Spark or a replay.

The Cassandra patch documentation is stored at
  docs/CASSANDRA_ALL_COMPONENTS_V4.md
