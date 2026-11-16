# Version 8 model version deletion

This patch was verified against the supplied `Version 8.zip`. The archive already includes station hierarchy and automatic training. Apply this deletion update directly to Version 8.

1. Stop FastAPI and Streamlit. Extract this ZIP.
2. From your Version 8 project root, the folder containing `backend` and `dashboard`, run `git apply --check "C:\path\to\Version_8_Model_Version_Delete.patch"`. If it succeeds, rerun the command without `--check`. Alternatively, copy the contents of `overlay` into a copy of the Version 8 project, keeping the same paths.
3. Restart FastAPI and Streamlit, then open **Model Management → Registry & Deployment → Delete model version**. Choose a station and model component, select an inactive version, and confirm the deletion.

Applying the patch changes code and tests only. The test installation preserved the exact bytes of the supplied station configuration, experiment registry and Amprion demand checkpoint.

The uploaded Amprion station has a configured Demand Dense NN checkpoint and no versioned registry entries for Amprion. Its list is empty until it has an eligible inactive version. Other station IDs present in the registry are outside the current station configuration.

The deletion control removes the selected inactive version's managed files. The active checkpoint, training datasets, and historical forecasts are kept.

Verification on the extracted copy included 34 passing tests, a Streamlit deletion-flow check and a registry API check with the actual Amprion configuration. The Windows runtime was outside the test environment.
