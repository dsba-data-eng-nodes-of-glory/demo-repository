# Version 7 automatic training on registration

Apply this after the Version 7 Station Hierarchy patch. The Home page includes the station-label hotfix already.

1. Stop the station pipelines from the dashboard, then stop FastAPI and Streamlit.
2. Extract the ZIP and copy the contents of `overlay` into your project root, replacing the matching files. This also works if you already installed the separate Home hotfix. The included unified diff is based on the hierarchy patch before the Home hotfix.
3. Start Kafka, FastAPI and Streamlit. Use a single FastAPI worker as in Version 7.

Upload a workbook in Station Management, select the required model components and click **Validate workbook and register station**. Registration queues the selected models. It creates the raw and featured Kafka topics for stations with Demand selected. With Kafka offline, training can still run and Activate retries topic creation once the broker is reachable.

| Component | Initial model |
|---|---|
| Demand forecasting | Dense NN |
| Solar forecasting | Dense NN |
| Wind Energy forecasting | Dense NN |
| Residual-load anomaly detection | Dense Autoencoder |

Training uses the current settings in `config/algorithm_configs.yaml` and runs one component at a time across registered stations. Each successful model becomes the station's deployed model automatically. Wait until training finishes, then click **Activate station** in Station Management or **Activate** on Home. Existing Demand-driven activation starts Spark and the predictor. For CSV replay it also starts the producer. A live Kafka station reads its external feed. Hierarchy parent views continue using the registered substations.

Station Management shows each component's status and lets you retry failed jobs with the saved workbook. A successful Demand model allows activation after the initial queue finishes, even if an optional model failed. Previously registered stations retain their models and assignments. Future registrations use the automatic workflow.

Verified with a local Kafka broker, a real uploaded Demand workbook, a real Dense NN training run and automatic deployment. Activation was checked through its service commands, including the deployed checkpoint path. Windows and the optional models still need a full local run.
