# Electricity Forecasting Platform

This repository contains a multi-station electricity demand forecasting platform. FastAPI manages station configuration, model experiments, and forecast results. A Streamlit dashboard supports station management, model comparison, pipeline controls, and live monitoring. Kafka and Spark handle streaming observations; models run in the station predictor.

## Forecast models and analytics

- Demand regression: LSTM, Random Forest, Gradient Boosted Trees, XGBoost, and K-Means cluster-average forecasts.
- Demand classification: GBT experiments train an auxiliary One-vs-Rest classifier. Its classes use training-only thresholds: the top 5% of training demand is `Spike`; the remaining demand is divided into `Low`, `Medium`, and `Normal` bands.
- K-Means: standardizes tabular lag and calendar features, assigns future intervals to a learned cluster, and uses that cluster's training-average demand as the MW estimate. Forecast rows include the selected cluster ID.
- Evaluation: chronological validation metrics include MAE, RMSE, R², WAPE, and bias. GBT experiments also report class accuracy and macro F1; K-Means reports Silhouette score and cluster profiles.

Class predictions and K-Means cluster IDs are added to saved forecast records when the matching model is deployed. Cluster numbers are arbitrary labels. K-Means forecasts are cluster averages, so values can repeat across intervals.

## Project structure

- `backend/`: FastAPI endpoints, station lifecycle, model experiments, and forecast results.
- `dashboard/`: Streamlit home page and station, model, pipeline, and monitoring pages.
- `src/features/`, `src/models/`, `src/spark/`, and `src/storage/`: feature definitions, training and serving, stream processing, and CSV persistence.
- `config/`: station, model, Kafka, Spark, and algorithm settings.

## Data contract

The existing multi-station workflow forecasts a **demand/load** series. Temperature is optional at registration; when absent, the loader writes a constant zero compatibility value and records that no measured temperature was provided. Models trained on that station therefore cannot learn temperature effects. Additional numeric columns, such as generation, are not currently forecast targets in this repository's station, Kafka, or forecast-storage contract.

Onboarding averages source observations into the selected interval and preserves empty buckets as gaps. Each continuous segment gets its own chronological training/validation split. Forecast windows are formed only when the full history and target lie inside one segment; segments are never stitched together. The latest segment must be long enough to seed the demo stream. Resampling does not fill long gaps with synthetic observations.
