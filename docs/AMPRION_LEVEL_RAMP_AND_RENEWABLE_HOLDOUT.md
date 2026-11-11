# Amprion — Demand Level/Ramp Strips and Renewable Holdout

This patch keeps the platform focused on the single `amprion` station.

## Demand anomaly display

The demand chart now has exactly two forecast-only anomaly tracks below the
main demand curve:

1. **Demand level (historical baseline)** — highlighted only when the supplied
   Dense Autoencoder classifier returns `LEVEL_SPIKE`.
2. **Ramp-up** — highlighted only when the supplied Dense Autoencoder
   classifier returns `RAMP_UP_SPIKE`.

The implementation reuses the package's `classify_spike_types()` logic from
`src/models/anomaly/demand_ae_inference.py`. That classifier first requires a
high-demand forecast relative to the learned historical baseline and a
combined threshold exceedance. It then assigns the dominant cause to either
level or upward 30-minute ramp. Low-demand anomalies remain in the underlying
inference results but are not drawn in these two high-demand strips, matching
the supplied source logic. Normal and insufficient-context rows are blank;
there is no third strip. Actual observations are used only as history to
construct trailing windows and are never labelled on the chart.

The forecast scorer adds these fields for the dashboard:

- `demand_level_anomaly`
- `ramp_up_anomaly`
- `historical_baseline_mw`
- `demand_level_score_z`
- `ramp_up_score_z`

## Renewable test files

The supplied holdout files are installed here:

```text
data/training/amprion/solar_test.csv
data/training/amprion/wind_test.csv
```

They are referenced by the existing `config/stations.yaml` entries:

```yaml
renewables:
  solar:
    test_file: data/training/amprion/solar_test.csv
  wind:
    test_file: data/training/amprion/wind_test.csv
```

The test data covers 2026-10-01 00:00 UTC through 2026-10-19 23:45 UTC at
15-minute frequency. The serving adapter combines train and test rows to
reconstruct historical inputs, but masks future generation targets before
feature construction. The future model covariates are only the supplied
`forecast_*` columns.

## Re-run renewable holdout evaluation

From the project root:

```bash
python -m scripts.evaluate_renewable_holdout
```

The script evaluates 96-step forecasts from daily 23:45 UTC origins. For each
origin it uses the future forecast-weather columns and does not expose future
realised generation to either model. It writes:

```text
data/processed/stations/amprion/renewable_holdout_predictions.csv
data/processed/stations/amprion/renewable_holdout_metrics.csv
data/processed/stations/amprion/renewable_holdout_horizon_metrics.csv
```

Metrics are reported in MWh per 15-minute interval. Solar and wind are
measured separately. The horizon file reports accuracy for each of the 96
lead steps.

## Re-run demand anomaly holdout scoring

The package-adjusted test runner uses the packaged Amprion checkpoint, column
names and paths:

```bash
python -m scripts.test_demand_ae
```

The underlying model and inference logic follows the supplied Dense
Autoencoder scripts. Package-relative imports and the project checkpoint
location are retained so it can run from the repository root.

## Tests

Run from the project root:

```bash
pytest -q
python -m compileall -q backend dashboard src scripts tests
```
