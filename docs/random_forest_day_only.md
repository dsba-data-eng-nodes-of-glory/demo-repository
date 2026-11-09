# Why Random Forest now uses the target-specific previous-day input only

Commit `2864be3` changes new Random Forest training from 38 inputs to 37 at the 15-minute interval. It keeps `demand_target_previous_day` and removes `demand_target_previous_week`. The original 36 inputs, including the weekly lag, remain.

## What problem does this address?

Adding both target-specific demand references improved average validation error compared with the original model, but the improvement varied across seasons and the summer result became worse. We therefore tested each added input separately to determine which configuration to retain.

The previous-day-only configuration had lower mean absolute error (MAE) than the configuration containing both inputs in every evaluated period. Removing the additional weekly reference follows that measured result; adding more inputs does not guarantee better forecasts.

## How these inputs differ from the original lags

The source data already contains historical demand. Feature engineering selects which historical values are supplied to each prediction row. The Random Forest receives selected inputs and summaries, rather than the entire seven-day history.

The original lag inputs are calculated once from the observed history and remain fixed across a 24-hour forecast batch. A target-specific reference selects demand at the future target timestamp minus 24 hours or 168 hours.

For example, with observations available through Tuesday 09:45 UTC:

| Forecast target | Original `demand_lag_96` | Target-specific previous-day input |
| --- | --- | --- |
| Tuesday 10:00 | Monday 10:00 demand | Monday 10:00 demand |
| Tuesday 11:00 | Monday 10:00 demand | Monday 11:00 demand |
| Tuesday 18:00 | Monday 10:00 demand | Monday 18:00 demand |

The inputs coincide for the first target and differ for later targets. The target-specific previous-week input follows the same alignment principle. Both reference calculations use fixed UTC durations and only observations available when the forecast is issued. The supervised regression label remains actual demand at the target timestamp.

## Evidence for removing the weekly reference

The recorded offline comparison used identical validation targets, training-origin selections and forest parameters for all four configurations. Lower MAE is better; units are MW.

| Validation period | Original 36 | Original + day 37 | Original + week 37 | Original + both 38 |
| --- | ---: | ---: | ---: | ---: |
| Jan–Mar 2025 | 163.46 | **129.04** | 196.15 | 158.30 |
| Jun–Aug 2025 | 121.35 | **113.84** | 133.72 | 123.71 |
| Sep–Nov 2025 | 153.87 | **126.59** | 153.77 | 127.65 |
| **Pooled** | **146.22** | **123.16** | **161.21** | **136.55** |

The previous-day-only configuration reduced pooled MAE by **15.8% versus the original** and **9.8% versus both added inputs**. Previous-week-only had **10.3% higher pooled MAE than the original**. Day-only also had the lowest pooled RMSE among these configurations: 184.77 MW, compared with 201.87 MW for both inputs.

The evaluation protocol was:

- Separate expanding training sets for the three periods; every training target preceded its validation period. Later folds could train on observations from earlier validation periods.
- Seven days of observed history and 96 quarter-hour targets per forecast origin.
- 300 training origins, giving 28,800 training examples per fit.
- 80 validation origins, giving 7,680 predictions per period and 23,040 predictions per configuration overall. Scored target blocks did not overlap.
- Fixed settings: 150 trees, maximum depth 16, minimum leaf size 2, seed 42 and one worker.
- Cached original/both-input predictions were reused after source, target and score checks; only the two missing one-input variants were fitted.

These are validation results used for feature selection. The comparisons were chosen after inspecting earlier validation results, so they are not an untouched final test. The selected configuration still needs confirmation on a reserved later period. The results apply to this dataset, feature set and forest settings; they do not establish a universal rule about weekly demand information.

## Scope and compatibility

- **New Random Forest training:** original 36 inputs plus the target-specific previous-day input, saved as feature schema `electricity_tabular_v3`.
- **Original weekly history:** `demand_lag_672`, weekly summary features and the seven-day history requirement remain.
- **Saved models:** existing 36- and 38-input artifacts continue to use their saved schemas. The weekly-reference implementation remains available for older 38-input models. Removing an input from an already-trained artifact requires retraining.
- **Other algorithms:** GBT and XGBoost retain their original 36-input contract; LSTM behavior is unchanged.
- **Parameters and deployment:** this commit changes feature selection and related tests. Forest parameters, environment settings, datasets, saved artifacts and deployment assignments are unchanged.

The feature layout identifier is separate from model version labels such as v4. New training uses the new layout; committing or pulling this code does not automatically retrain or replace an existing running model.

## Verification

The isolated staged code passed 21 focused tests covering RF-only input selection, retention of original features, equality to the tested day-only subset, future-label isolation and saved-schema inference for 36-, 37- and 38-input models.

Implementation: [feature construction](../src/models/tabular_forecasting.py), [training selection](../src/models/train_tabular.py), [feature tests](../tests/test_target_demand_features.py), and [algorithm-scope tests](../tests/test_tabular_algorithm_scope.py).
