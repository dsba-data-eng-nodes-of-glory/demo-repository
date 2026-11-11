"""Two-stage residual-load anomaly inference, one row per 15-minute timestamp."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.models.residual.residual_ae_common import (
    AE_GROUPS, FEATURE_NAMES, PHYSICAL_FEATURES, STAT_SCREEN_FEATURES, WINDOW_STEPS,
    combine_history_test, extract_windows, make_features,
    statistical_scores, valid_window_starts,
)
from src.models.residual.residual_ae_model import DenseAutoencoder


def load_model(bundle: dict, device: torch.device | None = None):
    if bundle.get("schema") != "residual_ae_v2_capacity" or list(bundle.get("feature_names", [])) != FEATURE_NAMES:
        raise ValueError("Incompatible residual-load checkpoint. Train a new model using train_residual_ae.py.")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseAutoencoder(
        input_dim=WINDOW_STEPS * len(FEATURE_NAMES),
        hidden_dim=bundle["hidden_dim"], bottleneck_dim=bundle["bottleneck_dim"],
    ).to(device)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    return model, device


def ae_group_errors(
    model: torch.nn.Module, scaled: np.ndarray, starts: np.ndarray,
    device: torch.device, batch_size: int = 256,
) -> dict[str, np.ndarray]:
    """One error per group, scored only at LAST step of trailing 96-step window.

    This is memory efficient: no Nx96 error cube is stored in memory.
    """
    result = {key: np.full(len(starts), np.nan, dtype=np.float32) for key in AE_GROUPS}
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(starts), batch_size):
            block = starts[offset:offset + batch_size]
            xb = torch.from_numpy(extract_windows(scaled, block)).to(device)
            recon = model(xb)
            errors = (recon - xb).square().reshape(-1, WINDOW_STEPS, len(FEATURE_NAMES))[:, -1, :]
            for key, indices in AE_GROUPS.items():
                result[key][offset:offset + len(block)] = errors[:, list(indices)].mean(dim=1).cpu().numpy()
    return result


def ae_score_from_errors(errors: dict[str, np.ndarray], reference: dict):
    z = {}
    for group in AE_GROUPS:
        med, scale = reference[group]
        z[group] = (np.asarray(errors[group], dtype=float) - med) / scale
    z_matrix = np.column_stack([z[k] for k in AE_GROUPS])
    total = np.full(len(z_matrix), np.nan)
    valid = np.isfinite(z_matrix).all(axis=1)
    total[valid] = z_matrix[valid].max(axis=1)
    return total, z


def score_residual_timestamps(
    test: pd.DataFrame, history: pd.DataFrame | None, bundle: dict,
    batch_size: int = 256, decision_rule: str = "both",
) -> pd.DataFrame:
    """Score future forecast or historical test rows.

    `both` = statistically unusual AND unusual AE reconstruction.
    `either` = sensitive alternative for analysis (not the operational default).
    Provide ~48h of earlier history because each 24h AE window contains a 24h change feature.
    """
    if decision_rule not in ("both", "either"):
        raise ValueError("decision_rule must be 'both' or 'either'")
    if not test.index.is_unique:
        raise ValueError("Test timestamps must be unique")
    combined = combine_history_test(test, history)
    features, expected = make_features(combined, bundle["baseline"])
    raw = features.to_numpy(dtype=np.float64)
    scaled = np.full(raw.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(raw).all(axis=1)
    scaled[ok] = bundle["scaler"].transform(raw[ok]).astype(np.float32)
    test_positions = combined.index.get_indexer(test.index)
    if (test_positions < 0).any():
        raise ValueError("Some test timestamps missing from the combined timeline")

    stat_all, z_all = statistical_scores(features, bundle["stat_reference"])
    stat = stat_all[test_positions]
    stage1 = np.isfinite(stat) & (stat > bundle["stat_threshold"])
    n = len(test)
    raw_error = {k: np.full(n, np.nan, dtype=float) for k in AE_GROUPS}
    valid_starts = valid_window_starts(scaled, test_positions - (WINDOW_STEPS - 1))
    if len(valid_starts):
        model, device = load_model(bundle)
        components = ae_group_errors(model, scaled, valid_starts, device, batch_size)
        target_positions = valid_starts + WINDOW_STEPS - 1
        # Reliable even if test index is not sorted: use timestamp mapping.
        lookup = {int(pos): row for row, pos in enumerate(test_positions)}
        rows = np.array([lookup[int(pos)] for pos in target_positions], dtype=int)
        for k in AE_GROUPS:
            raw_error[k][rows] = components[k]
    stage2_score, group_z = ae_score_from_errors(raw_error, bundle["ae_reference"])
    stage2 = np.isfinite(stage2_score) & (stage2_score > bundle["ae_threshold"])
    valid = np.isfinite(stat) & np.isfinite(stage2_score)
    decision = (stage1 & stage2) if decision_rule == "both" else (stage1 | stage2)
    classification = np.where(~valid, "INSUFFICIENT_DATA", np.where(decision, "YES", "NO"))
    confirmation = np.where(~valid, "INSUFFICIENT_DATA", np.where(
        stage1 & stage2, "CONFIRMED", np.where(stage1, "STAT_ONLY", np.where(stage2, "AE_ONLY", "NORMAL"))))
    signals = z_all[test_positions]
    signal_argmax = np.argmax(np.where(np.isfinite(signals[:, :len(STAT_SCREEN_FEATURES)]),
                                             signals[:, :len(STAT_SCREEN_FEATURES)], -np.inf), axis=1)
    stat_driver = np.array(STAT_SCREEN_FEATURES, dtype=object)[signal_argmax]
    stat_driver[~np.isfinite(stat)] = "INSUFFICIENT_DATA"
    groups = list(AE_GROUPS)
    group_matrix = np.column_stack([group_z[k] for k in groups])
    group_argmax = np.argmax(np.where(np.isfinite(group_matrix), group_matrix, -np.inf), axis=1)
    ae_driver = np.array(groups, dtype=object)[group_argmax]
    ae_driver[~np.isfinite(stage2_score)] = "INSUFFICIENT_DATA"
    exp = expected[test_positions]
    residual = test["residual"].to_numpy(dtype=float)
    direction = np.where(residual >= exp, "HIGH_RESIDUAL", "LOW_RESIDUAL").astype(object)
    direction[~valid] = "INSUFFICIENT_DATA"

    out = pd.DataFrame({
        "timestamp_utc": test.index,
        "demand": test["demand"].to_numpy(dtype=float),
        "wind_energy": test["wind"].to_numpy(dtype=float),
        "solar_energy": test["solar"].to_numpy(dtype=float),
        "solar_installed_capacity_mw": test["solar_capacity_mw"].to_numpy(dtype=float),
        "wind_installed_capacity_mw": test["wind_capacity_mw"].to_numpy(dtype=float),
        "residual_load": residual,
        "expected_residual_load": exp,
        "residual_deviation": residual - exp,
        "renewable_share": features["renewable_share"].to_numpy(dtype=float)[test_positions],
        "statistical_score": stat,
        "statistical_threshold": bundle["stat_threshold"],
        "statistical_flag": np.where(np.isfinite(stat), stage1, False),
        "statistical_driver": stat_driver,
        "autoencoder_score": stage2_score,
        "autoencoder_threshold": bundle["ae_threshold"],
        "autoencoder_flag": np.where(np.isfinite(stage2_score), stage2, False),
        "autoencoder_driver": ae_driver,
        "confirmation": confirmation,
        "classification": classification,
        "anomaly_direction": direction,
        "context_valid": valid,
    })
    for name in PHYSICAL_FEATURES:
        if name not in out:
            out[name] = features[name].to_numpy(dtype=float)[test_positions]
    for k in AE_GROUPS:
        out[f"ae_{k.lower()}_error"] = raw_error[k]
        out[f"ae_{k.lower()}_z"] = group_z[k]
    return out
