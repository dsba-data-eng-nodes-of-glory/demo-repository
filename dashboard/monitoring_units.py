"""Display conversions between interval energy and average power."""


def unit_factor(source: str, target: str, interval_minutes: int) -> float:
    """Convert a source value to the selected unit at the observation cadence."""
    if source == target:
        return 1.0
    hours = interval_minutes / 60
    return hours if source == "MW" else 1 / hours


def interval_unit_label(unit: str, interval_minutes: int) -> str:
    return f"MWh / {interval_minutes} min" if unit == "MWh" else unit
