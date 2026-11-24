"""Pure helpers for the ingestion DAG (no Airflow / GX imports).

Kept out of the DAG file so the logic can be unit-tested in CI.
"""

import pandas as pd

STATS_COLUMNS = (
    "file_name",
    "total_rows",
    "valid_rows",
    "invalid_rows",
    "criticality",
    "null_count",
    "type_error_count",
    "range_count",
    "consistency_count",
    "format_count",
    "duplicate_count",
    "schema_error",
)


def split_good_bad(
    df: pd.DataFrame, bad_indices: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into (good, bad) by row position."""
    bad = set(bad_indices)
    mask = df.index.isin(bad)
    return df[~mask], df[mask]


def build_stats_row(v: dict) -> tuple:
    """Map a validate_dataframe() result to one ingestion_stats row,
    in STATS_COLUMNS order."""
    errors = v["error_counts"]
    return (
        v["file_name"],
        v["total_rows"],
        v["total_rows"] - v["invalid_rows"],
        v["invalid_rows"],
        v["criticality"],
        errors["null"],
        errors["type_error"],
        errors["range"],
        errors["consistency"],
        errors["format"],
        errors["duplicate"],
        v["schema_error"],
    )


def report_reference(report: str, base_url: str) -> str:
    """Data Docs report for the alert: a markdown link when nginx serves the
    Data Docs (base_url set, Defense 2), otherwise the report path."""
    # Teams renders card text as markdown: "__none__" in the GX path would
    # turn into bold "none", so underscores are escaped in the visible text
    label = report.replace("_", "\\_")
    if report and base_url:
        return f"[{label}]({base_url.rstrip('/')}/{report})"
    return label or "not generated (schema error)"


def build_teams_alert(v: dict, base_url: str = "") -> dict:
    """Teams message (Adaptive Card) for a data quality validation result."""
    from utils.teams import adaptive_card

    level = v["criticality"].upper()
    errors = ", ".join(
        f"{name}: {count}" for name, count in v["error_counts"].items() if count
    )
    facts = [
        ("File", v["file_name"]),
        (
            "Invalid rows",
            f"{v['invalid_rows']}/{v['total_rows']} ({v['percent_invalid']}%)",
        ),
        ("Missing columns", ", ".join(v["missing_columns"]) or "none"),
        ("Errors", errors or "none"),
        ("Report", report_reference(v.get("report_filename", ""), base_url)),
    ]
    color = "Attention" if level == "HIGH" else "Warning"
    return adaptive_card(f"Data Quality Alert - {level}", facts, color)
