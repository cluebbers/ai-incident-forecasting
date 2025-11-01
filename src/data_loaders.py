
# data_loaders.py
# Lightweight, reusable data loading + merging helpers for the AI Incidents project.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd


# -------------------------
# Config (optional)
# -------------------------

@dataclass
class CSVLoadOptions:
    """Options used when reading CSVs with pandas."""
    low_memory: bool = False
    encoding: Optional[str] = None          # e.g., "utf-8"
    na_values: Optional[Sequence[str]] = None
    keep_default_na: bool = True
    # You may add dtype: dict[str, str] if you want strict typing


# -------------------------
# Loaders
# -------------------------

def load_classifications_mit(
    path: str | Path = "../data/classifications_MIT.csv",
    opts: CSVLoadOptions = CSVLoadOptions(),
) -> pd.DataFrame:
    """
    Load the MIT classifications CSV.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"classifications_MIT file not found: {path!s}")
    df = pd.read_csv(
        path,
        low_memory=opts.low_memory,
        encoding=opts.encoding,
        na_values=opts.na_values,
        keep_default_na=opts.keep_default_na,
    )
    return df


def load_incidents(
    path: str | Path = "../data/incidents.csv",
    opts: CSVLoadOptions = CSVLoadOptions(),
) -> pd.DataFrame:
    """
    Load the incidents CSV.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"incidents file not found: {path!s}")
    df = pd.read_csv(
        path,
        low_memory=opts.low_memory,
        encoding=opts.encoding,
        na_values=opts.na_values,
        keep_default_na=opts.keep_default_na,
    )
    return df


# -------------------------
# Merge utilities
# -------------------------

def merge_incidents(
    classifications_df: pd.DataFrame,
    incidents_df: pd.DataFrame,
    left_on: str = "Incident ID",
    right_on: str = "incident_id",
    date_col: str = "date",
    how: str = "inner",
    sort_by_date: bool = True,
    drop_na_date: bool = False,
) -> pd.DataFrame:
    """
    Merge a classifications dataframe with the incidents dataframe and normalize the date column.

    Parameters
    ----------
    classifications_df : DataFrame
        The classifications table (e.g., MIT classifications).
    incidents_df : DataFrame
        The incidents table.
    left_on : str
        Key column in classifications_df.
    right_on : str
        Key column in incidents_df.
    date_col : str
        Name of the date column in the merged table (defaults to 'date').
        If not present post-merge, no conversion is attempted.
    how : str
        pandas merge how (default 'inner').
    sort_by_date : bool
        If True, sort final result by the parsed date column.
    drop_na_date : bool
        If True, drop rows where date failed to parse.

    Returns
    -------
    DataFrame
        Merged dataframe with a parsed datetime column `date_col` (if present).
    """
    # Validate keys early
    if left_on not in classifications_df.columns:
        raise KeyError(
            f"left_on='{left_on}' not found in classifications_df. "
            f"Available columns: {list(classifications_df.columns)[:20]}..."
        )
    if right_on not in incidents_df.columns:
        raise KeyError(
            f"right_on='{right_on}' not found in incidents_df. "
            f"Available columns: {list(incidents_df.columns)[:20]}..."
        )

    merged = pd.merge(
        classifications_df,
        incidents_df,
        left_on=left_on,
        right_on=right_on,
        how=how,
    )

    # Parse date column if present
    if date_col in merged.columns:
        merged[date_col] = pd.to_datetime(merged[date_col], errors="coerce")
        if drop_na_date:
            merged = merged.dropna(subset=[date_col])

        if sort_by_date:
            merged = merged.sort_values(date_col)
    else:
        # Optional: warn by returning the dataframe unchanged
        # (Users may have a differently named date field or none at all)
        pass

    return merged


def load_and_merge(
    classifications_path: str | Path = "../data/classifications_MIT.csv",
    incidents_path: str | Path = "../data/incidents.csv",
    left_on: str = "Incident ID",
    right_on: str = "incident_id",
    date_col: str = "date",
    how: str = "inner",
    sort_by_date: bool = True,
    drop_na_date: bool = False,
    cls_opts: CSVLoadOptions = CSVLoadOptions(),
    inc_opts: CSVLoadOptions = CSVLoadOptions(),
) -> pd.DataFrame:
    """
    Convenience function: load both CSVs and return the merged dataframe.
    """
    cls_df = load_classifications_mit(classifications_path, cls_opts)
    inc_df = load_incidents(incidents_path, inc_opts)
    merged = merge_incidents(
        classifications_df=cls_df,
        incidents_df=inc_df,
        left_on=left_on,
        right_on=right_on,
        date_col=date_col,
        how=how,
        sort_by_date=sort_by_date,
        drop_na_date=drop_na_date,
    )
    return merged
