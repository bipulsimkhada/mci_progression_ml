from pathlib import Path
import pandas as pd

from mci_progression_ml.config import cog, dem, label

SCRIPT_DIR: Path = Path(__file__).resolve().parent

def find_csv(pattern: str) -> Path:
    raw_dir = SCRIPT_DIR / "raw"
    matches: list[Path] = [
        f for f in raw_dir.glob("*.csv")
        if pattern.lower() in f.name.lower()
    ]

    if not matches:
        raise FileNotFoundError(
            f"No CSV containing '{pattern}' found in {raw_dir}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple CSVs containing '{pattern}' found: {matches}"
        )

    return matches[0]

# ---- helper to add future diagnosis columns ----
def add_future_diag(df, months):
    right = df[["RID", "VISIT", "DIAGNOSIS"]].rename(columns={"DIAGNOSIS": f"DIAGNOSIS_{months}"})
    out = df.merge(
        right,
        left_on=["RID", df["VISIT"] + months],
        right_on=["RID", "VISIT"],
        how="left",
        suffixes=("", f"_{months}_dup"),
    )
    return out

import pandas as pd


def transform_diagnosis(df_diag: pd.DataFrame) -> pd.DataFrame:
    """Transform diagnosis data and derive future diagnosis/change information."""

    # Ensure VISIT is numeric and valid
    df_diag = df_diag.copy()
    df_diag["VISIT"] = pd.to_numeric(df_diag["VISIT"], errors="coerce")
    df_diag = df_diag.dropna(subset=["VISIT"]).copy()
    df_diag["VISIT"] = df_diag["VISIT"].astype(int)

    # Keep only needed base columns
    base_cols: list[str] = [
        "PTID",
        "RID",
        "VISCODE",
        "VISCODE2",
        "VISIT",
        "EXAMDATE",
        "DIAGNOSIS",
    ]
    df_diag = df_diag[base_cols].copy()

    # Add future diagnoses
    df: pd.DataFrame = df_diag.copy()

    for month in (6, 12, 18, 24):
        df = add_future_diag(df, month)

    # Identify diagnosis changes at each horizon
    change_sets: dict[int, pd.DataFrame] = {}

    for month in (6, 12, 18):
        diagnosis_col = f"DIAGNOSIS_{month}"

        change_sets[month] = df.loc[
            df[diagnosis_col].notna()
            & (df["DIAGNOSIS"] != df[diagnosis_col]),
            ["RID", "VISIT", diagnosis_col],
        ].copy()

    # Fill missing DIAGNOSIS_24 using 18 -> 12 -> 6,
    # but only when a diagnosis change occurred
    for month in (18, 12, 6):
        fill_col = f"FILL_{month}"
        diagnosis_col = f"DIAGNOSIS_{month}"

        df = df.merge(
            change_sets[month].rename(
                columns={diagnosis_col: fill_col}
            ),
            on=["RID", "VISIT"],
            how="left",
        )

    # Fill DIAGNOSIS_24 in priority order: 18 -> 12 -> 6
    for month in (18, 12, 6):
        df["DIAGNOSIS_24"] = df["DIAGNOSIS_24"].fillna(
            df[f"FILL_{month}"]
        )

    # Track when diagnosis first changed
    df["DIAGNOSIS_CHANGE_MONTH"] = -1

    df.loc[df["FILL_6"].notna(), "DIAGNOSIS_CHANGE_MONTH"] = 6

    df.loc[
        df["FILL_6"].isna() & df["FILL_12"].notna(),
        "DIAGNOSIS_CHANGE_MONTH",
    ] = 12

    df.loc[
        df["FILL_6"].isna()
        & df["FILL_12"].isna()
        & df["DIAGNOSIS_24"].notna()
        & (df["DIAGNOSIS"] != df["DIAGNOSIS_24"]),
        "DIAGNOSIS_CHANGE_MONTH",
    ] = 24

    # Cleanup
    df = df.drop(
        columns=["FILL_18", "FILL_12", "FILL_6"]
    )

    # Final column order
    dia_cols: list[str] = base_cols + [
        "DIAGNOSIS_6",
        "DIAGNOSIS_12",
        "DIAGNOSIS_18",
        "DIAGNOSIS_24",
        "DIAGNOSIS_CHANGE_MONTH",
    ]

    df = df[dia_cols]

    # Remove rows where any future diagnosis is 1
    cols_to_check: list[str] = [
        "DIAGNOSIS_6",
        "DIAGNOSIS_12",
        "DIAGNOSIS_18",
        "DIAGNOSIS_24",
    ]

    df = df[~(df[cols_to_check] == 1).any(axis=1)].copy()

    # Require DIAGNOSIS_24 to exist
    df_dia: pd.DataFrame = df.dropna(
        subset=["DIAGNOSIS_24"]
    ).copy()

    return df_dia


# Load
def create_dataset():

    df_dia = pd.read_csv(find_csv("DXSUM"))
    df_mmse = pd.read_csv(find_csv("MMSE"))
    df_cdr = pd.read_csv(find_csv("CDR"))
    df_adas = pd.read_csv(find_csv("ADAS"))
    df_faq = pd.read_csv(find_csv("FAQ"))
    df_apoe4 = pd.read_csv(find_csv("APOERES"))
    df_dem = pd.read_csv(find_csv("PTDEMOG"))
    df_fs = pd.read_csv(find_csv("UCSFFS"))


    # transform diagnosis data and derive future diagnosis/change information
    df_dia = transform_diagnosis(df_dia)

    # --------------------------------------------------------------
    # Merge MMSE
    # --------------------------------------------------------------

    df_temp: pd.DataFrame = df_dia.merge(
        df_mmse[
            ["RID", "VISCODE2", "MMSCORE", "VISDATE"]
        ],
        on=["RID", "VISCODE2"],
        how="left",
    )

    df_temp["EXAMDATE"] = df_temp["EXAMDATE"].fillna(
        df_temp["VISDATE"]
    )

    # --------------------------------------------------------------
    # Merge CDR
    # --------------------------------------------------------------

    df_temp = df_temp.merge(
        df_cdr[
            ["RID", "VISCODE2", "CDGLOBAL", "CDRSB"]
        ],
        on=["RID", "VISCODE2"],
        how="left",
    )

    # --------------------------------------------------------------
    # Merge ADAS
    # --------------------------------------------------------------

    df_temp = df_temp.merge(
        df_adas[
            ["RID", "VISCODE2", "TOTSCORE", "TOTAL13"]
        ],
        on=["RID", "VISCODE2"],
        how="left",
    )

    # --------------------------------------------------------------
    # Merge FAQ
    # --------------------------------------------------------------

    df_temp = df_temp.merge(
        df_faq[
            ["RID", "VISCODE2", "FAQTOTAL"]
        ],
        on=["RID", "VISCODE2"],
        how="left",
    )

    # --------------------------------------------------------------
    # Merge APOE4
    # --------------------------------------------------------------

    df_temp = df_temp.merge(
        df_apoe4[["RID", "APOE4"]],
        on="RID",
        how="left",
    )

    # --------------------------------------------------------------
    # Merge demographics
    # --------------------------------------------------------------

    df_temp = df_temp.merge(
        df_dem[
            [
                "RID",
                "PTGENDER",
                "PTDOB",
                "PTHAND",
                "PTEDUCAT",
            ]
        ].drop_duplicates(subset=["RID"]),
        on="RID",
        how="left",
    )

    # --------------------------------------------------------------
    # Calculate age
    # --------------------------------------------------------------

    df_temp["EXAMDATE"] = pd.to_datetime(
        df_temp["EXAMDATE"],
        errors="coerce",
    )

    df_temp["PTDOB"] = pd.to_datetime(
        df_temp["PTDOB"],
        errors="coerce",
    )

    df_temp["AGE"] = (
        (
            df_temp["EXAMDATE"]
            - df_temp["PTDOB"]
        )
        .dt.days
        .div(365.25)
        .round(2)
    )

    # Convert categorical variables to zero-based
    df_temp["PTHAND"] = df_temp["PTHAND"] - 1
    df_temp["PTGENDER"] = df_temp["PTGENDER"] - 1

    # --------------------------------------------------------------
    # Merge UCSF FreeSurfer data
    # --------------------------------------------------------------

    df: pd.DataFrame = pd.merge(
        df_temp,
        df_fs,
        on=["RID", "VISIT"],
    )

    fs_col: list[str] = [
        col
        for col in df_fs.columns
        if col.startswith("ST")
    ]

    #drop duplicates on RID, VISIT, keeping the one with least missing values
    df["_miss_count"] = df[fs_col].isna().sum(axis=1)
    df =df.sort_values(by=["RID", "VISIT", "_miss_count"], ascending=[True, True, True])
    df = df.drop_duplicates(subset=["RID", "VISIT"], keep="first").drop(columns=["_miss_count"])

    # --------------------------------------------------------------
    # Select final columns
    # --------------------------------------------------------------

    selected_columns: list[str] = (
        cog
        + dem
        + fs_col
        + label
        + ["RID", "VISIT"]
    )

    df = df[selected_columns].dropna()

    # --------------------------------------------------------------
    # Keep baseline MCI
    # --------------------------------------------------------------

    df = df[
        df["DIAGNOSIS"] == 2
    ].copy()

    # --------------------------------------------------------------
    # Remove invalid feature values
    # --------------------------------------------------------------

    feature_cols: list[str] = (
        cog + dem + fs_col
    )

    df = df[
        (df[feature_cols] >= 0).all(axis=1)
    ].copy()

    # --------------------------------------------------------------
    # Convert future diagnosis:
    # diagnosis 3 -> 1, otherwise -> 0
    # --------------------------------------------------------------

    df["DIAGNOSIS_24"] = (
        df["DIAGNOSIS_24"]
        .eq(3)
        .astype(int)
    )

    # --------------------------------------------------------------
    # Remove inconsistent diagnosis changes
    # --------------------------------------------------------------

    df = df[
        ~(
            (df["DIAGNOSIS_24"] == 0)
            & (df["DIAGNOSIS_CHANGE_MONTH"] != -1)
        )
    ].copy()

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    df.to_csv(
        SCRIPT_DIR / "final_dataset.csv",
        index=False,
    )

    print(df["DIAGNOSIS_24"].value_counts())

    return df


if __name__ == "__main__":
    create_dataset()