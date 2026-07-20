from __future__ import annotations

from pathlib import Path
import io
import json
import tarfile

import numpy as np
import pandas as pd

EDAIC_ARCHIVES_FOLDER = Path(
    r"D:\Downloads\Depression_Anxiety_Body_Movement\data\raw\e-daic-woz"
)
OUTPUT_FOLDER = Path(
    r"D:\Flutterr\multimodal-depression-from-video\dataset_audit_outputs\bovw_corrected"
)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV_PATH = OUTPUT_FOLDER / "edaic_bovw_corrected_summary.csv"
COLUMNS_CSV_PATH = OUTPUT_FOLDER / "edaic_bovw_corrected_columns.csv"
SUMMARY_JSON_PATH = OUTPUT_FOLDER / "edaic_bovw_corrected_summary.json"

MAX_ARCHIVES = 10


def get_participant_id(archive_path: Path) -> int | None:
    digits = "".join(ch for ch in archive_path.name if ch.isdigit())
    return int(digits[:3]) if len(digits) >= 3 else None


def load_bovw_dataframe(archive: tarfile.TarFile, member: tarfile.TarInfo) -> pd.DataFrame:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ValueError(f"Could not extract {member.name}")

    return pd.read_csv(
        io.BytesIO(extracted.read()),
        header=None,
        low_memory=False,
    )


def inspect_dataframe(dataframe: pd.DataFrame) -> dict:
    rows, columns = dataframe.shape
    first_column = dataframe.iloc[:, 0]

    numeric_dataframe = dataframe.apply(pd.to_numeric, errors="coerce")
    numeric_counts = numeric_dataframe.notna().sum(axis=0)

    fully_numeric = [
        int(index)
        for index, count in numeric_counts.items()
        if count == rows
    ]
    partially_numeric = [
        int(index)
        for index, count in numeric_counts.items()
        if 0 < count < rows
    ]
    non_numeric = [
        int(index)
        for index, count in numeric_counts.items()
        if count == 0
    ]

    numeric_feature_columns = [index for index in fully_numeric if index != 0]

    numeric_values = (
        numeric_dataframe[numeric_feature_columns].to_numpy(dtype=np.float32)
        if numeric_feature_columns
        else np.empty((rows, 0), dtype=np.float32)
    )

    return {
        "Rows": int(rows),
        "Columns": int(columns),
        "First_Column_Unique_Values": int(first_column.nunique(dropna=False)),
        "First_Column_Example": str(first_column.iloc[0]) if rows else "",
        "Fully_Numeric_Columns": len(fully_numeric),
        "Partially_Numeric_Columns": len(partially_numeric),
        "Non_Numeric_Columns": len(non_numeric),
        "Numeric_Feature_Columns_Excluding_First": len(numeric_feature_columns),
        "Finite_Value_Ratio": float(np.isfinite(numeric_values).mean()) if numeric_values.size else 0.0,
        "Zero_Value_Ratio": float((numeric_values == 0).mean()) if numeric_values.size else 0.0,
        "Non_Numeric_Column_Indices": non_numeric,
    }


if not EDAIC_ARCHIVES_FOLDER.exists():
    raise FileNotFoundError(
        "E-DAIC archive folder was not found:\n"
        f"{EDAIC_ARCHIVES_FOLDER}"
    )

archive_paths = sorted(
    list(EDAIC_ARCHIVES_FOLDER.glob("*.tar.gz"))
    + list(EDAIC_ARCHIVES_FOLDER.glob("*.tgz"))
)[:MAX_ARCHIVES]

if not archive_paths:
    raise FileNotFoundError(
        "No archives were found in:\n"
        f"{EDAIC_ARCHIVES_FOLDER}"
    )

print("=" * 76)
print("E-DAIC-WOZ CORRECTED BoVW INSPECTION")
print("=" * 76)
print(f"\nArchives selected: {len(archive_paths)}")

summary_records = []
column_records = []
example_preview = None

for archive_number, archive_path in enumerate(archive_paths, start=1):
    participant_id = get_participant_id(archive_path)
    print(f"\nInspecting {archive_number}/{len(archive_paths)}: {archive_path.name}")

    with tarfile.open(archive_path, mode="r:gz") as archive:
        bovw_members = [
            member
            for member in archive.getmembers()
            if member.isfile()
            and "bovw" in member.name.lower()
            and member.name.lower().endswith(".csv")
        ]

        if not bovw_members:
            print("  No BoVW CSV found.")
            summary_records.append({
                "Participant_ID": participant_id,
                "Archive": archive_path.name,
                "Member_Name": "",
                "BoVW_File_Found": 0,
            })
            continue

        for member in bovw_members:
            dataframe = load_bovw_dataframe(archive, member)
            inspection = inspect_dataframe(dataframe)

            summary_records.append({
                "Participant_ID": participant_id,
                "Archive": archive_path.name,
                "Member_Name": member.name,
                "BoVW_File_Found": 1,
                **{key: value for key, value in inspection.items() if not isinstance(value, list)},
            })

            for column_index in range(dataframe.shape[1]):
                series = dataframe.iloc[:, column_index]
                numeric_series = pd.to_numeric(series, errors="coerce")
                column_records.append({
                    "Participant_ID": participant_id,
                    "Member_Name": member.name,
                    "Column_Index": int(column_index),
                    "Non_Null_Count": int(series.notna().sum()),
                    "Numeric_Count": int(numeric_series.notna().sum()),
                    "Unique_Values": int(series.nunique(dropna=False)),
                    "First_Value": str(series.iloc[0]) if len(series) else "",
                })

            if example_preview is None:
                example_preview = {
                    "Participant_ID": participant_id,
                    "Member_Name": member.name,
                    "Shape": [int(dataframe.shape[0]), int(dataframe.shape[1])],
                    "First_Three_Rows": dataframe.head(3).astype(str).values.tolist(),
                }

            print(f"  Found: {member.name}")
            print(f"  True shape: ({inspection['Rows']}, {inspection['Columns']})")
            print(f"  First column unique values: {inspection['First_Column_Unique_Values']}")
            print(f"  First column example: {inspection['First_Column_Example']}")
            print(f"  Fully numeric columns: {inspection['Fully_Numeric_Columns']}")
            print(
                "  Numeric feature columns excluding first: "
                f"{inspection['Numeric_Feature_Columns_Excluding_First']}"
            )
            print(f"  Non-numeric column indices: {inspection['Non_Numeric_Column_Indices']}")
            print(f"  Zero-value ratio: {inspection['Zero_Value_Ratio']:.4f}")

summary_df = pd.DataFrame(summary_records)
columns_df = pd.DataFrame(column_records)

summary_df.to_csv(SUMMARY_CSV_PATH, index=False)
columns_df.to_csv(COLUMNS_CSV_PATH, index=False)

valid_summary_df = summary_df[summary_df["BoVW_File_Found"] == 1].copy()

summary_json = {
    "archives_inspected": len(archive_paths),
    "bovw_files_found": int(valid_summary_df.shape[0]),
    "unique_shapes": (
        valid_summary_df[["Rows", "Columns"]]
        .drop_duplicates()
        .to_dict(orient="records")
        if not valid_summary_df.empty
        else []
    ),
    "unique_numeric_feature_counts": (
        sorted(
            valid_summary_df["Numeric_Feature_Columns_Excluding_First"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        if not valid_summary_df.empty
        else []
    ),
    "example_preview": example_preview,
}

with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as output_file:
    json.dump(summary_json, output_file, indent=4, ensure_ascii=False)

print("\n" + "=" * 76)
print("SUMMARY")
print("=" * 76)
print(f"\nBoVW files found: {len(valid_summary_df)}")

if not valid_summary_df.empty:
    print("\nUnique true shapes:")
    print(
        valid_summary_df[["Rows", "Columns"]]
        .drop_duplicates()
        .to_string(index=False)
    )

    print("\nUnique numeric feature counts excluding first column:")
    print(
        sorted(
            valid_summary_df["Numeric_Feature_Columns_Excluding_First"]
            .astype(int)
            .unique()
            .tolist()
        )
    )

print("\nOutputs saved in:")
print(OUTPUT_FOLDER)
print("\nMost important output:")
print(SUMMARY_CSV_PATH)