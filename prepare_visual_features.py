from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. Paths
# ============================================================

DAIC_FOLDER = Path(
    r"D:\Downloads\Depression_Anxiety_Body_Movement\data\raw\daicwoz"
)

LABELS_PATH = (
    DAIC_FOLDER
    / "complete_Depression_AVEC2017.csv"
)

OUTPUT_FOLDER = Path(
    r"D:\Downloads\Depression_Anxiety_Body_Movement\data\processed"
    r"\daicwoz_visual"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. Selected visual features
# ============================================================

# Action Unit intensity features
AU_FEATURES = [
    "AU01_r",
    "AU02_r",
    "AU04_r",
    "AU05_r",
    "AU06_r",
    "AU09_r",
    "AU10_r",
    "AU12_r",
    "AU14_r",
    "AU15_r",
    "AU17_r",
    "AU20_r",
    "AU25_r",
    "AU26_r"
]

# Gaze vectors for both eyes
GAZE_FEATURES = [
    "x_0",
    "y_0",
    "z_0",
    "x_1",
    "y_1",
    "z_1",
    "x_h0",
    "y_h0",
    "z_h0",
    "x_h1",
    "y_h1",
    "z_h1"
]

# Head translation and rotation
POSE_FEATURES = [
    "Tx",
    "Ty",
    "Tz",
    "Rx",
    "Ry",
    "Rz"
]

SELECTED_FEATURES = (
    AU_FEATURES
    + GAZE_FEATURES
    + POSE_FEATURES
)

EXPECTED_FEATURE_COUNT = len(
    SELECTED_FEATURES
)


# ============================================================
# 3. Helper functions
# ============================================================

def read_feature_file(
    file_path: Path
) -> pd.DataFrame:
    """
    Read a comma-separated DAIC-WOZ visual feature file.
    """

    dataframe = pd.read_csv(
        file_path,
        sep=",",
        skipinitialspace=True,
        low_memory=False
    )

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    return dataframe


def convert_selected_columns_to_numeric(
    dataframe: pd.DataFrame,
    columns: list[str]
) -> pd.DataFrame:
    """
    Select requested columns and convert them to numeric.
    """

    result = dataframe[
        ["frame", "success"] + columns
    ].copy()

    for column in result.columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce"
        )

    return result


def clean_missing_values(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Replace infinite values and interpolate small missing gaps.
    """

    dataframe = dataframe.replace(
        [np.inf, -np.inf],
        np.nan
    )

    dataframe = dataframe.interpolate(
        method="linear",
        limit_direction="both"
    )

    dataframe = dataframe.ffill().bfill()

    return dataframe


def check_required_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    file_name: str
):
    """
    Raise an error when required columns are missing.
    """

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{file_name} is missing columns: "
            f"{missing_columns}"
        )


# ============================================================
# 4. Read labels
# ============================================================

if not LABELS_PATH.exists():
    print("ERROR: Labels file was not found:")
    print(LABELS_PATH)
    raise SystemExit


labels_df = pd.read_csv(
    LABELS_PATH
)

# Keep only participants with available labels.
labels_df = labels_df[
    labels_df["PHQ8_Binary"].isin([0, 1])
].copy()

labels_df["Participant_ID"] = (
    labels_df["Participant_ID"]
    .astype(int)
)

print("Labeled participants:")
print(len(labels_df))

print("\nExpected number of visual features:")
print(EXPECTED_FEATURE_COUNT)


# ============================================================
# 5. Process participants
# ============================================================

metadata_records = []
failed_participants = []

for number, participant_row in enumerate(
    labels_df.itertuples(index=False),
    start=1
):

    participant_id = int(
        participant_row.Participant_ID
    )

    participant_label = int(
        participant_row.PHQ8_Binary
    )

    participant_folder = (
        DAIC_FOLDER
        / f"{participant_id}_P"
    )

    au_path = (
        participant_folder
        / f"{participant_id}_CLNF_AUs.txt"
    )

    gaze_path = (
        participant_folder
        / f"{participant_id}_CLNF_gaze.txt"
    )

    pose_path = (
        participant_folder
        / f"{participant_id}_CLNF_pose.txt"
    )

    required_files = [
        au_path,
        gaze_path,
        pose_path
    ]

    if not all(
        file_path.exists()
        for file_path in required_files
    ):
        print(
            f"Skipping participant {participant_id}: "
            "one or more feature files are missing."
        )

        failed_participants.append(
            participant_id
        )

        continue

    try:
        # ----------------------------------------------------
        # Read files
        # ----------------------------------------------------

        au_df = read_feature_file(
            au_path
        )

        gaze_df = read_feature_file(
            gaze_path
        )

        pose_df = read_feature_file(
            pose_path
        )

        # ----------------------------------------------------
        # Check required columns
        # ----------------------------------------------------

        check_required_columns(
            au_df,
            ["frame", "success"] + AU_FEATURES,
            "Action Units file"
        )

        check_required_columns(
            gaze_df,
            ["frame", "success"] + GAZE_FEATURES,
            "Gaze file"
        )

        check_required_columns(
            pose_df,
            ["frame", "success"] + POSE_FEATURES,
            "Head Pose file"
        )

        # ----------------------------------------------------
        # Select and convert columns
        # ----------------------------------------------------

        au_selected = convert_selected_columns_to_numeric(
            au_df,
            AU_FEATURES
        )

        gaze_selected = convert_selected_columns_to_numeric(
            gaze_df,
            GAZE_FEATURES
        )

        pose_selected = convert_selected_columns_to_numeric(
            pose_df,
            POSE_FEATURES
        )

        # Rename success columns before merging
        au_selected = au_selected.rename(
            columns={
                "success": "AU_success"
            }
        )

        gaze_selected = gaze_selected.rename(
            columns={
                "success": "Gaze_success"
            }
        )

        pose_selected = pose_selected.rename(
            columns={
                "success": "Pose_success"
            }
        )

        # ----------------------------------------------------
        # Align modalities using frame number
        # ----------------------------------------------------

        combined_df = au_selected.merge(
            gaze_selected,
            on="frame",
            how="inner"
        )

        combined_df = combined_df.merge(
            pose_selected,
            on="frame",
            how="inner"
        )

        original_aligned_frames = len(
            combined_df
        )

        # ----------------------------------------------------
        # Keep frames where all selected modalities succeeded
        # ----------------------------------------------------

        combined_df = combined_df[
            (combined_df["AU_success"] == 1)
            & (combined_df["Gaze_success"] == 1)
            & (combined_df["Pose_success"] == 1)
        ].copy()

        successful_frames = len(
            combined_df
        )

        if successful_frames == 0:
            print(
                f"Skipping participant {participant_id}: "
                "no jointly successful frames."
            )

            failed_participants.append(
                participant_id
            )

            continue

        joint_tracking_percentage = (
            successful_frames
            / original_aligned_frames
            * 100
        )

        # Remove metadata/tracking columns
        combined_df = combined_df.drop(
            columns=[
                "frame",
                "AU_success",
                "Gaze_success",
                "Pose_success"
            ]
        )

        # ----------------------------------------------------
        # Clean missing values
        # ----------------------------------------------------

        combined_df = clean_missing_values(
            combined_df
        )

        combined_df = combined_df[
            SELECTED_FEATURES
        ]

        if combined_df.isna().any().any():
            print(
                f"Skipping participant {participant_id}: "
                "unresolved missing values remain."
            )

            failed_participants.append(
                participant_id
            )

            continue

        feature_array = combined_df.to_numpy(
            dtype=np.float32
        )

        if feature_array.shape[1] != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_FEATURE_COUNT} features, "
                f"but found {feature_array.shape[1]}."
            )

        # ----------------------------------------------------
        # Save participant features
        # ----------------------------------------------------

        output_path = (
            OUTPUT_FOLDER
            / f"{participant_id}_visual.npy"
        )

        np.save(
            output_path,
            feature_array
        )

        metadata_records.append(
            {
                "Participant_ID": participant_id,
                "Label": participant_label,
                "PHQ8_Score": int(
                    participant_row.PHQ8_Score
                ),
                "Original_Aligned_Frames":
                    original_aligned_frames,
                "Successful_Frames":
                    successful_frames,
                "Joint_Tracking_Percentage":
                    joint_tracking_percentage,
                "Number_of_Features":
                    feature_array.shape[1],
                "Feature_File":
                    str(output_path)
            }
        )

        if number % 20 == 0:
            print(
                f"Processed {number} of "
                f"{len(labels_df)} participants..."
            )

    except Exception as error:
        print(
            f"Error processing participant "
            f"{participant_id}: {error}"
        )

        failed_participants.append(
            participant_id
        )


# ============================================================
# 6. Save metadata
# ============================================================

metadata_df = pd.DataFrame(
    metadata_records
)

metadata_path = (
    OUTPUT_FOLDER
    / "visual_feature_metadata.csv"
)

metadata_df.to_csv(
    metadata_path,
    index=False
)


# Save feature names as well
feature_names_path = (
    OUTPUT_FOLDER
    / "visual_feature_names.csv"
)

pd.DataFrame(
    {
        "Feature_Index": range(
            EXPECTED_FEATURE_COUNT
        ),
        "Feature_Name": SELECTED_FEATURES
    }
).to_csv(
    feature_names_path,
    index=False
)


# ============================================================
# 7. Final summary
# ============================================================

print("\n" + "=" * 70)
print("VISUAL FEATURE EXTRACTION SUMMARY")
print("=" * 70)

print(
    "Successfully processed participants:",
    len(metadata_df)
)

print(
    "Failed or skipped participants:",
    len(set(failed_participants))
)

if failed_participants:
    print(
        "Participant IDs:",
        sorted(set(failed_participants))
    )

if not metadata_df.empty:
    print("\nSuccessful frames per participant:")

    print(
        metadata_df[
            "Successful_Frames"
        ].describe()
    )

    print("\nJoint tracking percentage:")

    print(
        metadata_df[
            "Joint_Tracking_Percentage"
        ].describe()
    )

    print("\nFeature dimensions:")

    print(
        metadata_df[
            "Number_of_Features"
        ].value_counts()
    )

print("\nOutputs saved in:")
print(OUTPUT_FOLDER)

print("\nMetadata:")
print(metadata_path)

print("\nFeature names:")
print(feature_names_path)