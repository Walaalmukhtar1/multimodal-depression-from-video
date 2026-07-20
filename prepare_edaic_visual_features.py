from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd


# ============================================================
# 1. Paths
# ============================================================

EDAIC_FOLDER = Path(
    r"D:\Downloads\Depression_Anxiety_Body_Movement\data\raw\e-daic-woz"
)

TRAIN_LABELS_PATH = EDAIC_FOLDER / "train_split.csv"
DEV_LABELS_PATH = EDAIC_FOLDER / "dev_split.csv"
TEST_LABELS_PATH = EDAIC_FOLDER / "test_split.csv"

OUTPUT_FOLDER = Path(
    r"D:\Downloads\Depression_Anxiety_Body_Movement\data\processed"
    r"\e_daic_woz_visual"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

METADATA_PATH = (
    OUTPUT_FOLDER
    / "edaic_visual_feature_metadata.csv"
)

FAILED_PATH = (
    OUTPUT_FOLDER
    / "edaic_failed_participants.csv"
)

FEATURE_NAMES_PATH = (
    OUTPUT_FOLDER
    / "edaic_visual_feature_names.csv"
)


# ============================================================
# 2. Selected features
# ============================================================

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
    "AU26_r",
]

GAZE_FEATURES = [
    "gaze_0_x",
    "gaze_0_y",
    "gaze_0_z",
    "gaze_1_x",
    "gaze_1_y",
    "gaze_1_z",
    "gaze_angle_x",
    "gaze_angle_y",
]

POSE_FEATURES = [
    "pose_Tx",
    "pose_Ty",
    "pose_Tz",
    "pose_Rx",
    "pose_Ry",
    "pose_Rz",
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
# 3. Find Windows tar command
# ============================================================

TAR_COMMAND = shutil.which("tar")

if TAR_COMMAND is None:
    raise RuntimeError(
        "Windows tar command was not found. "
        "Run 'where.exe tar' in PowerShell to check it."
    )

print("Using tar command:")
print(TAR_COMMAND)


# ============================================================
# 4. Read split files
# ============================================================

def read_split_file(
    file_path: Path,
    split_name: str,
) -> pd.DataFrame:
    """
    Read one label split and add its split name.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Labels file not found: {file_path}"
        )

    dataframe = pd.read_csv(
        file_path
    )

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    required_columns = [
        "Participant_ID",
        "Gender",
        "PHQ_Binary",
        "PHQ_Score",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{file_path.name} is missing columns: "
            f"{missing_columns}"
        )

    dataframe["Participant_ID"] = pd.to_numeric(
        dataframe["Participant_ID"],
        errors="raise",
    ).astype(int)

    dataframe["PHQ_Binary"] = pd.to_numeric(
        dataframe["PHQ_Binary"],
        errors="raise",
    ).astype(int)

    dataframe["PHQ_Score"] = pd.to_numeric(
        dataframe["PHQ_Score"],
        errors="raise",
    ).astype(int)

    dataframe["Split"] = split_name

    return dataframe


train_df = read_split_file(
    TRAIN_LABELS_PATH,
    "train",
)

dev_df = read_split_file(
    DEV_LABELS_PATH,
    "dev",
)

test_df = read_split_file(
    TEST_LABELS_PATH,
    "test",
)

labels_df = pd.concat(
    [
        train_df,
        dev_df,
        test_df,
    ],
    ignore_index=True,
)

labels_df = labels_df.sort_values(
    [
        "Split",
        "Participant_ID",
    ]
).reset_index(
    drop=True
)


# ============================================================
# 5. Archive helper functions
# ============================================================

def get_archive_path(
    participant_id: int,
) -> Path:
    """
    Return the path of one participant archive.
    """

    return (
        EDAIC_FOLDER
        / f"{participant_id}_P.tar.gz"
    )


def get_openface_member_name(
    participant_id: int,
) -> str:
    """
    Return the OpenFace CSV path inside the tar.gz archive.
    """

    return (
        f"{participant_id}_P/"
        f"features/"
        f"{participant_id}_OpenFace2.1.0_Pose_gaze_AUs.csv"
    )


def read_openface_using_tar(
    archive_path: Path,
    participant_id: int,
) -> pd.DataFrame:
    """
    Use Windows tar.exe to extract only the required OpenFace
    CSV into a temporary file, then load it with pandas.

    This avoids Python tarfile scanning the whole archive.
    """

    member_name = get_openface_member_name(
        participant_id
    )

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".csv",
            delete=False,
        ) as temporary_file:

            temporary_path = Path(
                temporary_file.name
            )

            process = subprocess.run(
                [
                    TAR_COMMAND,
                    "-xOf",
                    str(archive_path),
                    member_name,
                ],
                stdout=temporary_file,
                stderr=subprocess.PIPE,
                text=False,
                timeout=300,
                check=False,
            )

        if process.returncode != 0:
            error_message = process.stderr.decode(
                "utf-8",
                errors="replace",
            )

            raise RuntimeError(
                f"tar failed for participant "
                f"{participant_id}: {error_message.strip()}"
            )

        if temporary_path.stat().st_size == 0:
            raise RuntimeError(
                f"Extracted OpenFace CSV is empty for "
                f"participant {participant_id}."
            )

        dataframe = pd.read_csv(
            temporary_path,
            low_memory=False,
        )

        dataframe.columns = [
            str(column).strip()
            for column in dataframe.columns
        ]

        return dataframe

    except subprocess.TimeoutExpired as error:
        raise TimeoutError(
            f"Reading participant {participant_id} "
            "took longer than 5 minutes."
        ) from error

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            try:
                temporary_path.unlink()
            except PermissionError:
                pass


# ============================================================
# 6. Feature preparation functions
# ============================================================

def check_required_columns(
    dataframe: pd.DataFrame,
    participant_id: int,
) -> None:
    """
    Verify that the OpenFace CSV contains all required columns.
    """

    required_columns = [
        "frame",
        "timestamp",
        "confidence",
        "success",
    ] + SELECTED_FEATURES

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Participant {participant_id} is missing "
            f"columns: {missing_columns}"
        )


def select_and_convert_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select the required columns and convert them to numeric.
    """

    required_columns = [
        "frame",
        "timestamp",
        "confidence",
        "success",
    ] + SELECTED_FEATURES

    result = dataframe[
        required_columns
    ].copy()

    for column in required_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    return result


def clean_feature_values(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Replace invalid values and fill small missing gaps.
    """

    dataframe = dataframe.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    dataframe = dataframe.interpolate(
        method="linear",
        limit_direction="both",
    )

    dataframe = dataframe.ffill()
    dataframe = dataframe.bfill()

    return dataframe


# ============================================================
# 7. Save feature names
# ============================================================

feature_groups = (
    ["Action Unit"] * len(AU_FEATURES)
    + ["Gaze"] * len(GAZE_FEATURES)
    + ["Head Pose"] * len(POSE_FEATURES)
)

pd.DataFrame(
    {
        "Feature_Index": range(
            EXPECTED_FEATURE_COUNT
        ),
        "Feature_Name": SELECTED_FEATURES,
        "Feature_Group": feature_groups,
    }
).to_csv(
    FEATURE_NAMES_PATH,
    index=False,
)


# ============================================================
# 8. Process participants
# ============================================================

metadata_records = []
failed_records = []

print("\n" + "=" * 70)
print("E-DAIC-WOZ VISUAL FEATURE PREPARATION")
print("=" * 70)

print("\nLabeled participants:")
print(len(labels_df))

print("\nParticipants by split:")
print(
    labels_df["Split"]
    .value_counts()
)

print("\nExpected visual features:")
print(EXPECTED_FEATURE_COUNT)

print("\nStarting feature preparation...\n")


for participant_number, participant_row in enumerate(
    labels_df.itertuples(index=False),
    start=1,
):

    participant_id = int(
        participant_row.Participant_ID
    )

    split_name = str(
        participant_row.Split
    )

    archive_path = get_archive_path(
        participant_id
    )

    print(
        f"[{participant_number}/{len(labels_df)}] "
        f"Participant {participant_id} "
        f"({split_name})...",
        flush=True,
    )

    if not archive_path.exists():
        print(
            "  Skipped: archive is missing.",
            flush=True,
        )

        failed_records.append(
            {
                "Participant_ID":
                    participant_id,
                "Split":
                    split_name,
                "Reason":
                    "Archive missing",
            }
        )

        continue

    try:
        # ----------------------------------------------------
        # Read the OpenFace CSV using Windows tar.exe
        # ----------------------------------------------------

        openface_df = read_openface_using_tar(
            archive_path=archive_path,
            participant_id=participant_id,
        )

        original_frames = len(
            openface_df
        )

        if original_frames == 0:
            raise ValueError(
                "OpenFace CSV contains no rows."
            )

        # ----------------------------------------------------
        # Validate and select columns
        # ----------------------------------------------------

        check_required_columns(
            openface_df,
            participant_id,
        )

        selected_df = select_and_convert_columns(
            openface_df
        )

        selected_df = selected_df.dropna(
            subset=[
                "frame",
                "success",
            ]
        )

        # ----------------------------------------------------
        # Keep successful OpenFace frames only
        # ----------------------------------------------------

        successful_df = selected_df[
            selected_df["success"] == 1
        ].copy()

        successful_frames = len(
            successful_df
        )

        if successful_frames == 0:
            raise ValueError(
                "No successful OpenFace frames."
            )

        tracking_percentage = (
            successful_frames
            / original_frames
            * 100
        )

        # ----------------------------------------------------
        # Keep the 28 visual features
        # ----------------------------------------------------

        feature_df = successful_df[
            SELECTED_FEATURES
        ].copy()

        missing_before = int(
            feature_df.isna()
            .sum()
            .sum()
        )

        feature_df = clean_feature_values(
            feature_df
        )

        missing_after = int(
            feature_df.isna()
            .sum()
            .sum()
        )

        if missing_after > 0:
            raise ValueError(
                f"{missing_after} missing values remain "
                "after cleaning."
            )

        feature_array = feature_df.to_numpy(
            dtype=np.float32
        )

        if feature_array.shape[1] != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_FEATURE_COUNT} features, "
                f"found {feature_array.shape[1]}."
            )

        if not np.isfinite(
            feature_array
        ).all():
            raise ValueError(
                "Feature array contains NaN or infinity."
            )

        # ----------------------------------------------------
        # Save NPY file
        # ----------------------------------------------------

        output_path = (
            OUTPUT_FOLDER
            / f"{participant_id}_visual.npy"
        )

        np.save(
            output_path,
            feature_array,
        )

        metadata_records.append(
            {
                "Participant_ID":
                    participant_id,

                "Split":
                    split_name,

                "Label":
                    int(participant_row.PHQ_Binary),

                "PHQ_Score":
                    int(participant_row.PHQ_Score),

                "Gender":
                    str(participant_row.Gender),

                "Original_Frames":
                    original_frames,

                "Successful_Frames":
                    successful_frames,

                "Tracking_Success_Percentage":
                    tracking_percentage,

                "Missing_Values_Before_Cleaning":
                    missing_before,

                "Missing_Values_After_Cleaning":
                    missing_after,

                "Number_of_Features":
                    feature_array.shape[1],

                "Feature_File":
                    str(output_path),

                "Source_Archive":
                    str(archive_path),
            }
        )

        print(
            f"  Saved: {feature_array.shape} "
            f"| tracking {tracking_percentage:.2f}%",
            flush=True,
        )

    except Exception as error:
        print(
            f"  Failed: {error}",
            flush=True,
        )

        failed_records.append(
            {
                "Participant_ID":
                    participant_id,
                "Split":
                    split_name,
                "Reason":
                    str(error),
            }
        )

    # Save checkpoint reports after every participant.
    pd.DataFrame(
        metadata_records
    ).to_csv(
        METADATA_PATH,
        index=False,
    )

    pd.DataFrame(
        failed_records
    ).to_csv(
        FAILED_PATH,
        index=False,
    )


# ============================================================
# 9. Final DataFrames
# ============================================================

metadata_df = pd.DataFrame(
    metadata_records
)

failed_df = pd.DataFrame(
    failed_records
)


# ============================================================
# 10. Print summaries
# ============================================================

def print_split_summary(
    dataframe: pd.DataFrame,
    split_name: str,
) -> None:

    split_df = dataframe[
        dataframe["Split"] == split_name
    ]

    print("\n" + split_name.upper())
    print("-" * 40)

    print(
        "Participants:",
        len(split_df),
    )

    if split_df.empty:
        return

    print("\nClass distribution:")

    print(
        split_df["Label"]
        .value_counts()
        .sort_index()
    )

    print("\nSuccessful frames:")

    print(
        split_df[
            "Successful_Frames"
        ].describe()
    )

    print("\nTracking success:")

    print(
        split_df[
            "Tracking_Success_Percentage"
        ].describe()
    )


print("\n" + "=" * 70)
print("E-DAIC-WOZ FEATURE PREPARATION SUMMARY")
print("=" * 70)

print(
    "\nSuccessfully processed:",
    len(metadata_df),
)

print(
    "Failed or skipped:",
    len(failed_df),
)

if not metadata_df.empty:
    print("\nFeature dimensions:")

    print(
        metadata_df[
            "Number_of_Features"
        ].value_counts()
    )

    print("\nOverall tracking success:")

    print(
        metadata_df[
            "Tracking_Success_Percentage"
        ].describe()
    )

    print_split_summary(
        metadata_df,
        "train",
    )

    print_split_summary(
        metadata_df,
        "dev",
    )

    print_split_summary(
        metadata_df,
        "test",
    )

if not failed_df.empty:
    print("\nFailed participants:")

    print(
        failed_df.to_string(
            index=False
        )
    )

print("\nOutputs saved in:")
print(OUTPUT_FOLDER)

print("\nMetadata:")
print(METADATA_PATH)

print("\nFailed participant report:")
print(FAILED_PATH)

print("\nFeature names:")
print(FEATURE_NAMES_PATH)