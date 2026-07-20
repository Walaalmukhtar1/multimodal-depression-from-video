from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. Configuration
# ============================================================

FPS = 30
SECONDS_PER_WINDOW = 9

WINDOW_SIZE = FPS * SECONDS_PER_WINDOW

# Non-overlapping windows:
# Window 1: 0–269
# Window 2: 270–539
STRIDE = WINDOW_SIZE


# ============================================================
# 2. Paths
# ============================================================

DAIC_FOLDER = Path(
    r"D:\Downloads\Depression_Anxiety_Body_Movement\data\raw\daicwoz"
)

PROCESSED_FEATURE_FOLDER = Path(
    r"D:\Downloads\Depression_Anxiety_Body_Movement\data\processed"
    r"\daicwoz_visual"
)

FEATURE_METADATA_PATH = (
    PROCESSED_FEATURE_FOLDER
    / "visual_feature_metadata.csv"
)

TRAIN_SPLIT_PATH = (
    DAIC_FOLDER
    / "train_split_Depression_AVEC2017.csv"
)

VALIDATION_SPLIT_PATH = (
    DAIC_FOLDER
    / "dev_split_Depression_AVEC2017.csv"
)

OUTPUT_FOLDER = Path(
    r"D:\Downloads\Depression_Anxiety_Body_Movement\data\processed"
    r"\daicwoz_visual_windows"
)

TRAIN_OUTPUT_FOLDER = (
    OUTPUT_FOLDER
    / "train"
)

VALIDATION_OUTPUT_FOLDER = (
    OUTPUT_FOLDER
    / "validation"
)

TRAIN_OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

VALIDATION_OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. Check required files
# ============================================================

required_paths = {
    "Feature metadata": FEATURE_METADATA_PATH,
    "Training split": TRAIN_SPLIT_PATH,
    "Validation split": VALIDATION_SPLIT_PATH
}

for name, path in required_paths.items():

    print(f"{name}:")
    print(path)

    if not path.exists():
        print(f"\nERROR: {name} was not found.")
        raise SystemExit


# ============================================================
# 4. Read metadata and split files
# ============================================================

metadata_df = pd.read_csv(
    FEATURE_METADATA_PATH
)

train_split_df = pd.read_csv(
    TRAIN_SPLIT_PATH
)

validation_split_df = pd.read_csv(
    VALIDATION_SPLIT_PATH
)

metadata_df["Participant_ID"] = (
    metadata_df["Participant_ID"]
    .astype(int)
)

train_split_df["Participant_ID"] = (
    train_split_df["Participant_ID"]
    .astype(int)
)

validation_split_df["Participant_ID"] = (
    validation_split_df["Participant_ID"]
    .astype(int)
)


train_participant_ids = set(
    train_split_df["Participant_ID"]
)

validation_participant_ids = set(
    validation_split_df["Participant_ID"]
)


train_metadata_df = metadata_df[
    metadata_df["Participant_ID"].isin(
        train_participant_ids
    )
].copy()

validation_metadata_df = metadata_df[
    metadata_df["Participant_ID"].isin(
        validation_participant_ids
    )
].copy()


print("\nParticipants available after feature preparation:")

print(
    "Train:",
    len(train_metadata_df)
)

print(
    "Validation:",
    len(validation_metadata_df)
)


# ============================================================
# 5. Verify participant separation
# ============================================================

overlapping_participants = (
    set(train_metadata_df["Participant_ID"])
    &
    set(validation_metadata_df["Participant_ID"])
)

if overlapping_participants:

    print("\nERROR: Participants appear in both splits:")
    print(sorted(overlapping_participants))
    raise SystemExit

print("\nNo participant overlap between Train and Validation.")


# ============================================================
# 6. Function for loading a feature array
# ============================================================

def load_feature_array(
    feature_path: str
) -> np.ndarray:
    """
    Load one participant's prepared visual features.
    """

    path = Path(feature_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Feature file was not found: {path}"
        )

    feature_array = np.load(
        path
    )

    if feature_array.ndim != 2:
        raise ValueError(
            f"Expected a 2D array, but found shape "
            f"{feature_array.shape} in {path}"
        )

    if feature_array.shape[1] != 32:
        raise ValueError(
            f"Expected 32 visual features, but found "
            f"{feature_array.shape[1]} in {path}"
        )

    return feature_array.astype(
        np.float64,
        copy=False
    )


# ============================================================
# 7. Calculate normalization statistics from Train only
# ============================================================

print("\nCalculating normalization statistics from Train only...")

feature_sum = np.zeros(
    32,
    dtype=np.float64
)

feature_squared_sum = np.zeros(
    32,
    dtype=np.float64
)

total_train_frames = 0


for number, participant_row in enumerate(
    train_metadata_df.itertuples(index=False),
    start=1
):

    feature_array = load_feature_array(
        participant_row.Feature_File
    )

    feature_sum += feature_array.sum(
        axis=0
    )

    feature_squared_sum += (
        feature_array ** 2
    ).sum(
        axis=0
    )

    total_train_frames += (
        feature_array.shape[0]
    )

    if number % 20 == 0:

        print(
            f"Read {number} of "
            f"{len(train_metadata_df)} "
            "training participants..."
        )


if total_train_frames == 0:

    print(
        "\nERROR: No training frames were found."
    )

    raise SystemExit


train_mean = (
    feature_sum
    / total_train_frames
)

train_variance = (
    feature_squared_sum
    / total_train_frames
) - (
    train_mean ** 2
)

# Numerical errors can occasionally make a tiny variance negative.
train_variance = np.maximum(
    train_variance,
    0
)

train_std = np.sqrt(
    train_variance
)

# Avoid division by zero for constant features.
train_std[train_std < 1e-8] = 1.0


print("\nTotal Train frames used for normalization:")
print(total_train_frames)

print("\nTrain feature mean shape:")
print(train_mean.shape)

print("\nTrain feature standard deviation shape:")
print(train_std.shape)


# ============================================================
# 8. Save normalization parameters
# ============================================================

normalization_path = (
    OUTPUT_FOLDER
    / "train_normalization_parameters.npz"
)

np.savez(
    normalization_path,
    mean=train_mean.astype(np.float32),
    std=train_std.astype(np.float32),
    total_train_frames=np.array(
        [total_train_frames],
        dtype=np.int64
    )
)

print("\nNormalization parameters saved as:")
print(normalization_path)


# ============================================================
# 9. Windowing function
# ============================================================

def create_windows(
    normalized_array: np.ndarray,
    window_size: int,
    stride: int
) -> np.ndarray:
    """
    Divide a sequence into fixed-length windows.

    Incomplete frames at the end are dropped.
    """

    number_of_frames = (
        normalized_array.shape[0]
    )

    number_of_features = (
        normalized_array.shape[1]
    )

    if number_of_frames < window_size:

        return np.empty(
            (
                0,
                window_size,
                number_of_features
            ),
            dtype=np.float32
        )

    window_list = []

    final_start_index = (
        number_of_frames
        - window_size
    )

    for start_index in range(
        0,
        final_start_index + 1,
        stride
    ):

        end_index = (
            start_index
            + window_size
        )

        window = normalized_array[
            start_index:end_index
        ]

        window_list.append(
            window
        )

    return np.stack(
        window_list
    ).astype(
        np.float32
    )


# ============================================================
# 10. Process a complete split
# ============================================================

def process_split(
    split_metadata_df: pd.DataFrame,
    split_name: str,
    split_output_folder: Path
) -> pd.DataFrame:
    """
    Normalize and divide every participant into fixed windows.
    """

    output_records = []

    failed_participants = []

    for number, participant_row in enumerate(
        split_metadata_df.itertuples(index=False),
        start=1
    ):

        participant_id = int(
            participant_row.Participant_ID
        )

        label = int(
            participant_row.Label
        )

        phq8_score = int(
            participant_row.PHQ8_Score
        )

        try:

            feature_array = load_feature_array(
                participant_row.Feature_File
            )

            # Apply Train statistics to both Train and Validation.
            normalized_array = (
                feature_array
                - train_mean
            ) / train_std

            if not np.isfinite(
                normalized_array
            ).all():

                raise ValueError(
                    "Normalized array contains "
                    "NaN or infinite values."
                )

            windows = create_windows(
                normalized_array=normalized_array,
                window_size=WINDOW_SIZE,
                stride=STRIDE
            )

            number_of_windows = (
                windows.shape[0]
            )

            if number_of_windows == 0:

                print(
                    f"Skipping participant {participant_id}: "
                    f"fewer than {WINDOW_SIZE} frames."
                )

                failed_participants.append(
                    participant_id
                )

                continue

            output_path = (
                split_output_folder
                / f"{participant_id}_windows.npy"
            )

            np.save(
                output_path,
                windows
            )

            used_frames = (
                number_of_windows
                * WINDOW_SIZE
            )

            dropped_frames = (
                feature_array.shape[0]
                - used_frames
            )

            output_records.append(
                {
                    "Participant_ID":
                        participant_id,

                    "Split":
                        split_name,

                    "Label":
                        label,

                    "PHQ8_Score":
                        phq8_score,

                    "Original_Successful_Frames":
                        feature_array.shape[0],

                    "Window_Size_Frames":
                        WINDOW_SIZE,

                    "Window_Duration_Seconds":
                        SECONDS_PER_WINDOW,

                    "Stride_Frames":
                        STRIDE,

                    "Number_of_Windows":
                        number_of_windows,

                    "Used_Frames":
                        used_frames,

                    "Dropped_End_Frames":
                        dropped_frames,

                    "Number_of_Features":
                        windows.shape[2],

                    "Windows_File":
                        str(output_path)
                }
            )

            if number % 20 == 0:

                print(
                    f"{split_name}: processed "
                    f"{number} of "
                    f"{len(split_metadata_df)} "
                    "participants..."
                )

        except Exception as error:

            print(
                f"Error processing participant "
                f"{participant_id}: {error}"
            )

            failed_participants.append(
                participant_id
            )


    output_df = pd.DataFrame(
        output_records
    )

    print(
        f"\n{split_name} successfully processed:"
    )

    print(
        len(output_df)
    )

    print(
        f"{split_name} failed or skipped:"
    )

    print(
        len(set(failed_participants))
    )

    if failed_participants:

        print(
            sorted(
                set(failed_participants)
            )
        )

    return output_df


# ============================================================
# 11. Create Train and Validation windows
# ============================================================

print("\nCreating training windows...")

train_windows_metadata_df = process_split(
    split_metadata_df=train_metadata_df,
    split_name="Train",
    split_output_folder=TRAIN_OUTPUT_FOLDER
)


print("\nCreating validation windows...")

validation_windows_metadata_df = process_split(
    split_metadata_df=validation_metadata_df,
    split_name="Validation",
    split_output_folder=VALIDATION_OUTPUT_FOLDER
)


# ============================================================
# 12. Save window metadata
# ============================================================

train_windows_metadata_path = (
    OUTPUT_FOLDER
    / "train_windows_metadata.csv"
)

validation_windows_metadata_path = (
    OUTPUT_FOLDER
    / "validation_windows_metadata.csv"
)

train_windows_metadata_df.to_csv(
    train_windows_metadata_path,
    index=False
)

validation_windows_metadata_df.to_csv(
    validation_windows_metadata_path,
    index=False
)


# ============================================================
# 13. Dataset summary
# ============================================================

def print_split_summary(
    windows_metadata_df: pd.DataFrame,
    split_name: str
):

    print("\n" + "=" * 70)
    print(f"{split_name.upper()} WINDOW SUMMARY")
    print("=" * 70)

    if windows_metadata_df.empty:

        print("No participants were processed.")
        return

    print("\nParticipants:")
    print(len(windows_metadata_df))

    print("\nTotal windows:")
    print(
        windows_metadata_df[
            "Number_of_Windows"
        ].sum()
    )

    print("\nWindows per participant:")
    print(
        windows_metadata_df[
            "Number_of_Windows"
        ].describe()
    )

    print("\nParticipant class counts:")
    print(
        windows_metadata_df[
            "Label"
        ].value_counts()
        .sort_index()
    )

    windows_by_class = (
        windows_metadata_df
        .groupby("Label")[
            "Number_of_Windows"
        ]
        .sum()
    )

    print("\nWindow counts by class:")
    print(
        windows_by_class
    )

    print("\nDropped end frames:")
    print(
        windows_metadata_df[
            "Dropped_End_Frames"
        ].describe()
    )


print_split_summary(
    train_windows_metadata_df,
    "Train"
)

print_split_summary(
    validation_windows_metadata_df,
    "Validation"
)


# ============================================================
# 14. Save combined metadata
# ============================================================

all_windows_metadata_df = pd.concat(
    [
        train_windows_metadata_df,
        validation_windows_metadata_df
    ],
    ignore_index=True
)

all_windows_metadata_path = (
    OUTPUT_FOLDER
    / "all_windows_metadata.csv"
)

all_windows_metadata_df.to_csv(
    all_windows_metadata_path,
    index=False
)


# ============================================================
# 15. Final message
# ============================================================

print("\n" + "=" * 70)
print("NORMALIZATION AND WINDOWING COMPLETED")
print("=" * 70)

print("\nWindow shape:")
print(
    f"({WINDOW_SIZE}, 32)"
)

print("\nTrain windows folder:")
print(TRAIN_OUTPUT_FOLDER)

print("\nValidation windows folder:")
print(VALIDATION_OUTPUT_FOLDER)

print("\nTrain metadata:")
print(train_windows_metadata_path)

print("\nValidation metadata:")
print(validation_windows_metadata_path)

print("\nCombined metadata:")
print(all_windows_metadata_path)