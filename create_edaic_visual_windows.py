from __future__ import annotations

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
# 0–269, 270–539, 540–809...
STRIDE = WINDOW_SIZE

EXPECTED_FEATURE_COUNT = 28


# ============================================================
# 2. Paths
# ============================================================

FEATURE_FOLDER = Path(
    r"D:\Downloads\Depression_Anxiety_Body_Movement\data\processed"
    r"\e_daic_woz_visual"
)

METADATA_PATH = (
    FEATURE_FOLDER
    / "edaic_visual_feature_metadata.csv"
)

OUTPUT_FOLDER = Path(
    r"D:\Downloads\Depression_Anxiety_Body_Movement\data\processed"
    r"\e_daic_woz_visual_windows"
)

TRAIN_OUTPUT_FOLDER = OUTPUT_FOLDER / "train"
DEV_OUTPUT_FOLDER = OUTPUT_FOLDER / "dev"
TEST_OUTPUT_FOLDER = OUTPUT_FOLDER / "test"

TRAIN_OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

DEV_OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

TEST_OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 3. Check metadata
# ============================================================

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Metadata file was not found:\n{METADATA_PATH}"
    )


# ============================================================
# 4. Read metadata
# ============================================================

metadata_df = pd.read_csv(
    METADATA_PATH
)

required_metadata_columns = [
    "Participant_ID",
    "Split",
    "Label",
    "PHQ_Score",
    "Feature_File",
]

missing_metadata_columns = [
    column
    for column in required_metadata_columns
    if column not in metadata_df.columns
]

if missing_metadata_columns:
    raise ValueError(
        "Metadata is missing columns: "
        f"{missing_metadata_columns}"
    )


metadata_df["Participant_ID"] = pd.to_numeric(
    metadata_df["Participant_ID"],
    errors="raise",
).astype(int)

metadata_df["Label"] = pd.to_numeric(
    metadata_df["Label"],
    errors="raise",
).astype(int)

metadata_df["PHQ_Score"] = pd.to_numeric(
    metadata_df["PHQ_Score"],
    errors="raise",
).astype(int)

metadata_df["Split"] = (
    metadata_df["Split"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# 5. Separate splits
# ============================================================

train_metadata_df = metadata_df[
    metadata_df["Split"] == "train"
].copy()

dev_metadata_df = metadata_df[
    metadata_df["Split"] == "dev"
].copy()

test_metadata_df = metadata_df[
    metadata_df["Split"] == "test"
].copy()


print("=" * 70)
print("E-DAIC-WOZ NORMALIZATION AND WINDOWING")
print("=" * 70)

print("\nParticipants available:")

print(
    "Train:",
    len(train_metadata_df),
)

print(
    "Dev:",
    len(dev_metadata_df),
)

print(
    "Test:",
    len(test_metadata_df),
)


# ============================================================
# 6. Check participant overlap
# ============================================================

train_ids = set(
    train_metadata_df["Participant_ID"]
)

dev_ids = set(
    dev_metadata_df["Participant_ID"]
)

test_ids = set(
    test_metadata_df["Participant_ID"]
)

train_dev_overlap = train_ids & dev_ids
train_test_overlap = train_ids & test_ids
dev_test_overlap = dev_ids & test_ids

if (
    train_dev_overlap
    or train_test_overlap
    or dev_test_overlap
):
    raise ValueError(
        "Participant overlap was found between splits.\n"
        f"Train/Dev: {sorted(train_dev_overlap)}\n"
        f"Train/Test: {sorted(train_test_overlap)}\n"
        f"Dev/Test: {sorted(dev_test_overlap)}"
    )

print(
    "\nNo participant overlap between "
    "Train, Dev, and Test."
)


# ============================================================
# 7. Load one participant array
# ============================================================

def load_feature_array(
    feature_path: str,
) -> np.ndarray:
    """
    Load one participant's visual feature array.
    """

    path = Path(
        feature_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Feature file was not found: {path}"
        )

    feature_array = np.load(
        path
    )

    if feature_array.ndim != 2:
        raise ValueError(
            f"Expected a 2D array, found "
            f"{feature_array.shape} in {path}"
        )

    if (
        feature_array.shape[1]
        != EXPECTED_FEATURE_COUNT
    ):
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, "
            f"found {feature_array.shape[1]} in {path}"
        )

    if not np.isfinite(
        feature_array
    ).all():
        raise ValueError(
            f"NaN or infinity found in {path}"
        )

    return feature_array.astype(
        np.float64,
        copy=False,
    )


# ============================================================
# 8. Calculate normalization from Train only
# ============================================================

print(
    "\nCalculating normalization statistics "
    "from Train only..."
)

feature_sum = np.zeros(
    EXPECTED_FEATURE_COUNT,
    dtype=np.float64,
)

feature_squared_sum = np.zeros(
    EXPECTED_FEATURE_COUNT,
    dtype=np.float64,
)

total_train_frames = 0


for number, participant_row in enumerate(
    train_metadata_df.itertuples(index=False),
    start=1,
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
    raise ValueError(
        "No Train frames were found."
    )


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

# Prevent tiny negative values caused by floating-point errors.
train_variance = np.maximum(
    train_variance,
    0,
)

train_std = np.sqrt(
    train_variance
)

# Avoid dividing by zero.
train_std[
    train_std < 1e-8
] = 1.0


print(
    "\nTotal Train frames used for normalization:"
)

print(
    total_train_frames
)

print(
    "\nMean shape:"
)

print(
    train_mean.shape
)

print(
    "\nStandard deviation shape:"
)

print(
    train_std.shape
)


# ============================================================
# 9. Save normalization parameters
# ============================================================

NORMALIZATION_PATH = (
    OUTPUT_FOLDER
    / "train_normalization_parameters.npz"
)

np.savez(
    NORMALIZATION_PATH,

    mean=train_mean.astype(
        np.float32
    ),

    std=train_std.astype(
        np.float32
    ),

    total_train_frames=np.array(
        [total_train_frames],
        dtype=np.int64,
    ),

    window_size=np.array(
        [WINDOW_SIZE],
        dtype=np.int64,
    ),

    seconds_per_window=np.array(
        [SECONDS_PER_WINDOW],
        dtype=np.int64,
    ),

    fps=np.array(
        [FPS],
        dtype=np.int64,
    ),
)

print(
    "\nNormalization parameters saved:"
)

print(
    NORMALIZATION_PATH
)


# ============================================================
# 10. Windowing function
# ============================================================

def create_windows(
    normalized_array: np.ndarray,
    window_size: int,
    stride: int,
) -> np.ndarray:
    """
    Divide one participant sequence into fixed-size windows.

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
                number_of_features,
            ),
            dtype=np.float32,
        )

    windows = []

    last_start = (
        number_of_frames
        - window_size
    )

    for start_index in range(
        0,
        last_start + 1,
        stride,
    ):
        end_index = (
            start_index
            + window_size
        )

        window = normalized_array[
            start_index:end_index
        ]

        windows.append(
            window
        )

    return np.stack(
        windows
    ).astype(
        np.float32
    )


# ============================================================
# 11. Process one split
# ============================================================

def process_split(
    split_metadata_df: pd.DataFrame,
    split_name: str,
    split_output_folder: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalize and create windows for one complete split.
    """

    metadata_records = []
    failed_records = []

    print(
        f"\nCreating {split_name} windows..."
    )

    for number, participant_row in enumerate(
        split_metadata_df.itertuples(index=False),
        start=1,
    ):
        participant_id = int(
            participant_row.Participant_ID
        )

        try:
            feature_array = load_feature_array(
                participant_row.Feature_File
            )

            normalized_array = (
                feature_array
                - train_mean
            ) / train_std

            if not np.isfinite(
                normalized_array
            ).all():
                raise ValueError(
                    "Normalized array contains "
                    "NaN or infinity."
                )

            windows = create_windows(
                normalized_array=normalized_array,
                window_size=WINDOW_SIZE,
                stride=STRIDE,
            )

            number_of_windows = (
                windows.shape[0]
            )

            if number_of_windows == 0:
                raise ValueError(
                    f"Participant has fewer than "
                    f"{WINDOW_SIZE} successful frames."
                )

            windows_path = (
                split_output_folder
                / f"{participant_id}_windows.npy"
            )

            np.save(
                windows_path,
                windows,
            )

            used_frames = (
                number_of_windows
                * WINDOW_SIZE
            )

            dropped_end_frames = (
                feature_array.shape[0]
                - used_frames
            )

            metadata_records.append(
                {
                    "Participant_ID":
                        participant_id,

                    "Split":
                        split_name,

                    "Label":
                        int(
                            participant_row.Label
                        ),

                    "PHQ_Score":
                        int(
                            participant_row.PHQ_Score
                        ),

                    "Original_Successful_Frames":
                        feature_array.shape[0],

                    "Window_Size_Frames":
                        WINDOW_SIZE,

                    "Window_Duration_Seconds":
                        SECONDS_PER_WINDOW,

                    "FPS":
                        FPS,

                    "Stride_Frames":
                        STRIDE,

                    "Number_of_Windows":
                        number_of_windows,

                    "Used_Frames":
                        used_frames,

                    "Dropped_End_Frames":
                        dropped_end_frames,

                    "Number_of_Features":
                        windows.shape[2],

                    "Windows_File":
                        str(
                            windows_path
                        ),
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
                f"{split_name}: participant "
                f"{participant_id} failed: {error}"
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

    return (
        pd.DataFrame(
            metadata_records
        ),
        pd.DataFrame(
            failed_records
        ),
    )


# ============================================================
# 12. Create Train / Dev / Test windows
# ============================================================

train_windows_df, train_failed_df = process_split(
    split_metadata_df=train_metadata_df,
    split_name="train",
    split_output_folder=TRAIN_OUTPUT_FOLDER,
)

dev_windows_df, dev_failed_df = process_split(
    split_metadata_df=dev_metadata_df,
    split_name="dev",
    split_output_folder=DEV_OUTPUT_FOLDER,
)

test_windows_df, test_failed_df = process_split(
    split_metadata_df=test_metadata_df,
    split_name="test",
    split_output_folder=TEST_OUTPUT_FOLDER,
)


# ============================================================
# 13. Save metadata files
# ============================================================

TRAIN_METADATA_PATH = (
    OUTPUT_FOLDER
    / "train_windows_metadata.csv"
)

DEV_METADATA_PATH = (
    OUTPUT_FOLDER
    / "dev_windows_metadata.csv"
)

TEST_METADATA_PATH = (
    OUTPUT_FOLDER
    / "test_windows_metadata.csv"
)

ALL_METADATA_PATH = (
    OUTPUT_FOLDER
    / "all_windows_metadata.csv"
)

FAILED_METADATA_PATH = (
    OUTPUT_FOLDER
    / "failed_window_participants.csv"
)


train_windows_df.to_csv(
    TRAIN_METADATA_PATH,
    index=False,
)

dev_windows_df.to_csv(
    DEV_METADATA_PATH,
    index=False,
)

test_windows_df.to_csv(
    TEST_METADATA_PATH,
    index=False,
)


all_windows_df = pd.concat(
    [
        train_windows_df,
        dev_windows_df,
        test_windows_df,
    ],
    ignore_index=True,
)

all_windows_df.to_csv(
    ALL_METADATA_PATH,
    index=False,
)


all_failed_df = pd.concat(
    [
        train_failed_df,
        dev_failed_df,
        test_failed_df,
    ],
    ignore_index=True,
)

all_failed_df.to_csv(
    FAILED_METADATA_PATH,
    index=False,
)


# ============================================================
# 14. Print summary
# ============================================================

def print_split_summary(
    dataframe: pd.DataFrame,
    split_name: str,
) -> None:
    """
    Print participant and window statistics.
    """

    print("\n" + "=" * 70)
    print(
        f"{split_name.upper()} WINDOW SUMMARY"
    )
    print("=" * 70)

    if dataframe.empty:
        print(
            "No participants were processed."
        )
        return

    print("\nParticipants:")

    print(
        len(dataframe)
    )

    print("\nTotal windows:")

    print(
        int(
            dataframe[
                "Number_of_Windows"
            ].sum()
        )
    )

    print("\nWindows per participant:")

    print(
        dataframe[
            "Number_of_Windows"
        ].describe()
    )

    print("\nParticipant class counts:")

    print(
        dataframe[
            "Label"
        ]
        .value_counts()
        .sort_index()
    )

    print("\nWindow counts by class:")

    print(
        dataframe
        .groupby(
            "Label"
        )[
            "Number_of_Windows"
        ]
        .sum()
        .sort_index()
    )

    print("\nDropped end frames:")

    print(
        dataframe[
            "Dropped_End_Frames"
        ].describe()
    )


print_split_summary(
    train_windows_df,
    "Train",
)

print_split_summary(
    dev_windows_df,
    "Dev",
)

print_split_summary(
    test_windows_df,
    "Test",
)


# ============================================================
# 15. Final message
# ============================================================

print("\n" + "=" * 70)
print("E-DAIC-WOZ WINDOWING COMPLETED")
print("=" * 70)

print("\nWindow shape:")

print(
    f"({WINDOW_SIZE}, "
    f"{EXPECTED_FEATURE_COUNT})"
)

print("\nTrain output:")
print(TRAIN_OUTPUT_FOLDER)

print("\nDev output:")
print(DEV_OUTPUT_FOLDER)

print("\nTest output:")
print(TEST_OUTPUT_FOLDER)

print("\nNormalization parameters:")
print(NORMALIZATION_PATH)

print("\nCombined metadata:")
print(ALL_METADATA_PATH)

print("\nFailed window participants:")
print(FAILED_METADATA_PATH)