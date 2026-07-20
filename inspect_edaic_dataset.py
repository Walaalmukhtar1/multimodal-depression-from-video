from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

EDAIC_FOLDER = Path(
    r"D:\Downloads\Depression_Anxiety_Body_Movement\data\raw\e-daic-woz"
)

TRAIN_PATH = EDAIC_FOLDER / "train_split.csv"
DEV_PATH = EDAIC_FOLDER / "dev_split.csv"
TEST_PATH = EDAIC_FOLDER / "test_split.csv"
DETAILED_PATH = EDAIC_FOLDER / "detailed_lables.csv"


# ============================================================
# Read labels
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)
dev_df = pd.read_csv(DEV_PATH)
test_df = pd.read_csv(TEST_PATH)
detailed_df = pd.read_csv(DETAILED_PATH)


# ============================================================
# Get labeled participant IDs
# ============================================================

train_ids = set(
    train_df["Participant_ID"].astype(int)
)

dev_ids = set(
    dev_df["Participant_ID"].astype(int)
)

test_ids = set(
    test_df["Participant_ID"].astype(int)
)

all_label_ids = (
    train_ids
    | dev_ids
    | test_ids
)


# ============================================================
# Get archive participant IDs
# ============================================================

archive_files = list(
    EDAIC_FOLDER.glob("*_P.tar.gz")
)

archive_ids = set()

for archive_path in archive_files:

    participant_text = (
        archive_path.name
        .replace("_P.tar.gz", "")
    )

    try:
        participant_id = int(
            participant_text
        )

        archive_ids.add(
            participant_id
        )

    except ValueError:
        print(
            "Could not read participant ID from:",
            archive_path.name
        )


# ============================================================
# Compare labels and archives
# ============================================================

missing_archives = sorted(
    all_label_ids - archive_ids
)

archives_without_labels = sorted(
    archive_ids - all_label_ids
)


# ============================================================
# Print summary
# ============================================================

print("=" * 70)
print("E-DAIC-WOZ DATASET INSPECTION")
print("=" * 70)

print("\nSplit counts:")

print("Train:", len(train_ids))
print("Dev:", len(dev_ids))
print("Test:", len(test_ids))

print("\nTotal labeled participants:")
print(len(all_label_ids))

print("\nTotal participant archives:")
print(len(archive_ids))

print("\nOverlap checks:")

print(
    "Train and Dev:",
    sorted(train_ids & dev_ids)
)

print(
    "Train and Test:",
    sorted(train_ids & test_ids)
)

print(
    "Dev and Test:",
    sorted(dev_ids & test_ids)
)

print("\nParticipants with labels but no archive:")
print(missing_archives)

print("\nNumber missing:")
print(len(missing_archives))

print("\nArchives without labels:")
print(archives_without_labels)

print("\nNumber without labels:")
print(len(archives_without_labels))


# ============================================================
# Class distribution
# ============================================================

def print_class_distribution(
    dataframe: pd.DataFrame,
    split_name: str
):

    counts = (
        dataframe["PHQ_Binary"]
        .value_counts()
        .sort_index()
    )

    print(
        f"\n{split_name} class distribution:"
    )

    print(counts)


print_class_distribution(
    train_df,
    "Train"
)

print_class_distribution(
    dev_df,
    "Dev"
)

print_class_distribution(
    test_df,
    "Test"
)


# ============================================================
# Detailed labels check
# ============================================================

detailed_ids = set(
    detailed_df["Participant"]
    .astype(int)
)

print("\nDetailed label participants:")
print(len(detailed_ids))

print("\nSplit IDs missing from detailed labels:")
print(
    sorted(
        all_label_ids - detailed_ids
    )
)

print("\nInspection completed.")