from pathlib import Path

import pandas as pd


DAIC_FOLDER = Path(
    r"D:\Downloads\Depression_Anxiety_Body_Movement\data\raw\daicwoz"
)

# نستخدم مشارك موجود عنده كل الملفات
PARTICIPANT_ID = 302

participant_folder = (
    DAIC_FOLDER
    / f"{PARTICIPANT_ID}_P"
)

files = {
    "Action Units": (
        participant_folder
        / f"{PARTICIPANT_ID}_CLNF_AUs.txt"
    ),
    "Gaze": (
        participant_folder
        / f"{PARTICIPANT_ID}_CLNF_gaze.txt"
    ),
    "Head Pose": (
        participant_folder
        / f"{PARTICIPANT_ID}_CLNF_pose.txt"
    )
}


for file_name, file_path in files.items():

    print("\n" + "=" * 70)
    print(file_name)
    print("=" * 70)

    print("File path:")
    print(file_path)

    if not file_path.exists():
        print("ERROR: File was not found.")
        continue

    try:
        df = pd.read_csv(
            file_path,
            sep=",",
            skipinitialspace=True,
            low_memory=False
        )

        df.columns = [
            str(column).strip()
            for column in df.columns
        ]

        print("\nShape:")
        print(df.shape)

        print("\nColumns:")
        for column in df.columns:
            print(repr(column))

        print("\nFirst 3 rows:")
        print(df.head(3))

    except Exception as error:
        print("\nCould not read file:")
        print(error)