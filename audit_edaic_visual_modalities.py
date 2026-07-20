from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import csv
import io
import json
import re
import tarfile


# ============================================================
# 1. Paths
# ============================================================

EDAIC_ARCHIVES_FOLDER = Path(
    r"D:\Downloads\Depression_Anxiety_Body_Movement\data\raw\e-daic-woz"
)

OUTPUT_FOLDER = Path(
    r"D:\Flutterr\multimodal-depression-from-video"
    r"\dataset_audit_outputs"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_JSON_PATH = (
    OUTPUT_FOLDER
    / "edaic_visual_file_inventory.json"
)

FILE_COUNTS_CSV_PATH = (
    OUTPUT_FOLDER
    / "edaic_archive_file_counts.csv"
)

MATCHED_FILES_CSV_PATH = (
    OUTPUT_FOLDER
    / "edaic_matched_visual_files.csv"
)

HEADERS_CSV_PATH = (
    OUTPUT_FOLDER
    / "edaic_visual_csv_headers.csv"
)


# ============================================================
# 2. Search terms
# ============================================================

CATEGORY_PATTERNS = {
    "openface_combined":
        re.compile(
            r"openface.*pose.*gaze.*au|pose.*gaze.*au",
            re.IGNORECASE,
        ),

    "action_units":
        re.compile(
            r"\bau\b|aus|action.?unit",
            re.IGNORECASE,
        ),

    "gaze":
        re.compile(
            r"gaze",
            re.IGNORECASE,
        ),

    "head_pose":
        re.compile(
            r"pose|head",
            re.IGNORECASE,
        ),

    "facial_landmarks":
        re.compile(
            r"landmark|landmarks|clnf_features|2d|3d",
            re.IGNORECASE,
        ),

    "face_embeddings":
        re.compile(
            r"face.?embedding|embedding|embeddings",
            re.IGNORECASE,
        ),

    "body_landmarks":
        re.compile(
            r"body|pose_body|skeleton|mediapipe_pose",
            re.IGNORECASE,
        ),

    "hand_landmarks":
        re.compile(
            r"hand|hands",
            re.IGNORECASE,
        ),

    "blink":
        re.compile(
            r"blink|blinking|eye.?aspect|ear",
            re.IGNORECASE,
        ),

    "video":
        re.compile(
            r"\.(mp4|avi|mov|mkv)$",
            re.IGNORECASE,
        ),

    "audio":
        re.compile(
            r"\.(wav|mp3|flac)$",
            re.IGNORECASE,
        ),

    "transcript":
        re.compile(
            r"transcript|\.txt$",
            re.IGNORECASE,
        ),
}


# ============================================================
# 3. Helpers
# ============================================================

def get_participant_id(
    archive_path: Path,
) -> int | None:

    match = re.search(
        r"(\d+)",
        archive_path.name,
    )

    if match is None:
        return None

    return int(
        match.group(1)
    )


def categorize_member(
    member_name: str,
) -> list[str]:

    categories = []

    for category_name, pattern in (
        CATEGORY_PATTERNS.items()
    ):
        if pattern.search(
            member_name
        ):
            categories.append(
                category_name
            )

    return categories


def read_csv_header_from_tar(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
) -> list[str]:

    extracted_file = archive.extractfile(
        member
    )

    if extracted_file is None:
        return []

    raw_line = extracted_file.readline()

    try:
        decoded_line = raw_line.decode(
            "utf-8-sig"
        )

    except UnicodeDecodeError:
        decoded_line = raw_line.decode(
            "latin-1",
            errors="replace",
        )

    reader = csv.reader(
        io.StringIO(
            decoded_line
        )
    )

    try:
        return [
            column.strip()
            for column in next(
                reader
            )
        ]

    except StopIteration:
        return []


# ============================================================
# 4. Find archives
# ============================================================

if not EDAIC_ARCHIVES_FOLDER.exists():
    raise FileNotFoundError(
        "E-DAIC archive folder was not found:\n"
        f"{EDAIC_ARCHIVES_FOLDER}"
    )


archive_paths = sorted(
    list(EDAIC_ARCHIVES_FOLDER.glob("*.tar.gz"))
    + list(EDAIC_ARCHIVES_FOLDER.glob("*.tgz"))
)[:10]


if not archive_paths:
    raise FileNotFoundError(
        "No .tar.gz or .tgz archives were found in:\n"
        f"{EDAIC_ARCHIVES_FOLDER}"
    )


print("=" * 76)
print("E-DAIC-WOZ VISUAL FILE INVENTORY")
print("=" * 76)

print(
    f"\nArchives found: "
    f"{len(archive_paths)}"
)


# ============================================================
# 5. Inspect all archives
# ============================================================

category_archive_counts = Counter()
extension_counts = Counter()

archive_records = []
matched_file_records = []
header_records = []

example_files_by_category = defaultdict(
    list
)

archives_with_errors = []


for archive_index, archive_path in enumerate(
    archive_paths,
    start=1,
):

    participant_id = get_participant_id(
        archive_path
    )

    archive_category_flags = {
        category_name: False
        for category_name in CATEGORY_PATTERNS
    }

    total_files = 0

    try:
        with tarfile.open(
            archive_path,
            mode="r:gz",
        ) as archive:

            members = [
                member
                for member in archive.getmembers()
                if member.isfile()
            ]

            total_files = len(
                members
            )

            for member in members:

                member_path = Path(
                    member.name
                )

                suffixes = "".join(
                    member_path.suffixes
                ).lower()

                extension = (
                    suffixes
                    if suffixes
                    else "[no extension]"
                )

                extension_counts[
                    extension
                ] += 1

                categories = categorize_member(
                    member.name
                )

                for category_name in categories:

                    archive_category_flags[
                        category_name
                    ] = True

                    if (
                        len(
                            example_files_by_category[
                                category_name
                            ]
                        )
                        < 10
                    ):
                        example_files_by_category[
                            category_name
                        ].append(
                            member.name
                        )

                    matched_file_records.append(
                        {
                            "Participant_ID":
                                participant_id,

                            "Archive":
                                archive_path.name,

                            "Category":
                                category_name,

                            "Member_Name":
                                member.name,

                            "Size_Bytes":
                                int(
                                    member.size
                                ),
                        }
                    )

                if (
                    member.name.lower()
                    .endswith(".csv")
                    and categories
                ):
                    header = (
                        read_csv_header_from_tar(
                            archive,
                            member,
                        )
                    )

                    header_records.append(
                        {
                            "Participant_ID":
                                participant_id,

                            "Archive":
                                archive_path.name,

                            "Member_Name":
                                member.name,

                            "Categories":
                                "|".join(
                                    categories
                                ),

                            "Number_of_Columns":
                                len(
                                    header
                                ),

                            "Columns":
                                "|".join(
                                    header
                                ),
                        }
                    )

    except Exception as error:

        archives_with_errors.append(
            {
                "Archive":
                    archive_path.name,

                "Error":
                    str(
                        error
                    ),
            }
        )

        print(
            f"\nERROR in "
            f"{archive_path.name}: "
            f"{error}"
        )

        continue


    for category_name, is_present in (
        archive_category_flags.items()
    ):
        if is_present:
            category_archive_counts[
                category_name
            ] += 1


    archive_record = {
        "Participant_ID":
            participant_id,

        "Archive":
            archive_path.name,

        "Total_Files":
            total_files,
    }

    for category_name, is_present in (
        archive_category_flags.items()
    ):
        archive_record[
            category_name
        ] = int(
            is_present
        )

    archive_records.append(
        archive_record
    )


    if (
        archive_index % 25 == 0
        or archive_index
        == len(archive_paths)
    ):
        print(
            f"{archive_index}/"
            f"{len(archive_paths)} "
            f"archives inspected"
        )


# ============================================================
# 6. Save CSV outputs
# ============================================================

import pandas as pd


archive_df = pd.DataFrame(
    archive_records
)

matched_files_df = pd.DataFrame(
    matched_file_records
)

headers_df = pd.DataFrame(
    header_records
)


archive_df.to_csv(
    FILE_COUNTS_CSV_PATH,
    index=False,
)

matched_files_df.to_csv(
    MATCHED_FILES_CSV_PATH,
    index=False,
)

headers_df.to_csv(
    HEADERS_CSV_PATH,
    index=False,
)


# ============================================================
# 7. Summary
# ============================================================

summary = {
    "archives_folder":
        str(
            EDAIC_ARCHIVES_FOLDER
        ),

    "archives_found":
        len(
            archive_paths
        ),

    "archives_successfully_inspected":
        len(
            archive_records
        ),

    "archives_with_errors":
        archives_with_errors,

    "archive_counts_by_category": {
        category_name:
            int(
                category_archive_counts[
                    category_name
                ]
            )

        for category_name in (
            CATEGORY_PATTERNS
        )
    },

    "top_file_extensions": [
        {
            "extension":
                extension,

            "count":
                int(
                    count
                ),
        }

        for extension, count in (
            extension_counts
            .most_common(
                30
            )
        )
    ],

    "example_files_by_category": {
        category_name:
            examples

        for category_name, examples in (
            example_files_by_category
            .items()
        )
    },
}


with open(
    SUMMARY_JSON_PATH,
    "w",
    encoding="utf-8",
) as output_file:

    json.dump(
        summary,
        output_file,
        indent=4,
        ensure_ascii=False,
    )


# ============================================================
# 8. Print important findings
# ============================================================

print("\n" + "=" * 76)
print("FILES AVAILABLE BY CATEGORY")
print("=" * 76)

for category_name in (
    CATEGORY_PATTERNS
):
    print(
        f"{category_name}: "
        f"{category_archive_counts[category_name]}"
        f"/{len(archive_records)} archives"
    )


print("\n" + "=" * 76)
print("EXAMPLE MATCHED FILES")
print("=" * 76)

for category_name in (
    CATEGORY_PATTERNS
):
    examples = (
        example_files_by_category.get(
            category_name,
            [],
        )
    )

    print(
        f"\n{category_name}:"
    )

    if examples:
        for example in examples[:3]:
            print(
                f"  - {example}"
            )

    else:
        print(
            "  No matching files found."
        )


print("\n" + "=" * 76)
print("IMPORTANT NEXT-STEP CHECK")
print("=" * 76)

needed_categories = [
    "facial_landmarks",
    "face_embeddings",
    "body_landmarks",
    "hand_landmarks",
    "blink",
    "video",
]

for category_name in needed_categories:

    count = category_archive_counts[
        category_name
    ]

    if count > 0:
        status = "AVAILABLE"
    else:
        status = "NOT FOUND"

    print(
        f"{category_name}: "
        f"{status} "
        f"({count} archives)"
    )


print("\nOutputs saved in:")
print(
    OUTPUT_FOLDER
)

print("\nMost important file:")
print(
    SUMMARY_JSON_PATH
)