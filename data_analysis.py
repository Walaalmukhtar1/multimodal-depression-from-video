from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Paths
# ============================================================

daic_folder = Path(
    r"D:\Downloads\Depression_Anxiety_Body_Movement\data\raw\daicwoz"
)

labels_path = (
    daic_folder
    / "complete_Depression_AVEC2017.csv"
)

output_folder = (
    Path(__file__).resolve().parent
    / "analysis_outputs"
    / "daic_woz"
    / "class_feature_analysis"
)

output_folder.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. Check required paths
# ============================================================

if not daic_folder.exists():
    print("ERROR: DAIC-WOZ folder was not found.")
    print(daic_folder)
    raise SystemExit

if not labels_path.exists():
    print("ERROR: DAIC-WOZ labels file was not found.")
    print(labels_path)
    raise SystemExit


# ============================================================
# 3. Read labels
# ============================================================

labels_df = pd.read_csv(labels_path)

# Keep only participants with valid labels.
# Test participants have PHQ8_Binary = -1.
labeled_df = labels_df[
    labels_df["PHQ8_Binary"].isin([0, 1])
].copy()

labeled_df["Class"] = labeled_df[
    "PHQ8_Binary"
].map({
    0: "Control",
    1: "Depression"
})

labeled_df["Participant_ID"] = (
    labeled_df["Participant_ID"]
    .astype(int)
)

print("Total labeled participants:")
print(len(labeled_df))

print("\nClass counts:")
print(labeled_df["Class"].value_counts())


# ============================================================
# 4. Helper functions
# ============================================================

def read_feature_file(file_path):
    """
    Read an OpenFace/DAIC-WOZ feature file.
    """

    try:
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

    except Exception as error:
        print(f"\nCould not read: {file_path}")
        print(f"Reason: {error}")
        return None


def find_column(dataframe, possible_names):
    """
    Find a column without depending on capitalization.
    """

    normalized_columns = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for possible_name in possible_names:

        normalized_name = (
            possible_name.strip().lower()
        )

        if normalized_name in normalized_columns:
            return normalized_columns[normalized_name]

    return None


def keep_successful_frames(dataframe):
    """
    Keep only frames where visual tracking succeeded.

    If no success column exists, use all frames.
    """

    success_column = find_column(
        dataframe,
        ["success", "tracking_success"]
    )

    if success_column is None:
        return dataframe.copy()

    success_values = pd.to_numeric(
        dataframe[success_column],
        errors="coerce"
    )

    filtered_dataframe = dataframe[
        success_values == 1
    ].copy()

    return filtered_dataframe


def numeric_mean(dataframe, column_name):
    """
    Calculate a numeric column mean safely.
    """

    if column_name not in dataframe.columns:
        return np.nan

    values = pd.to_numeric(
        dataframe[column_name],
        errors="coerce"
    )

    values = values.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return values.mean()


def numeric_std(dataframe, column_name):
    """
    Calculate a numeric column standard deviation safely.
    """

    if column_name not in dataframe.columns:
        return np.nan

    values = pd.to_numeric(
        dataframe[column_name],
        errors="coerce"
    )

    values = values.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return values.std()


def calculate_cohens_d(control_values, depression_values):
    """
    Calculate Cohen's d.

    Positive d:
    Depression mean is higher.

    Negative d:
    Control mean is higher.
    """

    control_values = pd.Series(
        control_values
    ).dropna()

    depression_values = pd.Series(
        depression_values
    ).dropna()

    n_control = len(control_values)
    n_depression = len(depression_values)

    if n_control < 2 or n_depression < 2:
        return np.nan

    control_variance = control_values.var(
        ddof=1
    )

    depression_variance = depression_values.var(
        ddof=1
    )

    pooled_variance = (
        (
            (n_control - 1) * control_variance
        )
        +
        (
            (n_depression - 1)
            * depression_variance
        )
    ) / (
        n_control
        + n_depression
        - 2
    )

    if pooled_variance <= 0:
        return np.nan

    pooled_standard_deviation = np.sqrt(
        pooled_variance
    )

    effect_size = (
        depression_values.mean()
        - control_values.mean()
    ) / pooled_standard_deviation

    return effect_size


# ============================================================
# 5. Expected files
# ============================================================

head_pose_pattern = "{id}_CLNF_pose.txt"
gaze_pattern = "{id}_CLNF_gaze.txt"
au_pattern = "{id}_CLNF_AUs.txt"


# ============================================================
# 6. Extract participant-level features
# ============================================================

participant_records = []

total_participants = len(labeled_df)

for number, participant_row in enumerate(
    labeled_df.itertuples(index=False),
    start=1
):
    participant_id = int(
        participant_row.Participant_ID
    )

    participant_class = (
        participant_row.Class
    )

    phq8_score = (
        participant_row.PHQ8_Score
    )

    participant_folder = (
        daic_folder
        / f"{participant_id}_P"
    )

    record = {
        "Participant_ID": participant_id,
        "Class": participant_class,
        "PHQ8_Score": phq8_score
    }

    # --------------------------------------------------------
    # Head pose features
    # --------------------------------------------------------

    head_pose_path = (
        participant_folder
        / head_pose_pattern.format(
            id=participant_id
        )
    )

    if head_pose_path.exists():

        pose_df = read_feature_file(
            head_pose_path
        )

        if pose_df is not None:

            pose_df = keep_successful_frames(
                pose_df
            )

            pose_columns = [
                "pose_Tx",
                "pose_Ty",
                "pose_Tz",
                "pose_Rx",
                "pose_Ry",
                "pose_Rz"
            ]

            for column in pose_columns:

                record[
                    f"{column}_mean"
                ] = numeric_mean(
                    pose_df,
                    column
                )

                record[
                    f"{column}_std"
                ] = numeric_std(
                    pose_df,
                    column
                )

    # --------------------------------------------------------
    # Gaze features
    # --------------------------------------------------------

    gaze_path = (
        participant_folder
        / gaze_pattern.format(
            id=participant_id
        )
    )

    if gaze_path.exists():

        gaze_df = read_feature_file(
            gaze_path
        )

        if gaze_df is not None:

            gaze_df = keep_successful_frames(
                gaze_df
            )

            gaze_columns = [
                "gaze_angle_x",
                "gaze_angle_y"
            ]

            for column in gaze_columns:

                record[
                    f"{column}_mean"
                ] = numeric_mean(
                    gaze_df,
                    column
                )

                record[
                    f"{column}_std"
                ] = numeric_std(
                    gaze_df,
                    column
                )

    # --------------------------------------------------------
    # Action Unit intensity features
    # --------------------------------------------------------

    au_path = (
        participant_folder
        / au_pattern.format(
            id=participant_id
        )
    )

    if au_path.exists():

        au_df = read_feature_file(
            au_path
        )

        if au_df is not None:

            au_df = keep_successful_frames(
                au_df
            )

            # OpenFace AU intensity columns usually end with "_r".
            # Columns ending with "_c" are binary occurrence values.
            au_intensity_columns = [
                column
                for column in au_df.columns
                if str(column).strip().endswith("_r")
            ]

            for column in au_intensity_columns:

                clean_column_name = (
                    str(column).strip()
                )

                record[
                    f"{clean_column_name}_mean"
                ] = numeric_mean(
                    au_df,
                    column
                )

                record[
                    f"{clean_column_name}_std"
                ] = numeric_std(
                    au_df,
                    column
                )

    participant_records.append(record)

    if number % 20 == 0:

        print(
            f"Processed {number} of "
            f"{total_participants} participants..."
        )


participant_features_df = pd.DataFrame(
    participant_records
)


# ============================================================
# 7. Save participant feature table
# ============================================================

participant_features_df.to_csv(
    output_folder
    / "participant_behavioral_features.csv",
    index=False
)

print("\nParticipant feature table shape:")
print(participant_features_df.shape)

print("\nParticipant feature columns:")
print(participant_features_df.columns.tolist())


# ============================================================
# 8. Identify usable numeric features
# ============================================================

metadata_columns = {
    "Participant_ID",
    "Class",
    "PHQ8_Score"
}

feature_columns = [
    column
    for column in participant_features_df.columns
    if column not in metadata_columns
]

usable_feature_columns = []

for feature in feature_columns:

    valid_percentage = (
        participant_features_df[feature]
        .notna()
        .mean()
        * 100
    )

    # Keep features available for at least 90% of participants.
    if valid_percentage >= 90:
        usable_feature_columns.append(
            feature
        )

print("\nTotal extracted features:")
print(len(feature_columns))

print("\nUsable features:")
print(len(usable_feature_columns))


# ============================================================
# 9. Class-wise statistical summary
# ============================================================

summary_records = []

for feature in usable_feature_columns:

    control_values = participant_features_df.loc[
        participant_features_df["Class"]
        == "Control",
        feature
    ].dropna()

    depression_values = participant_features_df.loc[
        participant_features_df["Class"]
        == "Depression",
        feature
    ].dropna()

    cohens_d = calculate_cohens_d(
        control_values,
        depression_values
    )

    summary_records.append(
        {
            "Feature": feature,
            "Control_Count": len(control_values),
            "Depression_Count": len(
                depression_values
            ),
            "Control_Mean": control_values.mean(),
            "Depression_Mean":
                depression_values.mean(),
            "Control_Median":
                control_values.median(),
            "Depression_Median":
                depression_values.median(),
            "Control_Std": control_values.std(),
            "Depression_Std":
                depression_values.std(),
            "Mean_Difference_Depression_Minus_Control":
                (
                    depression_values.mean()
                    - control_values.mean()
                ),
            "Cohens_d": cohens_d,
            "Absolute_Cohens_d": (
                abs(cohens_d)
                if pd.notna(cohens_d)
                else np.nan
            )
        }
    )


feature_summary_df = pd.DataFrame(
    summary_records
)

feature_summary_df = (
    feature_summary_df
    .sort_values(
        "Absolute_Cohens_d",
        ascending=False
    )
    .reset_index(drop=True)
)

feature_summary_df.to_csv(
    output_folder
    / "class_feature_comparison.csv",
    index=False
)


# ============================================================
# 10. Print strongest class differences
# ============================================================

print("\n" + "=" * 85)
print("TOP FEATURES BY ABSOLUTE COHEN'S D")
print("=" * 85)

display_columns = [
    "Feature",
    "Control_Mean",
    "Depression_Mean",
    "Mean_Difference_Depression_Minus_Control",
    "Cohens_d",
    "Absolute_Cohens_d"
]

print(
    feature_summary_df[
        display_columns
    ]
    .head(20)
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 11. Effect-size interpretation
# ============================================================

def effect_size_category(effect_size):

    if pd.isna(effect_size):
        return "Unavailable"

    absolute_effect = abs(effect_size)

    if absolute_effect < 0.2:
        return "Negligible"

    if absolute_effect < 0.5:
        return "Small"

    if absolute_effect < 0.8:
        return "Medium"

    return "Large"


feature_summary_df[
    "Effect_Size_Category"
] = feature_summary_df[
    "Cohens_d"
].apply(
    effect_size_category
)

print("\nEffect-size categories:")

print(
    feature_summary_df[
        "Effect_Size_Category"
    ].value_counts()
)

feature_summary_df.to_csv(
    output_folder
    / "class_feature_comparison.csv",
    index=False
)


# ============================================================
# 12. Plot 1: Top effect sizes
# ============================================================

top_effect_features = (
    feature_summary_df
    .dropna(
        subset=["Cohens_d"]
    )
    .head(12)
    .sort_values(
        "Absolute_Cohens_d",
        ascending=True
    )
)

plt.figure(
    figsize=(11, 8)
)

bars = plt.barh(
    top_effect_features["Feature"],
    top_effect_features["Cohens_d"]
)

plt.axvline(
    x=0,
    linewidth=1
)

plt.title(
    "Top Behavioral Feature Differences: "
    "Depression vs Control"
)

plt.xlabel(
    "Cohen's d\n"
    "Positive = Higher in Depression, "
    "Negative = Higher in Control"
)

plt.ylabel("Feature")

for bar, effect_size in zip(
    bars,
    top_effect_features["Cohens_d"]
):

    horizontal_alignment = (
        "left"
        if effect_size >= 0
        else "right"
    )

    text_offset = (
        0.01
        if effect_size >= 0
        else -0.01
    )

    plt.text(
        effect_size + text_offset,
        bar.get_y()
        + bar.get_height() / 2,
        f"{effect_size:.2f}",
        va="center",
        ha=horizontal_alignment
    )

plt.tight_layout()

plt.savefig(
    output_folder
    / "01_top_feature_effect_sizes.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 13. Create boxplots for the top six features
# ============================================================

top_six_features = (
    feature_summary_df
    .dropna(
        subset=["Absolute_Cohens_d"]
    )
    .head(6)["Feature"]
    .tolist()
)

for feature_number, feature in enumerate(
    top_six_features,
    start=1
):

    control_values = participant_features_df.loc[
        participant_features_df["Class"]
        == "Control",
        feature
    ].dropna()

    depression_values = participant_features_df.loc[
        participant_features_df["Class"]
        == "Depression",
        feature
    ].dropna()

    effect_row = feature_summary_df[
        feature_summary_df["Feature"]
        == feature
    ].iloc[0]

    effect_size = effect_row["Cohens_d"]

    plt.figure(
        figsize=(8, 6)
    )

    plt.boxplot(
        [
            control_values,
            depression_values
        ],
        tick_labels=[
            "Control",
            "Depression"
        ],
        showfliers=True
    )

    plt.title(
        f"{feature}\n"
        f"Cohen's d = {effect_size:.2f}"
    )

    plt.xlabel("Class")
    plt.ylabel("Participant-Level Feature Mean")
    plt.tight_layout()

    safe_feature_name = (
        feature
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )

    plt.savefig(
        output_folder
        / (
            f"{feature_number + 1:02d}_"
            f"{safe_feature_name}_boxplot.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


# ============================================================
# 14. Compare PHQ-8 score with features
# ============================================================

correlation_records = []

for feature in usable_feature_columns:

    correlation_data = participant_features_df[
        [
            "PHQ8_Score",
            feature
        ]
    ].dropna()

    if len(correlation_data) < 3:
        continue

    correlation = correlation_data[
        "PHQ8_Score"
    ].corr(
        correlation_data[feature],
        method="spearman"
    )

    correlation_records.append(
        {
            "Feature": feature,
            "Spearman_Correlation_with_PHQ8":
                correlation,
            "Absolute_Correlation": (
                abs(correlation)
                if pd.notna(correlation)
                else np.nan
            )
        }
    )


correlation_df = pd.DataFrame(
    correlation_records
)

correlation_df = (
    correlation_df
    .sort_values(
        "Absolute_Correlation",
        ascending=False
    )
    .reset_index(drop=True)
)

correlation_df.to_csv(
    output_folder
    / "feature_phq8_correlations.csv",
    index=False
)

print("\n" + "=" * 85)
print("TOP SPEARMAN CORRELATIONS WITH PHQ-8 SCORE")
print("=" * 85)

print(
    correlation_df
    .head(20)
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 15. Plot 8: Top PHQ-8 correlations
# ============================================================

top_correlations = (
    correlation_df
    .dropna(
        subset=[
            "Spearman_Correlation_with_PHQ8"
        ]
    )
    .head(12)
    .sort_values(
        "Absolute_Correlation",
        ascending=True
    )
)

plt.figure(
    figsize=(11, 8)
)

bars = plt.barh(
    top_correlations["Feature"],
    top_correlations[
        "Spearman_Correlation_with_PHQ8"
    ]
)

plt.axvline(
    x=0,
    linewidth=1
)

plt.title(
    "Top Behavioral Feature Correlations "
    "with PHQ-8 Score"
)

plt.xlabel(
    "Spearman Correlation\n"
    "Positive = Increases with PHQ-8 Score"
)

plt.ylabel("Feature")

for bar, correlation in zip(
    bars,
    top_correlations[
        "Spearman_Correlation_with_PHQ8"
    ]
):

    horizontal_alignment = (
        "left"
        if correlation >= 0
        else "right"
    )

    text_offset = (
        0.005
        if correlation >= 0
        else -0.005
    )

    plt.text(
        correlation + text_offset,
        bar.get_y()
        + bar.get_height() / 2,
        f"{correlation:.2f}",
        va="center",
        ha=horizontal_alignment
    )

plt.tight_layout()

plt.savefig(
    output_folder
    / "08_top_phq8_feature_correlations.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 16. Final message
# ============================================================

print("\nAnalysis completed successfully.")

print("\nOutputs were saved in:")
print(output_folder)