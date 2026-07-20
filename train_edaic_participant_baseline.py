from __future__ import annotations

from pathlib import Path
import json
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================================
# 1. Configuration
# ============================================================

SEED = 42

EXPECTED_WINDOW_FRAMES = 270
EXPECTED_FEATURES = 28

# The model is selected using Dev participant F1.
PRIMARY_SELECTION_METRIC = "f1"

warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning,
)


# ============================================================
# 2. Paths
# ============================================================

WINDOWS_FOLDER = Path(
    r"D:\Downloads\Depression_Anxiety_Body_Movement\data\processed"
    r"\e_daic_woz_visual_windows"
)

TRAIN_METADATA_PATH = (
    WINDOWS_FOLDER
    / "train_windows_metadata.csv"
)

DEV_METADATA_PATH = (
    WINDOWS_FOLDER
    / "dev_windows_metadata.csv"
)

TEST_METADATA_PATH = (
    WINDOWS_FOLDER
    / "test_windows_metadata.csv"
)

OUTPUT_FOLDER = Path(
    r"D:\Flutterr\multimodal-depression-from-video"
    r"\model_outputs\edaic_participant_baseline"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

FEATURES_FOLDER = (
    OUTPUT_FOLDER
    / "participant_features"
)

FEATURES_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

TRAIN_FEATURES_PATH = (
    FEATURES_FOLDER
    / "train_participant_features.csv"
)

DEV_FEATURES_PATH = (
    FEATURES_FOLDER
    / "dev_participant_features.csv"
)

TEST_FEATURES_PATH = (
    FEATURES_FOLDER
    / "test_participant_features.csv"
)

MODEL_COMPARISON_PATH = (
    OUTPUT_FOLDER
    / "dev_model_comparison.csv"
)

FINAL_RESULTS_PATH = (
    OUTPUT_FOLDER
    / "final_results.json"
)

TEST_PREDICTIONS_PATH = (
    OUTPUT_FOLDER
    / "test_participant_predictions.csv"
)

TEST_REPORT_PATH = (
    OUTPUT_FOLDER
    / "test_classification_report.txt"
)

CONFUSION_MATRIX_PATH = (
    OUTPUT_FOLDER
    / "test_confusion_matrix.png"
)


# ============================================================
# 3. Read metadata
# ============================================================

def read_metadata(
    metadata_path: Path,
) -> pd.DataFrame:

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file was not found:\n{metadata_path}"
        )

    dataframe = pd.read_csv(
        metadata_path
    )

    required_columns = [
        "Participant_ID",
        "Split",
        "Label",
        "Number_of_Windows",
        "Number_of_Features",
        "Windows_File",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{metadata_path.name} is missing columns: "
            f"{missing_columns}"
        )

    dataframe["Participant_ID"] = pd.to_numeric(
        dataframe["Participant_ID"],
        errors="raise",
    ).astype(int)

    dataframe["Label"] = pd.to_numeric(
        dataframe["Label"],
        errors="raise",
    ).astype(int)

    dataframe["Number_of_Windows"] = pd.to_numeric(
        dataframe["Number_of_Windows"],
        errors="raise",
    ).astype(int)

    return dataframe


train_metadata_df = read_metadata(
    TRAIN_METADATA_PATH
)

dev_metadata_df = read_metadata(
    DEV_METADATA_PATH
)

test_metadata_df = read_metadata(
    TEST_METADATA_PATH
)


print("=" * 72)
print("E-DAIC-WOZ PARTICIPANT-LEVEL STATISTICAL BASELINE")
print("=" * 72)

print("\nParticipants:")

print(
    f"Train: {len(train_metadata_df)}"
)

print(
    f"Dev:   {len(dev_metadata_df)}"
)

print(
    f"Test:  {len(test_metadata_df)}"
)


# ============================================================
# 4. Feature extraction
# ============================================================

STATISTIC_NAMES = [
    "mean",
    "std",
    "minimum",
    "p10",
    "p25",
    "median",
    "p75",
    "p90",
    "maximum",
]

WINDOW_STATISTIC_NAMES = [
    "window_mean_mean",
    "window_mean_std",
    "window_std_mean",
    "window_std_std",
]


def calculate_participant_features(
    windows: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """
    Convert all windows belonging to one participant into
    one fixed-length statistical feature vector.

    Input shape:
        number_of_windows × 270 × 28
    """

    if windows.ndim != 3:
        raise ValueError(
            f"Expected a 3D window array, found {windows.shape}"
        )

    if windows.shape[1:] != (
        EXPECTED_WINDOW_FRAMES,
        EXPECTED_FEATURES,
    ):
        raise ValueError(
            "Unexpected window shape. "
            f"Expected (*, {EXPECTED_WINDOW_FRAMES}, "
            f"{EXPECTED_FEATURES}), found {windows.shape}"
        )

    windows = np.asarray(
        windows,
        dtype=np.float32,
    )

    flattened_frames = windows.reshape(
        -1,
        EXPECTED_FEATURES,
    )

    frame_statistics = [
        np.mean(
            flattened_frames,
            axis=0,
        ),

        np.std(
            flattened_frames,
            axis=0,
        ),

        np.min(
            flattened_frames,
            axis=0,
        ),

        np.percentile(
            flattened_frames,
            10,
            axis=0,
        ),

        np.percentile(
            flattened_frames,
            25,
            axis=0,
        ),

        np.percentile(
            flattened_frames,
            50,
            axis=0,
        ),

        np.percentile(
            flattened_frames,
            75,
            axis=0,
        ),

        np.percentile(
            flattened_frames,
            90,
            axis=0,
        ),

        np.max(
            flattened_frames,
            axis=0,
        ),
    ]

    # Describe how each feature changes between windows.
    window_means = np.mean(
        windows,
        axis=1,
    )

    window_stds = np.std(
        windows,
        axis=1,
    )

    window_statistics = [
        np.mean(
            window_means,
            axis=0,
        ),

        np.std(
            window_means,
            axis=0,
        ),

        np.mean(
            window_stds,
            axis=0,
        ),

        np.std(
            window_stds,
            axis=0,
        ),
    ]

    feature_vector = np.concatenate(
        frame_statistics
        + window_statistics,
        axis=0,
    ).astype(
        np.float32
    )

    feature_names = []

    for statistic_name in STATISTIC_NAMES:
        for feature_index in range(
            EXPECTED_FEATURES
        ):
            feature_names.append(
                f"feature_{feature_index:02d}_{statistic_name}"
            )

    for statistic_name in WINDOW_STATISTIC_NAMES:
        for feature_index in range(
            EXPECTED_FEATURES
        ):
            feature_names.append(
                f"feature_{feature_index:02d}_{statistic_name}"
            )

    return (
        feature_vector,
        feature_names,
    )


def extract_split_features(
    metadata_df: pd.DataFrame,
    split_name: str,
    output_path: Path,
) -> pd.DataFrame:

    records = []
    expected_feature_names = None

    print(
        f"\nExtracting {split_name} participant features..."
    )

    for participant_number, row in enumerate(
        metadata_df.itertuples(index=False),
        start=1,
    ):
        participant_id = int(
            row.Participant_ID
        )

        label = int(
            row.Label
        )

        windows_file = Path(
            str(row.Windows_File)
        )

        if not windows_file.exists():
            raise FileNotFoundError(
                f"Windows file was not found:\n{windows_file}"
            )

        windows = np.load(
            windows_file,
            mmap_mode="r",
        )

        (
            feature_vector,
            feature_names,
        ) = calculate_participant_features(
            windows
        )

        if expected_feature_names is None:
            expected_feature_names = feature_names

        elif feature_names != expected_feature_names:
            raise ValueError(
                "Feature names changed between participants."
            )

        record = {
            "Participant_ID":
                participant_id,

            "Label":
                label,

            "Number_of_Windows":
                int(windows.shape[0]),
        }

        for feature_name, feature_value in zip(
            feature_names,
            feature_vector,
        ):
            record[feature_name] = float(
                feature_value
            )

        records.append(
            record
        )

        if (
            participant_number % 20 == 0
            or participant_number == len(metadata_df)
        ):
            print(
                f"{participant_number}/{len(metadata_df)} "
                f"participants completed"
            )

    result_df = pd.DataFrame(
        records
    )

    result_df.to_csv(
        output_path,
        index=False,
    )

    return result_df


# Extracting features is fast, but reuse the CSVs if they already exist.
if (
    TRAIN_FEATURES_PATH.exists()
    and DEV_FEATURES_PATH.exists()
    and TEST_FEATURES_PATH.exists()
):
    print(
        "\nExisting participant feature files found."
    )

    print(
        "Loading saved features instead of extracting again."
    )

    train_features_df = pd.read_csv(
        TRAIN_FEATURES_PATH
    )

    dev_features_df = pd.read_csv(
        DEV_FEATURES_PATH
    )

    test_features_df = pd.read_csv(
        TEST_FEATURES_PATH
    )

else:
    extraction_start_time = time.time()

    train_features_df = extract_split_features(
        train_metadata_df,
        "Train",
        TRAIN_FEATURES_PATH,
    )

    dev_features_df = extract_split_features(
        dev_metadata_df,
        "Dev",
        DEV_FEATURES_PATH,
    )

    test_features_df = extract_split_features(
        test_metadata_df,
        "Test",
        TEST_FEATURES_PATH,
    )

    extraction_time = (
        time.time()
        - extraction_start_time
    )

    print(
        f"\nFeature extraction time: "
        f"{extraction_time:.1f} seconds"
    )


# ============================================================
# 5. Prepare matrices
# ============================================================

excluded_columns = [
    "Participant_ID",
    "Label",
    "Number_of_Windows",
]

feature_columns = [
    column
    for column in train_features_df.columns
    if column not in excluded_columns
]

print(
    f"\nNumber of statistical features: "
    f"{len(feature_columns)}"
)


def prepare_xy(
    dataframe: pd.DataFrame,
):
    features = (
        dataframe[
            feature_columns
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
        .to_numpy(
            dtype=np.float32
        )
    )

    labels = (
        dataframe["Label"]
        .to_numpy(
            dtype=np.int64
        )
    )

    return (
        features,
        labels,
    )


X_train, y_train = prepare_xy(
    train_features_df
)

X_dev, y_dev = prepare_xy(
    dev_features_df
)

X_test, y_test = prepare_xy(
    test_features_df
)


# ============================================================
# 6. Metrics
# ============================================================

def calculate_metrics(
    true_labels,
    predicted_labels,
) -> dict[str, float]:

    return {
        "accuracy":
            accuracy_score(
                true_labels,
                predicted_labels,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                true_labels,
                predicted_labels,
            ),

        "precision":
            precision_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            ),

        "recall":
            recall_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            ),

        "f1":
            f1_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            ),
    }


# ============================================================
# 7. Candidate models
# ============================================================

candidate_models = {
    "Logistic Regression C=0.01":
        Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),

                (
                    "classifier",
                    LogisticRegression(
                        C=0.01,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=SEED,
                    ),
                ),
            ]
        ),

    "Logistic Regression C=0.1":
        Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),

                (
                    "classifier",
                    LogisticRegression(
                        C=0.1,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=SEED,
                    ),
                ),
            ]
        ),

    "Logistic Regression C=1.0":
        Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),

                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=SEED,
                    ),
                ),
            ]
        ),

    "SVM Linear C=0.01":
        Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),

                (
                    "classifier",
                    SVC(
                        C=0.01,
                        kernel="linear",
                        class_weight="balanced",
                        probability=True,
                        random_state=SEED,
                    ),
                ),
            ]
        ),

    "SVM Linear C=0.1":
        Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),

                (
                    "classifier",
                    SVC(
                        C=0.1,
                        kernel="linear",
                        class_weight="balanced",
                        probability=True,
                        random_state=SEED,
                    ),
                ),
            ]
        ),

    "SVM RBF C=1.0":
        Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),

                (
                    "classifier",
                    SVC(
                        C=1.0,
                        kernel="rbf",
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=SEED,
                    ),
                ),
            ]
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),

    "Extra Trees":
        ExtraTreesClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
}


# ============================================================
# 8. Select model using Dev only
# ============================================================

comparison_records = []
trained_models = {}

print("\n" + "=" * 72)
print("MODEL SELECTION USING DEV SET")
print("=" * 72)


for model_name, model in candidate_models.items():
    model_start_time = time.time()

    model.fit(
        X_train,
        y_train,
    )

    dev_predictions = model.predict(
        X_dev
    )

    dev_metrics = calculate_metrics(
        y_dev,
        dev_predictions,
    )

    elapsed_time = (
        time.time()
        - model_start_time
    )

    trained_models[
        model_name
    ] = model

    comparison_records.append(
        {
            "Model":
                model_name,

            **dev_metrics,

            "training_time_seconds":
                elapsed_time,
        }
    )

    print(
        f"\n{model_name}"
    )

    print(
        f"Dev F1: "
        f"{dev_metrics['f1']:.4f}"
    )

    print(
        f"Dev recall: "
        f"{dev_metrics['recall']:.4f}"
    )

    print(
        f"Dev balanced accuracy: "
        f"{dev_metrics['balanced_accuracy']:.4f}"
    )


comparison_df = pd.DataFrame(
    comparison_records
)

comparison_df = comparison_df.sort_values(
    by=[
        "f1",
        "balanced_accuracy",
        "recall",
    ],
    ascending=[
        False,
        False,
        False,
    ],
).reset_index(
    drop=True
)

comparison_df.to_csv(
    MODEL_COMPARISON_PATH,
    index=False,
)


best_model_name = str(
    comparison_df.iloc[0]["Model"]
)

best_model = trained_models[
    best_model_name
]


print("\n" + "=" * 72)
print("BEST MODEL SELECTED FROM DEV")
print("=" * 72)

print(
    f"\nBest model: {best_model_name}"
)

print(
    f"Dev F1: "
    f"{comparison_df.iloc[0]['f1']:.4f}"
)

print(
    f"Dev balanced accuracy: "
    f"{comparison_df.iloc[0]['balanced_accuracy']:.4f}"
)

print(
    f"Dev recall: "
    f"{comparison_df.iloc[0]['recall']:.4f}"
)


# ============================================================
# 9. Final Test evaluation
# ============================================================

# The Test set is evaluated only for the model selected using Dev.
test_predictions = best_model.predict(
    X_test
)

if hasattr(
    best_model,
    "predict_proba",
):
    test_probabilities = best_model.predict_proba(
        X_test
    )[:, 1]

else:
    test_probabilities = np.full(
        shape=len(X_test),
        fill_value=np.nan,
    )


test_metrics = calculate_metrics(
    y_test,
    test_predictions,
)


test_predictions_df = pd.DataFrame(
    {
        "Participant_ID":
            test_features_df[
                "Participant_ID"
            ].astype(int),

        "True_Label":
            y_test,

        "Predicted_Label":
            test_predictions,

        "Depression_Probability":
            test_probabilities,

        "Number_of_Windows":
            test_features_df[
                "Number_of_Windows"
            ].astype(int),
    }
)

test_predictions_df.to_csv(
    TEST_PREDICTIONS_PATH,
    index=False,
)


# ============================================================
# 10. Reports
# ============================================================

test_report = classification_report(
    y_test,
    test_predictions,
    target_names=[
        "Control",
        "Depression",
    ],
    digits=4,
    zero_division=0,
)

TEST_REPORT_PATH.write_text(
    test_report,
    encoding="utf-8",
)


matrix = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1],
)


# ============================================================
# 11. Confusion matrix plot
# ============================================================

figure, axis = plt.subplots(
    figsize=(6, 5)
)

image = axis.imshow(
    matrix
)

figure.colorbar(
    image,
    ax=axis,
)

axis.set_title(
    f"Participant-Level Test Confusion Matrix\n"
    f"{best_model_name}"
)

axis.set_xlabel(
    "Predicted label"
)

axis.set_ylabel(
    "True label"
)

axis.set_xticks(
    [0, 1],
    labels=[
        "Control",
        "Depression",
    ],
)

axis.set_yticks(
    [0, 1],
    labels=[
        "Control",
        "Depression",
    ],
)

for row_index in range(2):
    for column_index in range(2):
        axis.text(
            column_index,
            row_index,
            str(
                matrix[
                    row_index,
                    column_index
                ]
            ),
            ha="center",
            va="center",
        )

figure.tight_layout()

figure.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    figure
)


# ============================================================
# 12. Save final results
# ============================================================

final_results = {
    "model_selection_dataset":
        "Dev",

    "model_selection_metric":
        "Participant-level F1",

    "best_model":
        best_model_name,

    "number_of_features":
        len(feature_columns),

    "train_participants":
        int(len(X_train)),

    "dev_participants":
        int(len(X_dev)),

    "test_participants":
        int(len(X_test)),

    "best_dev_metrics": {
        "accuracy":
            float(
                comparison_df.iloc[0][
                    "accuracy"
                ]
            ),

        "balanced_accuracy":
            float(
                comparison_df.iloc[0][
                    "balanced_accuracy"
                ]
            ),

        "precision":
            float(
                comparison_df.iloc[0][
                    "precision"
                ]
            ),

        "recall":
            float(
                comparison_df.iloc[0][
                    "recall"
                ]
            ),

        "f1":
            float(
                comparison_df.iloc[0][
                    "f1"
                ]
            ),
    },

    "test_metrics": {
        metric_name:
            float(metric_value)

        for metric_name, metric_value
        in test_metrics.items()
    },

    "test_confusion_matrix":
        matrix.tolist(),
}

with open(
    FINAL_RESULTS_PATH,
    "w",
    encoding="utf-8",
) as output_file:
    json.dump(
        final_results,
        output_file,
        indent=4,
    )


# ============================================================
# 13. Final output
# ============================================================

print("\n" + "=" * 72)
print("FINAL TEST RESULTS")
print("=" * 72)

print(
    f"\nSelected model: {best_model_name}"
)

print("\nParticipant-level Test metrics:")

for metric_name, metric_value in (
    test_metrics.items()
):
    print(
        f"{metric_name}: "
        f"{metric_value:.4f}"
    )

print("\nClassification report:")

print(
    test_report
)

print("\nConfusion matrix:")

print(
    matrix
)

print("\nBaseline to beat:")

print(
    "LSTM participant F1: 0.4444"
)

print(
    "LSTM balanced accuracy: 0.5943"
)

print("\nOutputs saved in:")

print(
    OUTPUT_FOLDER
)