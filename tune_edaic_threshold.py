from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# 1. Paths
# ============================================================

MODEL_OUTPUT_FOLDER = Path(
    r"D:\Flutterr\multimodal-depression-from-video"
    r"\model_outputs\edaic_lstm_baseline"
)

DEV_PREDICTIONS_PATH = (
    MODEL_OUTPUT_FOLDER
    / "final_dev_participant_predictions.csv"
)

TEST_PREDICTIONS_PATH = (
    MODEL_OUTPUT_FOLDER
    / "final_test_participant_predictions.csv"
)

THRESHOLD_RESULTS_PATH = (
    MODEL_OUTPUT_FOLDER
    / "threshold_tuning_results.csv"
)

BEST_THRESHOLD_PATH = (
    MODEL_OUTPUT_FOLDER
    / "best_threshold.json"
)

TUNED_DEV_PREDICTIONS_PATH = (
    MODEL_OUTPUT_FOLDER
    / "tuned_dev_participant_predictions.csv"
)

TUNED_TEST_PREDICTIONS_PATH = (
    MODEL_OUTPUT_FOLDER
    / "tuned_test_participant_predictions.csv"
)

TUNED_TEST_REPORT_PATH = (
    MODEL_OUTPUT_FOLDER
    / "tuned_test_classification_report.txt"
)

THRESHOLD_PLOT_PATH = (
    MODEL_OUTPUT_FOLDER
    / "threshold_tuning_plot.png"
)

TUNED_TEST_CONFUSION_MATRIX_PATH = (
    MODEL_OUTPUT_FOLDER
    / "tuned_test_confusion_matrix.png"
)


# ============================================================
# 2. Check required files
# ============================================================

required_files = {
    "Dev predictions": DEV_PREDICTIONS_PATH,
    "Test predictions": TEST_PREDICTIONS_PATH,
}

for file_name, file_path in required_files.items():

    if not file_path.exists():

        raise FileNotFoundError(
            f"{file_name} was not found:\n"
            f"{file_path}"
        )


# ============================================================
# 3. Read predictions
# ============================================================

dev_df = pd.read_csv(
    DEV_PREDICTIONS_PATH
)

test_df = pd.read_csv(
    TEST_PREDICTIONS_PATH
)


required_columns = [
    "Participant_ID",
    "True_Label",
    "Depression_Probability",
    "Number_of_Windows",
]


for dataframe_name, dataframe in [
    ("Dev", dev_df),
    ("Test", test_df),
]:

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:

        raise ValueError(
            f"{dataframe_name} predictions are missing "
            f"columns: {missing_columns}"
        )


    dataframe["True_Label"] = pd.to_numeric(
        dataframe["True_Label"],
        errors="raise",
    ).astype(int)

    dataframe["Depression_Probability"] = pd.to_numeric(
        dataframe["Depression_Probability"],
        errors="raise",
    )


print("=" * 70)
print("E-DAIC-WOZ PARTICIPANT THRESHOLD TUNING")
print("=" * 70)

print("\nDev participants:")
print(len(dev_df))

print("\nTest participants:")
print(len(test_df))


# ============================================================
# 4. Metric function
# ============================================================

def calculate_metrics(
    true_labels,
    predicted_labels,
) -> dict[str, float]:
    """
    Calculate participant-level binary classification metrics.
    """

    return {
        "Accuracy":
            accuracy_score(
                true_labels,
                predicted_labels,
            ),

        "Balanced_Accuracy":
            balanced_accuracy_score(
                true_labels,
                predicted_labels,
            ),

        "Precision":
            precision_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            ),

        "Recall":
            recall_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            ),

        "F1":
            f1_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            ),
    }


# ============================================================
# 5. Test thresholds on Dev only
# ============================================================

# Test thresholds from 0.05 to 0.95.
threshold_values = np.arange(
    0.05,
    0.951,
    0.01,
)

threshold_records = []


for threshold in threshold_values:

    dev_predictions = (
        dev_df["Depression_Probability"]
        >= threshold
    ).astype(int)

    metrics = calculate_metrics(
        true_labels=dev_df["True_Label"],
        predicted_labels=dev_predictions,
    )

    threshold_records.append(
        {
            "Threshold":
                float(threshold),

            **metrics,

            "Predicted_Control":
                int(
                    (dev_predictions == 0).sum()
                ),

            "Predicted_Depression":
                int(
                    (dev_predictions == 1).sum()
                ),
        }
    )


threshold_results_df = pd.DataFrame(
    threshold_records
)


# ============================================================
# 6. Select best threshold
# ============================================================

# Primary metric: highest F1.
# Tie-breaker: highest balanced accuracy.
best_row = (
    threshold_results_df
    .sort_values(
        by=[
            "F1",
            "Balanced_Accuracy",
            "Recall",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )
    .iloc[0]
)


best_threshold = float(
    best_row["Threshold"]
)


print("\nBest threshold selected from Dev:")
print(
    f"{best_threshold:.2f}"
)

print("\nBest Dev metrics:")

for metric_name in [
    "Accuracy",
    "Balanced_Accuracy",
    "Precision",
    "Recall",
    "F1",
]:

    print(
        f"{metric_name}: "
        f"{best_row[metric_name]:.4f}"
    )


# ============================================================
# 7. Compare against default threshold 0.50
# ============================================================

default_threshold = 0.50

default_dev_predictions = (
    dev_df["Depression_Probability"]
    >= default_threshold
).astype(int)

default_dev_metrics = calculate_metrics(
    true_labels=dev_df["True_Label"],
    predicted_labels=default_dev_predictions,
)


print("\nDefault threshold 0.50 Dev metrics:")

for metric_name, metric_value in (
    default_dev_metrics.items()
):

    print(
        f"{metric_name}: "
        f"{metric_value:.4f}"
    )


# ============================================================
# 8. Save complete threshold search
# ============================================================

threshold_results_df.to_csv(
    THRESHOLD_RESULTS_PATH,
    index=False,
)


best_threshold_information = {
    "Selection_Dataset":
        "Dev",

    "Selection_Metric":
        "Participant-level F1",

    "Best_Threshold":
        best_threshold,

    "Best_Dev_Metrics": {
        metric_name:
            float(best_row[metric_name])

        for metric_name in [
            "Accuracy",
            "Balanced_Accuracy",
            "Precision",
            "Recall",
            "F1",
        ]
    },

    "Default_Threshold":
        default_threshold,

    "Default_Dev_Metrics": {
        metric_name:
            float(metric_value)

        for metric_name, metric_value
        in default_dev_metrics.items()
    },
}


with open(
    BEST_THRESHOLD_PATH,
    "w",
    encoding="utf-8",
) as output_file:

    json.dump(
        best_threshold_information,
        output_file,
        indent=4,
    )


# ============================================================
# 9. Apply threshold to Dev
# ============================================================

tuned_dev_df = dev_df.copy()

tuned_dev_df["Predicted_Label"] = (
    tuned_dev_df["Depression_Probability"]
    >= best_threshold
).astype(int)

tuned_dev_df.to_csv(
    TUNED_DEV_PREDICTIONS_PATH,
    index=False,
)


# ============================================================
# 10. Apply the same Dev-selected threshold to Test
# ============================================================

tuned_test_df = test_df.copy()

tuned_test_df["Predicted_Label"] = (
    tuned_test_df["Depression_Probability"]
    >= best_threshold
).astype(int)

tuned_test_df.to_csv(
    TUNED_TEST_PREDICTIONS_PATH,
    index=False,
)


tuned_test_metrics = calculate_metrics(
    true_labels=tuned_test_df["True_Label"],
    predicted_labels=tuned_test_df["Predicted_Label"],
)


print("\n" + "=" * 70)
print("TEST RESULTS USING DEV-SELECTED THRESHOLD")
print("=" * 70)

print(
    f"\nApplied threshold: "
    f"{best_threshold:.2f}"
)

for metric_name, metric_value in (
    tuned_test_metrics.items()
):

    print(
        f"{metric_name}: "
        f"{metric_value:.4f}"
    )


# ============================================================
# 11. Test classification report
# ============================================================

test_report = classification_report(
    tuned_test_df["True_Label"],
    tuned_test_df["Predicted_Label"],

    target_names=[
        "Control",
        "Depression",
    ],

    digits=4,
    zero_division=0,
)

print("\nTest participant classification report:")
print(test_report)

TUNED_TEST_REPORT_PATH.write_text(
    test_report,
    encoding="utf-8",
)


# ============================================================
# 12. Threshold tuning plot
# ============================================================

figure, axis = plt.subplots(
    figsize=(9, 6)
)

axis.plot(
    threshold_results_df["Threshold"],
    threshold_results_df["F1"],
    label="Dev F1",
)

axis.plot(
    threshold_results_df["Threshold"],
    threshold_results_df["Balanced_Accuracy"],
    label="Dev balanced accuracy",
)

axis.plot(
    threshold_results_df["Threshold"],
    threshold_results_df["Recall"],
    label="Dev recall",
)

axis.axvline(
    x=best_threshold,
    linestyle="--",
    label=f"Best threshold = {best_threshold:.2f}",
)

axis.axvline(
    x=0.50,
    linestyle=":",
    label="Default threshold = 0.50",
)

axis.set_xlabel(
    "Depression probability threshold"
)

axis.set_ylabel(
    "Metric value"
)

axis.set_title(
    "Dev Participant-Level Threshold Tuning"
)

axis.legend()

figure.tight_layout()

figure.savefig(
    THRESHOLD_PLOT_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    figure
)


# ============================================================
# 13. Tuned Test confusion matrix
# ============================================================

matrix = confusion_matrix(
    tuned_test_df["True_Label"],
    tuned_test_df["Predicted_Label"],
    labels=[0, 1],
)


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
    "Test Participant-Level Confusion Matrix\n"
    f"Threshold = {best_threshold:.2f}"
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
    TUNED_TEST_CONFUSION_MATRIX_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    figure
)


# ============================================================
# 14. Final summary
# ============================================================

print("\nConfusion matrix:")

print(matrix)

print("\nFiles saved:")

print(
    "\nThreshold search results:"
)
print(THRESHOLD_RESULTS_PATH)

print(
    "\nBest threshold information:"
)
print(BEST_THRESHOLD_PATH)

print(
    "\nTuned Test predictions:"
)
print(TUNED_TEST_PREDICTIONS_PATH)

print(
    "\nThreshold plot:"
)
print(THRESHOLD_PLOT_PATH)

print(
    "\nTuned Test confusion matrix:"
)
print(TUNED_TEST_CONFUSION_MATRIX_PATH)