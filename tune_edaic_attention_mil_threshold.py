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

OUTPUT_FOLDER = Path(
    r"D:\Flutterr\multimodal-depression-from-video"
    r"\model_outputs\edaic_attention_mil"
)

DEV_PREDICTIONS_PATH = (
    OUTPUT_FOLDER
    / "final_dev_participant_predictions.csv"
)

TEST_PREDICTIONS_PATH = (
    OUTPUT_FOLDER
    / "final_test_participant_predictions.csv"
)

THRESHOLD_RESULTS_PATH = (
    OUTPUT_FOLDER
    / "threshold_tuning_results.csv"
)

BEST_THRESHOLD_PATH = (
    OUTPUT_FOLDER
    / "best_threshold.json"
)

TUNED_DEV_PREDICTIONS_PATH = (
    OUTPUT_FOLDER
    / "tuned_dev_participant_predictions.csv"
)

TUNED_TEST_PREDICTIONS_PATH = (
    OUTPUT_FOLDER
    / "tuned_test_participant_predictions.csv"
)

TUNED_TEST_REPORT_PATH = (
    OUTPUT_FOLDER
    / "tuned_test_classification_report.txt"
)

THRESHOLD_PLOT_PATH = (
    OUTPUT_FOLDER
    / "threshold_tuning_plot.png"
)

TUNED_TEST_CONFUSION_MATRIX_PATH = (
    OUTPUT_FOLDER
    / "tuned_test_confusion_matrix.png"
)


# ============================================================
# 2. Load predictions
# ============================================================

for file_path in [
    DEV_PREDICTIONS_PATH,
    TEST_PREDICTIONS_PATH,
]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required prediction file was not found:\n{file_path}"
        )


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
            f"{dataframe_name} predictions are missing columns: "
            f"{missing_columns}"
        )

    dataframe["True_Label"] = pd.to_numeric(
        dataframe["True_Label"],
        errors="raise",
    ).astype(int)

    dataframe["Depression_Probability"] = pd.to_numeric(
        dataframe["Depression_Probability"],
        errors="raise",
    )


print("=" * 72)
print("E-DAIC-WOZ ATTENTION MIL THRESHOLD TUNING")
print("=" * 72)

print(f"\nDev participants:  {len(dev_df)}")
print(f"Test participants: {len(test_df)}")


# ============================================================
# 3. Metrics
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
# 4. Search thresholds on Dev only
# ============================================================

threshold_values = np.arange(
    0.05,
    0.951,
    0.01,
)

records = []

for threshold in threshold_values:
    predictions = (
        dev_df["Depression_Probability"]
        >= threshold
    ).astype(int)

    metrics = calculate_metrics(
        dev_df["True_Label"],
        predictions,
    )

    combined_score = (
        metrics["f1"]
        + metrics["balanced_accuracy"]
    ) / 2

    records.append(
        {
            "threshold":
                float(threshold),

            **metrics,

            "combined_score":
                float(combined_score),

            "predicted_control":
                int(
                    (predictions == 0).sum()
                ),

            "predicted_depression":
                int(
                    (predictions == 1).sum()
                ),
        }
    )


results_df = pd.DataFrame(
    records
)


# ============================================================
# 5. Select best thresholds
# ============================================================

best_f1_row = (
    results_df
    .sort_values(
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
    )
    .iloc[0]
)

best_combined_row = (
    results_df
    .sort_values(
        by=[
            "combined_score",
            "f1",
            "balanced_accuracy",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )
    .iloc[0]
)


best_f1_threshold = float(
    best_f1_row["threshold"]
)

best_combined_threshold = float(
    best_combined_row["threshold"]
)


print("\nBest threshold by Dev F1:")
print(f"{best_f1_threshold:.2f}")

print("\nDev metrics at best-F1 threshold:")
for metric_name in [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
]:
    print(
        f"{metric_name}: "
        f"{best_f1_row[metric_name]:.4f}"
    )


print("\nBest threshold by combined Dev score:")
print(f"{best_combined_threshold:.2f}")

print("\nDev metrics at best-combined threshold:")
for metric_name in [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "combined_score",
]:
    print(
        f"{metric_name}: "
        f"{best_combined_row[metric_name]:.4f}"
    )


# ============================================================
# 6. Choose final threshold
# ============================================================

# Primary choice:
# maximize the average of F1 and balanced accuracy.
selected_threshold = best_combined_threshold

selected_row = best_combined_row


# ============================================================
# 7. Compare with default threshold
# ============================================================

default_threshold = 0.50

default_dev_predictions = (
    dev_df["Depression_Probability"]
    >= default_threshold
).astype(int)

default_dev_metrics = calculate_metrics(
    dev_df["True_Label"],
    default_dev_predictions,
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
# 8. Apply selected threshold
# ============================================================

tuned_dev_df = dev_df.copy()

tuned_dev_df["Predicted_Label"] = (
    tuned_dev_df["Depression_Probability"]
    >= selected_threshold
).astype(int)

tuned_test_df = test_df.copy()

tuned_test_df["Predicted_Label"] = (
    tuned_test_df["Depression_Probability"]
    >= selected_threshold
).astype(int)


tuned_dev_df.to_csv(
    TUNED_DEV_PREDICTIONS_PATH,
    index=False,
)

tuned_test_df.to_csv(
    TUNED_TEST_PREDICTIONS_PATH,
    index=False,
)


test_metrics = calculate_metrics(
    tuned_test_df["True_Label"],
    tuned_test_df["Predicted_Label"],
)


print("\n" + "=" * 72)
print("TEST RESULTS USING DEV-SELECTED THRESHOLD")
print("=" * 72)

print(
    f"\nApplied threshold: "
    f"{selected_threshold:.2f}"
)

for metric_name, metric_value in (
    test_metrics.items()
):
    print(
        f"{metric_name}: "
        f"{metric_value:.4f}"
    )


# ============================================================
# 9. Classification report and confusion matrix
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

print("\nClassification report:")
print(test_report)

TUNED_TEST_REPORT_PATH.write_text(
    test_report,
    encoding="utf-8",
)


matrix = confusion_matrix(
    tuned_test_df["True_Label"],
    tuned_test_df["Predicted_Label"],
    labels=[0, 1],
)

print("\nConfusion matrix:")
print(matrix)


# ============================================================
# 10. Save results
# ============================================================

results_df.to_csv(
    THRESHOLD_RESULTS_PATH,
    index=False,
)


best_threshold_information = {
    "selection_dataset":
        "Dev",

    "selection_rule":
        "Average of participant F1 and balanced accuracy",

    "selected_threshold":
        selected_threshold,

    "best_f1_threshold":
        best_f1_threshold,

    "best_combined_threshold":
        best_combined_threshold,

    "selected_dev_metrics": {
        metric_name:
            float(selected_row[metric_name])

        for metric_name in [
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "combined_score",
        ]
    },

    "default_threshold":
        default_threshold,

    "default_dev_metrics": {
        metric_name:
            float(metric_value)

        for metric_name, metric_value
        in default_dev_metrics.items()
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
# 11. Threshold plot
# ============================================================

figure, axis = plt.subplots(
    figsize=(9, 6)
)

axis.plot(
    results_df["threshold"],
    results_df["f1"],
    label="Dev F1",
)

axis.plot(
    results_df["threshold"],
    results_df["balanced_accuracy"],
    label="Dev balanced accuracy",
)

axis.plot(
    results_df["threshold"],
    results_df["combined_score"],
    label="Combined score",
)

axis.plot(
    results_df["threshold"],
    results_df["recall"],
    label="Dev recall",
)

axis.axvline(
    x=selected_threshold,
    linestyle="--",
    label=(
        f"Selected threshold = "
        f"{selected_threshold:.2f}"
    ),
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
    "Attention MIL Dev Threshold Tuning"
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
# 12. Test confusion matrix plot
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
    "Attention MIL Tuned Test Confusion Matrix\n"
    f"Threshold = {selected_threshold:.2f}"
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
# 13. Final summary
# ============================================================

print("\nBaseline comparison:")
print("Baseline LSTM F1: 0.4444")
print("Baseline LSTM balanced accuracy: 0.5943")

print("\nFiles saved in:")
print(OUTPUT_FOLDER)