from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import json
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 1. Experiment configuration
# ============================================================

SEED = 42

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

HIDDEN_SIZE = 128
NUMBER_OF_LSTM_LAYERS = 2
DROPOUT = 0.30

EXPECTED_WINDOW_FRAMES = 270
EXPECTED_FEATURES = 28

EARLY_STOPPING_PATIENCE = 7

# On Windows, starting multiple DataLoader workers may cause issues.
NUM_WORKERS = 0

# Number of participant arrays kept open in memory-mapped cache.
ARRAY_CACHE_SIZE = 8


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
    r"\model_outputs\edaic_lstm_baseline"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_PATH = (
    OUTPUT_FOLDER
    / "best_lstm_model.pt"
)

TRAINING_HISTORY_PATH = (
    OUTPUT_FOLDER
    / "training_history.csv"
)

FINAL_RESULTS_PATH = (
    OUTPUT_FOLDER
    / "final_results.json"
)


# ============================================================
# 3. Reproducibility
# ============================================================

def set_random_seed(seed: int) -> None:
    """
    Set random seeds to make results more reproducible.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_random_seed(SEED)


# ============================================================
# 4. Select computing device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("E-DAIC-WOZ LSTM BASELINE")
print("=" * 70)

print("\nDevice:")
print(DEVICE)

if DEVICE.type == "cuda":
    print("\nGPU:")
    print(torch.cuda.get_device_name(0))


# ============================================================
# 5. Check required metadata files
# ============================================================

required_paths = {
    "Train metadata": TRAIN_METADATA_PATH,
    "Dev metadata": DEV_METADATA_PATH,
    "Test metadata": TEST_METADATA_PATH,
}

for name, path in required_paths.items():
    if not path.exists():
        raise FileNotFoundError(
            f"{name} was not found:\n{path}"
        )


# ============================================================
# 6. Read metadata
# ============================================================

def read_windows_metadata(
    metadata_path: Path,
) -> pd.DataFrame:
    """
    Read and validate one windows metadata file.
    """

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

    dataframe["Number_of_Features"] = pd.to_numeric(
        dataframe["Number_of_Features"],
        errors="raise",
    ).astype(int)

    return dataframe


train_metadata_df = read_windows_metadata(
    TRAIN_METADATA_PATH
)

dev_metadata_df = read_windows_metadata(
    DEV_METADATA_PATH
)

test_metadata_df = read_windows_metadata(
    TEST_METADATA_PATH
)


print("\nParticipant counts:")

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
# 7. Dataset class
# ============================================================

class VisualWindowDataset(Dataset):
    """
    Creates one dataset item for every 9-second window.

    Each item contains:
        features:       shape (270, 28)
        label:          0 or 1
        participant_id
        window_index
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        array_cache_size: int = 8,
    ) -> None:

        self.samples = []

        self.array_cache_size = array_cache_size

        self.array_cache: OrderedDict[
            str,
            np.ndarray
        ] = OrderedDict()

        for row in metadata_df.itertuples(
            index=False
        ):
            participant_id = int(
                row.Participant_ID
            )

            label = int(
                row.Label
            )

            windows_file = str(
                row.Windows_File
            )

            number_of_windows = int(
                row.Number_of_Windows
            )

            path = Path(
                windows_file
            )

            if not path.exists():
                raise FileNotFoundError(
                    f"Windows file was not found: {path}"
                )

            for window_index in range(
                number_of_windows
            ):
                self.samples.append(
                    {
                        "windows_file":
                            windows_file,

                        "window_index":
                            window_index,

                        "label":
                            label,

                        "participant_id":
                            participant_id,
                    }
                )

    def __len__(self) -> int:
        return len(
            self.samples
        )

    def _load_array(
        self,
        windows_file: str,
    ) -> np.ndarray:
        """
        Open a participant NPY using memory mapping.

        A small cache avoids reopening the same participant file
        for every consecutive window.
        """

        if windows_file in self.array_cache:
            array = self.array_cache.pop(
                windows_file
            )

            self.array_cache[
                windows_file
            ] = array

            return array

        array = np.load(
            windows_file,
            mmap_mode="r",
        )

        if array.ndim != 3:
            raise ValueError(
                f"Expected a 3D windows array, "
                f"found {array.shape} in {windows_file}"
            )

        if array.shape[1:] != (
            EXPECTED_WINDOW_FRAMES,
            EXPECTED_FEATURES,
        ):
            raise ValueError(
                f"Expected window shape "
                f"({EXPECTED_WINDOW_FRAMES}, "
                f"{EXPECTED_FEATURES}), "
                f"found {array.shape[1:]} "
                f"in {windows_file}"
            )

        self.array_cache[
            windows_file
        ] = array

        while (
            len(self.array_cache)
            > self.array_cache_size
        ):
            self.array_cache.popitem(
                last=False
            )

        return array

    def __getitem__(
        self,
        index: int,
    ):
        sample = self.samples[
            index
        ]

        windows_array = self._load_array(
            sample["windows_file"]
        )

        window = np.array(
            windows_array[
                sample["window_index"]
            ],
            dtype=np.float32,
            copy=True,
        )

        features_tensor = torch.from_numpy(
            window
        )

        label_tensor = torch.tensor(
            sample["label"],
            dtype=torch.long,
        )

        participant_tensor = torch.tensor(
            sample["participant_id"],
            dtype=torch.long,
        )

        window_index_tensor = torch.tensor(
            sample["window_index"],
            dtype=torch.long,
        )

        return (
            features_tensor,
            label_tensor,
            participant_tensor,
            window_index_tensor,
        )


train_dataset = VisualWindowDataset(
    train_metadata_df,
    array_cache_size=ARRAY_CACHE_SIZE,
)

dev_dataset = VisualWindowDataset(
    dev_metadata_df,
    array_cache_size=ARRAY_CACHE_SIZE,
)

test_dataset = VisualWindowDataset(
    test_metadata_df,
    array_cache_size=ARRAY_CACHE_SIZE,
)


print("\nWindow counts:")

print(
    "Train:",
    len(train_dataset),
)

print(
    "Dev:",
    len(dev_dataset),
)

print(
    "Test:",
    len(test_dataset),
)


# ============================================================
# 8. DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(
        DEVICE.type == "cuda"
    ),
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(
        DEVICE.type == "cuda"
    ),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(
        DEVICE.type == "cuda"
    ),
)


# ============================================================
# 9. LSTM model
# ============================================================

class DepressionLSTM(nn.Module):
    """
    LSTM classifier for visual feature sequences.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        number_of_layers: int,
        dropout: float,
        number_of_classes: int = 2,
    ) -> None:

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=number_of_layers,
            batch_first=True,
            dropout=(
                dropout
                if number_of_layers > 1
                else 0.0
            ),
            bidirectional=False,
        )

        self.normalization = nn.LayerNorm(
            hidden_size
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Linear(
            hidden_size,
            number_of_classes,
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:

        sequence_output, _ = self.lstm(
            features
        )

        # Use the representation of the final frame.
        final_hidden_state = sequence_output[
            :,
            -1,
            :
        ]

        final_hidden_state = self.normalization(
            final_hidden_state
        )

        final_hidden_state = self.dropout(
            final_hidden_state
        )

        logits = self.classifier(
            final_hidden_state
        )

        return logits


model = DepressionLSTM(
    input_size=EXPECTED_FEATURES,
    hidden_size=HIDDEN_SIZE,
    number_of_layers=NUMBER_OF_LSTM_LAYERS,
    dropout=DROPOUT,
).to(
    DEVICE
)


print("\nModel:")
print(model)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("\nTrainable parameters:")
print(
    f"{trainable_parameters:,}"
)


# ============================================================
# 10. Weighted loss for class imbalance
# ============================================================

train_window_counts_by_class = (
    train_metadata_df
    .groupby("Label")[
        "Number_of_Windows"
    ]
    .sum()
    .sort_index()
)

control_windows = int(
    train_window_counts_by_class.loc[0]
)

depression_windows = int(
    train_window_counts_by_class.loc[1]
)

total_train_windows = (
    control_windows
    + depression_windows
)

number_of_classes = 2

control_weight = (
    total_train_windows
    / (
        number_of_classes
        * control_windows
    )
)

depression_weight = (
    total_train_windows
    / (
        number_of_classes
        * depression_windows
    )
)

class_weights = torch.tensor(
    [
        control_weight,
        depression_weight,
    ],
    dtype=torch.float32,
    device=DEVICE,
)


print("\nTrain windows by class:")

print(
    "Control:",
    control_windows,
)

print(
    "Depression:",
    depression_windows,
)

print("\nClass weights:")

print(
    "Control:",
    round(control_weight, 4),
)

print(
    "Depression:",
    round(depression_weight, 4),
)


loss_function = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

learning_rate_scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
)


# ============================================================
# 11. Metric calculation
# ============================================================

def calculate_binary_metrics(
    true_labels: list[int],
    predicted_labels: list[int],
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
# 12. Participant-level aggregation
# ============================================================

def aggregate_participant_predictions(
    participant_ids: list[int],
    true_labels: list[int],
    depression_probabilities: list[float],
    threshold: float = 0.5,
):
    """
    Average all window depression probabilities belonging
    to the same participant.
    """

    prediction_df = pd.DataFrame(
        {
            "Participant_ID":
                participant_ids,

            "True_Label":
                true_labels,

            "Depression_Probability":
                depression_probabilities,
        }
    )

    participant_df = (
        prediction_df
        .groupby(
            "Participant_ID",
            as_index=False,
        )
        .agg(
            True_Label=(
                "True_Label",
                "first",
            ),

            Depression_Probability=(
                "Depression_Probability",
                "mean",
            ),

            Number_of_Windows=(
                "Depression_Probability",
                "size",
            ),
        )
    )

    participant_df["Predicted_Label"] = (
        participant_df[
            "Depression_Probability"
        ]
        >= threshold
    ).astype(int)

    metrics = calculate_binary_metrics(
        participant_df[
            "True_Label"
        ].tolist(),

        participant_df[
            "Predicted_Label"
        ].tolist(),
    )

    return (
        participant_df,
        metrics,
    )


# ============================================================
# 13. Training one epoch
# ============================================================

def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
) -> tuple[float, dict[str, float]]:

    model.train()

    total_loss = 0.0
    total_samples = 0

    all_true_labels = []
    all_predictions = []

    for (
        features,
        labels,
        _,
        _,
    ) in data_loader:

        features = features.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            features
        )

        loss = loss_function(
            logits,
            labels
        )

        loss.backward()

        # Prevent unstable exploding gradients.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        batch_size = labels.size(
            0
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        all_true_labels.extend(
            labels.detach()
            .cpu()
            .tolist()
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .tolist()
        )

    average_loss = (
        total_loss
        / total_samples
    )

    metrics = calculate_binary_metrics(
        all_true_labels,
        all_predictions,
    )

    return (
        average_loss,
        metrics,
    )


# ============================================================
# 14. Evaluate one split
# ============================================================

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_true_labels = []
    all_predictions = []
    all_depression_probabilities = []
    all_participant_ids = []

    for (
        features,
        labels,
        participant_ids,
        _,
    ) in data_loader:

        features = features.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        logits = model(
            features
        )

        loss = loss_function(
            logits,
            labels
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        depression_probabilities = (
            probabilities[
                :,
                1
            ]
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        batch_size = labels.size(
            0
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

        all_true_labels.extend(
            labels.cpu()
            .tolist()
        )

        all_predictions.extend(
            predictions.cpu()
            .tolist()
        )

        all_depression_probabilities.extend(
            depression_probabilities.cpu()
            .tolist()
        )

        all_participant_ids.extend(
            participant_ids.cpu()
            .tolist()
        )

    average_loss = (
        total_loss
        / total_samples
    )

    window_metrics = calculate_binary_metrics(
        all_true_labels,
        all_predictions,
    )

    (
        participant_predictions_df,
        participant_metrics,
    ) = aggregate_participant_predictions(
        participant_ids=all_participant_ids,
        true_labels=all_true_labels,
        depression_probabilities=(
            all_depression_probabilities
        ),
        threshold=0.5,
    )

    return {
        "loss":
            average_loss,

        "window_metrics":
            window_metrics,

        "participant_metrics":
            participant_metrics,

        "participant_predictions":
            participant_predictions_df,

        "window_true_labels":
            all_true_labels,

        "window_predictions":
            all_predictions,
    }


# ============================================================
# 15. Train model
# ============================================================

history_records = []

best_dev_participant_f1 = -1.0
epochs_without_improvement = 0

training_start_time = time.time()


for epoch in range(
    1,
    EPOCHS + 1,
):
    epoch_start_time = time.time()

    (
        train_loss,
        train_metrics,
    ) = train_one_epoch(
        model,
        train_loader,
    )

    dev_results = evaluate_model(
        model,
        dev_loader,
    )

    dev_window_metrics = (
        dev_results[
            "window_metrics"
        ]
    )

    dev_participant_metrics = (
        dev_results[
            "participant_metrics"
        ]
    )

    current_learning_rate = (
        optimizer
        .param_groups[0]["lr"]
    )

    epoch_duration = (
        time.time()
        - epoch_start_time
    )

    history_record = {
        "Epoch":
            epoch,

        "Learning_Rate":
            current_learning_rate,

        "Train_Loss":
            train_loss,

        "Train_Accuracy":
            train_metrics["accuracy"],

        "Train_Balanced_Accuracy":
            train_metrics[
                "balanced_accuracy"
            ],

        "Train_Precision":
            train_metrics["precision"],

        "Train_Recall":
            train_metrics["recall"],

        "Train_F1":
            train_metrics["f1"],

        "Dev_Loss":
            dev_results["loss"],

        "Dev_Window_Accuracy":
            dev_window_metrics[
                "accuracy"
            ],

        "Dev_Window_Balanced_Accuracy":
            dev_window_metrics[
                "balanced_accuracy"
            ],

        "Dev_Window_Precision":
            dev_window_metrics[
                "precision"
            ],

        "Dev_Window_Recall":
            dev_window_metrics[
                "recall"
            ],

        "Dev_Window_F1":
            dev_window_metrics[
                "f1"
            ],

        "Dev_Participant_Accuracy":
            dev_participant_metrics[
                "accuracy"
            ],

        "Dev_Participant_Balanced_Accuracy":
            dev_participant_metrics[
                "balanced_accuracy"
            ],

        "Dev_Participant_Precision":
            dev_participant_metrics[
                "precision"
            ],

        "Dev_Participant_Recall":
            dev_participant_metrics[
                "recall"
            ],

        "Dev_Participant_F1":
            dev_participant_metrics[
                "f1"
            ],

        "Epoch_Time_Seconds":
            epoch_duration,
    }

    history_records.append(
        history_record
    )

    pd.DataFrame(
        history_records
    ).to_csv(
        TRAINING_HISTORY_PATH,
        index=False,
    )

    print("\n" + "-" * 70)

    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    print(
        f"Train loss: {train_loss:.4f} "
        f"| Train F1: "
        f"{train_metrics['f1']:.4f}"
    )

    print(
        f"Dev window F1: "
        f"{dev_window_metrics['f1']:.4f} "
        f"| Dev window recall: "
        f"{dev_window_metrics['recall']:.4f}"
    )

    print(
        f"Dev participant F1: "
        f"{dev_participant_metrics['f1']:.4f} "
        f"| Dev participant recall: "
        f"{dev_participant_metrics['recall']:.4f} "
        f"| Dev balanced accuracy: "
        f"{dev_participant_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Learning rate: "
        f"{current_learning_rate:.7f}"
    )

    print(
        f"Epoch time: "
        f"{epoch_duration:.1f} seconds"
    )

    current_dev_f1 = (
        dev_participant_metrics[
            "f1"
        ]
    )

    learning_rate_scheduler.step(
        current_dev_f1
    )

    if (
        current_dev_f1
        > best_dev_participant_f1
    ):
        best_dev_participant_f1 = (
            current_dev_f1
        )

        epochs_without_improvement = 0

        torch.save(
            {
                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "best_dev_participant_f1":
                    best_dev_participant_f1,

                "model_configuration": {
                    "input_size":
                        EXPECTED_FEATURES,

                    "hidden_size":
                        HIDDEN_SIZE,

                    "number_of_layers":
                        NUMBER_OF_LSTM_LAYERS,

                    "dropout":
                        DROPOUT,
                },

                "class_weights":
                    class_weights
                    .detach()
                    .cpu()
                    .tolist(),
            },
            BEST_MODEL_PATH,
        )

        dev_results[
            "participant_predictions"
        ].to_csv(
            OUTPUT_FOLDER
            / "best_dev_participant_predictions.csv",
            index=False,
        )

        print(
            "New best model saved."
        )

    else:
        epochs_without_improvement += 1

        print(
            "No participant-level F1 improvement "
            f"for {epochs_without_improvement} epoch(s)."
        )

    if (
        epochs_without_improvement
        >= EARLY_STOPPING_PATIENCE
    ):
        print(
            "\nEarly stopping activated."
        )

        break


training_duration = (
    time.time()
    - training_start_time
)


# ============================================================
# 16. Load best model
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

print(
    "Best epoch:",
    checkpoint["epoch"],
)

print(
    "Best Dev participant F1:",
    checkpoint[
        "best_dev_participant_f1"
    ],
)


# ============================================================
# 17. Final Dev and Test evaluation
# ============================================================

final_dev_results = evaluate_model(
    model,
    dev_loader,
)

final_test_results = evaluate_model(
    model,
    test_loader,
)


final_dev_results[
    "participant_predictions"
].to_csv(
    OUTPUT_FOLDER
    / "final_dev_participant_predictions.csv",
    index=False,
)

final_test_results[
    "participant_predictions"
].to_csv(
    OUTPUT_FOLDER
    / "final_test_participant_predictions.csv",
    index=False,
)


# ============================================================
# 18. Save classification reports
# ============================================================

def save_participant_classification_report(
    participant_predictions_df: pd.DataFrame,
    output_path: Path,
) -> None:

    report = classification_report(
        participant_predictions_df[
            "True_Label"
        ],

        participant_predictions_df[
            "Predicted_Label"
        ],

        target_names=[
            "Control",
            "Depression",
        ],

        digits=4,

        zero_division=0,
    )

    output_path.write_text(
        report,
        encoding="utf-8",
    )


save_participant_classification_report(
    final_dev_results[
        "participant_predictions"
    ],

    OUTPUT_FOLDER
    / "dev_participant_classification_report.txt",
)

save_participant_classification_report(
    final_test_results[
        "participant_predictions"
    ],

    OUTPUT_FOLDER
    / "test_participant_classification_report.txt",
)


# ============================================================
# 19. Confusion matrix plots
# ============================================================

def save_confusion_matrix_plot(
    true_labels,
    predicted_labels,
    title: str,
    output_path: Path,
) -> None:

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
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
        title
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
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


dev_participant_df = (
    final_dev_results[
        "participant_predictions"
    ]
)

test_participant_df = (
    final_test_results[
        "participant_predictions"
    ]
)


save_confusion_matrix_plot(
    true_labels=(
        dev_participant_df[
            "True_Label"
        ]
    ),

    predicted_labels=(
        dev_participant_df[
            "Predicted_Label"
        ]
    ),

    title=(
        "Dev Participant-Level "
        "Confusion Matrix"
    ),

    output_path=(
        OUTPUT_FOLDER
        / "dev_participant_confusion_matrix.png"
    ),
)

save_confusion_matrix_plot(
    true_labels=(
        test_participant_df[
            "True_Label"
        ]
    ),

    predicted_labels=(
        test_participant_df[
            "Predicted_Label"
        ]
    ),

    title=(
        "Test Participant-Level "
        "Confusion Matrix"
    ),

    output_path=(
        OUTPUT_FOLDER
        / "test_participant_confusion_matrix.png"
    ),
)


# ============================================================
# 20. Training history plots
# ============================================================

history_df = pd.DataFrame(
    history_records
)

figure, axis = plt.subplots(
    figsize=(8, 5)
)

axis.plot(
    history_df["Epoch"],
    history_df["Train_Loss"],
    label="Train loss",
)

axis.plot(
    history_df["Epoch"],
    history_df["Dev_Loss"],
    label="Dev loss",
)

axis.set_xlabel(
    "Epoch"
)

axis.set_ylabel(
    "Loss"
)

axis.set_title(
    "LSTM Training and Dev Loss"
)

axis.legend()

figure.tight_layout()

figure.savefig(
    OUTPUT_FOLDER
    / "loss_history.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    figure
)


figure, axis = plt.subplots(
    figsize=(8, 5)
)

axis.plot(
    history_df["Epoch"],
    history_df["Train_F1"],
    label="Train window F1",
)

axis.plot(
    history_df["Epoch"],
    history_df["Dev_Window_F1"],
    label="Dev window F1",
)

axis.plot(
    history_df["Epoch"],
    history_df["Dev_Participant_F1"],
    label="Dev participant F1",
)

axis.set_xlabel(
    "Epoch"
)

axis.set_ylabel(
    "F1-score"
)

axis.set_title(
    "LSTM F1-Score History"
)

axis.legend()

figure.tight_layout()

figure.savefig(
    OUTPUT_FOLDER
    / "f1_history.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    figure
)


# ============================================================
# 21. Save final results
# ============================================================

final_results = {
    "device":
        str(DEVICE),

    "best_epoch":
        int(
            checkpoint["epoch"]
        ),

    "training_duration_seconds":
        float(
            training_duration
        ),

    "model_configuration": {
        "input_features":
            EXPECTED_FEATURES,

        "window_frames":
            EXPECTED_WINDOW_FRAMES,

        "hidden_size":
            HIDDEN_SIZE,

        "number_of_lstm_layers":
            NUMBER_OF_LSTM_LAYERS,

        "dropout":
            DROPOUT,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,
    },

    "dev_window_metrics":
        final_dev_results[
            "window_metrics"
        ],

    "dev_participant_metrics":
        final_dev_results[
            "participant_metrics"
        ],

    "test_window_metrics":
        final_test_results[
            "window_metrics"
        ],

    "test_participant_metrics":
        final_test_results[
            "participant_metrics"
        ],
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
# 22. Final output
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print("\nWindow-level metrics:")

for metric_name, metric_value in (
    final_test_results[
        "window_metrics"
    ].items()
):
    print(
        f"{metric_name}: "
        f"{metric_value:.4f}"
    )


print("\nParticipant-level metrics:")

for metric_name, metric_value in (
    final_test_results[
        "participant_metrics"
    ].items()
):
    print(
        f"{metric_name}: "
        f"{metric_value:.4f}"
    )


print("\nParticipant-level classification report:")

print(
    classification_report(
        test_participant_df[
            "True_Label"
        ],

        test_participant_df[
            "Predicted_Label"
        ],

        target_names=[
            "Control",
            "Depression",
        ],

        digits=4,

        zero_division=0,
    )
)


print("\nOutputs saved in:")
print(OUTPUT_FOLDER)

print("\nBest model:")
print(BEST_MODEL_PATH)

print("\nFinal results:")
print(FINAL_RESULTS_PATH)