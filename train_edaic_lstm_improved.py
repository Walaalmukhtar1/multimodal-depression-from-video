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
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)


# ============================================================
# 1. Experiment configuration
# ============================================================

SEED = 42

BATCH_SIZE = 32
EPOCHS = 30

LEARNING_RATE = 5e-4
WEIGHT_DECAY = 5e-4

HIDDEN_SIZE = 64
NUMBER_OF_LSTM_LAYERS = 1

LSTM_DROPOUT = 0.0
CLASSIFIER_DROPOUT = 0.50

EXPECTED_WINDOW_FRAMES = 270
EXPECTED_FEATURES = 28

EARLY_STOPPING_PATIENCE = 7

NUM_WORKERS = 0
ARRAY_CACHE_SIZE = 8

# The sampler draws the same total number of windows
# as the original Train set during each epoch.
USE_PARTICIPANT_BALANCED_SAMPLING = True


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
    r"\model_outputs\edaic_lstm_improved"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

BEST_MODEL_PATH = (
    OUTPUT_FOLDER
    / "best_improved_lstm_model.pt"
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
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_random_seed(SEED)


# ============================================================
# 4. Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("E-DAIC-WOZ IMPROVED LSTM")
print("=" * 70)

print("\nDevice:")
print(DEVICE)

if DEVICE.type == "cuda":
    print("\nGPU:")
    print(torch.cuda.get_device_name(0))


# ============================================================
# 5. Read and validate metadata
# ============================================================

def read_windows_metadata(
    metadata_path: Path,
) -> pd.DataFrame:

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata was not found:\n{metadata_path}"
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


print("\nParticipants:")

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
# 6. Dataset
# ============================================================

class VisualWindowDataset(Dataset):
    """
    One item represents one 9-second visual window.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        array_cache_size: int = 8,
    ) -> None:

        self.samples: list[dict] = []

        self.array_cache_size = (
            array_cache_size
        )

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

            if not Path(
                windows_file
            ).exists():
                raise FileNotFoundError(
                    f"Windows file was not found: "
                    f"{windows_file}"
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

                        "participant_window_count":
                            number_of_windows,
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
                f"Expected a 3D array, found "
                f"{array.shape} in {windows_file}"
            )

        if array.shape[1:] != (
            EXPECTED_WINDOW_FRAMES,
            EXPECTED_FEATURES,
        ):
            raise ValueError(
                f"Expected window shape "
                f"({EXPECTED_WINDOW_FRAMES}, "
                f"{EXPECTED_FEATURES}), "
                f"found {array.shape[1:]}"
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

        return (
            torch.from_numpy(
                window
            ),

            torch.tensor(
                sample["label"],
                dtype=torch.long,
            ),

            torch.tensor(
                sample["participant_id"],
                dtype=torch.long,
            ),

            torch.tensor(
                sample["window_index"],
                dtype=torch.long,
            ),
        )


train_dataset = VisualWindowDataset(
    train_metadata_df,
    ARRAY_CACHE_SIZE,
)

dev_dataset = VisualWindowDataset(
    dev_metadata_df,
    ARRAY_CACHE_SIZE,
)

test_dataset = VisualWindowDataset(
    test_metadata_df,
    ARRAY_CACHE_SIZE,
)


print("\nWindows:")

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
# 7. Participant-balanced sampler
# ============================================================

def create_participant_balanced_sampler(
    dataset: VisualWindowDataset,
) -> WeightedRandomSampler:
    """
    Give each participant approximately equal total probability.

    A participant with 200 windows will have a smaller weight
    per window than a participant with 50 windows.
    """

    sample_weights = []

    for sample in dataset.samples:
        participant_window_count = int(
            sample[
                "participant_window_count"
            ]
        )

        weight = (
            1.0
            / participant_window_count
        )

        sample_weights.append(
            weight
        )

    weights_tensor = torch.tensor(
        sample_weights,
        dtype=torch.double,
    )

    return WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=len(dataset),
        replacement=True,
    )


if USE_PARTICIPANT_BALANCED_SAMPLING:

    train_sampler = (
        create_participant_balanced_sampler(
            train_dataset
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

else:

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
# 8. Improved LSTM
# ============================================================

class ImprovedDepressionLSTM(nn.Module):
    """
    Smaller LSTM using temporal mean and max pooling.

    Instead of using only the final frame, it summarizes
    information from the entire 9-second window.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        number_of_layers: int,
        lstm_dropout: float,
        classifier_dropout: float,
    ) -> None:

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=number_of_layers,
            batch_first=True,
            dropout=(
                lstm_dropout
                if number_of_layers > 1
                else 0.0
            ),
            bidirectional=False,
        )

        pooled_size = (
            hidden_size * 2
        )

        self.normalization = nn.LayerNorm(
            pooled_size
        )

        self.classifier = nn.Sequential(
            nn.Dropout(
                classifier_dropout
            ),

            nn.Linear(
                pooled_size,
                hidden_size,
            ),

            nn.ReLU(),

            nn.Dropout(
                classifier_dropout
            ),

            nn.Linear(
                hidden_size,
                2,
            ),
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:

        sequence_output, _ = self.lstm(
            features
        )

        mean_pooled = sequence_output.mean(
            dim=1
        )

        max_pooled = sequence_output.max(
            dim=1
        ).values

        pooled_features = torch.cat(
            [
                mean_pooled,
                max_pooled,
            ],
            dim=1,
        )

        pooled_features = self.normalization(
            pooled_features
        )

        logits = self.classifier(
            pooled_features
        )

        return logits


model = ImprovedDepressionLSTM(
    input_size=EXPECTED_FEATURES,
    hidden_size=HIDDEN_SIZE,
    number_of_layers=(
        NUMBER_OF_LSTM_LAYERS
    ),
    lstm_dropout=LSTM_DROPOUT,
    classifier_dropout=(
        CLASSIFIER_DROPOUT
    ),
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
# 9. Class weights based on participants
# ============================================================

participant_class_counts = (
    train_metadata_df[
        "Label"
    ]
    .value_counts()
    .sort_index()
)

control_participants = int(
    participant_class_counts.loc[0]
)

depression_participants = int(
    participant_class_counts.loc[1]
)

total_participants = (
    control_participants
    + depression_participants
)

control_weight = (
    total_participants
    / (
        2
        * control_participants
    )
)

depression_weight = (
    total_participants
    / (
        2
        * depression_participants
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


print("\nTrain participants by class:")

print(
    "Control:",
    control_participants,
)

print(
    "Depression:",
    depression_participants,
)

print("\nParticipant-based class weights:")

print(
    "Control:",
    round(
        control_weight,
        4,
    ),
)

print(
    "Depression:",
    round(
        depression_weight,
        4,
    ),
)


loss_function = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.05,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
)


# ============================================================
# 10. Metrics
# ============================================================

def calculate_binary_metrics(
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


def aggregate_participant_predictions(
    participant_ids,
    true_labels,
    depression_probabilities,
    threshold: float = 0.5,
):
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
        ],

        participant_df[
            "Predicted_Label"
        ],
    )

    return (
        participant_df,
        metrics,
    )


# ============================================================
# 11. Train one epoch
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
):

    model.train()

    total_loss = 0.0
    total_samples = 0

    true_labels = []
    predicted_labels = []

    for (
        features,
        labels,
        _,
        _,
    ) in loader:

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

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=3.0,
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

        predictions = logits.argmax(
            dim=1
        )

        true_labels.extend(
            labels.detach()
            .cpu()
            .tolist()
        )

        predicted_labels.extend(
            predictions.detach()
            .cpu()
            .tolist()
        )

    average_loss = (
        total_loss
        / total_samples
    )

    metrics = calculate_binary_metrics(
        true_labels,
        predicted_labels,
    )

    return (
        average_loss,
        metrics,
    )


# ============================================================
# 12. Evaluate
# ============================================================

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    true_labels = []
    predicted_labels = []

    depression_probabilities = []
    participant_ids_list = []

    for (
        features,
        labels,
        participant_ids,
        _,
    ) in loader:

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

        predictions = logits.argmax(
            dim=1
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

        true_labels.extend(
            labels.cpu()
            .tolist()
        )

        predicted_labels.extend(
            predictions.cpu()
            .tolist()
        )

        depression_probabilities.extend(
            probabilities[
                :,
                1
            ]
            .cpu()
            .tolist()
        )

        participant_ids_list.extend(
            participant_ids
            .cpu()
            .tolist()
        )

    average_loss = (
        total_loss
        / total_samples
    )

    window_metrics = calculate_binary_metrics(
        true_labels,
        predicted_labels,
    )

    (
        participant_df,
        participant_metrics,
    ) = aggregate_participant_predictions(
        participant_ids_list,
        true_labels,
        depression_probabilities,
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
            participant_df,
    }


# ============================================================
# 13. Training loop
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

    learning_rate = (
        optimizer
        .param_groups[0]["lr"]
    )

    epoch_time = (
        time.time()
        - epoch_start_time
    )

    record = {
        "Epoch":
            epoch,

        "Learning_Rate":
            learning_rate,

        "Train_Loss":
            train_loss,

        "Train_F1":
            train_metrics["f1"],

        "Train_Balanced_Accuracy":
            train_metrics[
                "balanced_accuracy"
            ],

        "Dev_Loss":
            dev_results["loss"],

        "Dev_Window_F1":
            dev_window_metrics["f1"],

        "Dev_Window_Recall":
            dev_window_metrics["recall"],

        "Dev_Participant_F1":
            dev_participant_metrics["f1"],

        "Dev_Participant_Recall":
            dev_participant_metrics[
                "recall"
            ],

        "Dev_Participant_Balanced_Accuracy":
            dev_participant_metrics[
                "balanced_accuracy"
            ],

        "Epoch_Time_Seconds":
            epoch_time,
    }

    history_records.append(
        record
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
        f"| Recall: "
        f"{dev_window_metrics['recall']:.4f}"
    )

    print(
        f"Dev participant F1: "
        f"{dev_participant_metrics['f1']:.4f} "
        f"| Recall: "
        f"{dev_participant_metrics['recall']:.4f} "
        f"| Balanced accuracy: "
        f"{dev_participant_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Learning rate: "
        f"{learning_rate:.7f}"
    )

    print(
        f"Epoch time: "
        f"{epoch_time:.1f} seconds"
    )

    current_dev_f1 = (
        dev_participant_metrics["f1"]
    )

    scheduler.step(
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

                "configuration": {
                    "hidden_size":
                        HIDDEN_SIZE,

                    "number_of_layers":
                        NUMBER_OF_LSTM_LAYERS,

                    "classifier_dropout":
                        CLASSIFIER_DROPOUT,

                    "participant_balanced_sampling":
                        USE_PARTICIPANT_BALANCED_SAMPLING,
                },
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
# 14. Load best model
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST IMPROVED MODEL")
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
# 15. Final Dev and Test evaluation
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
# 16. Save reports
# ============================================================

test_participant_df = (
    final_test_results[
        "participant_predictions"
    ]
)

test_report = classification_report(
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

(
    OUTPUT_FOLDER
    / "test_participant_classification_report.txt"
).write_text(
    test_report,
    encoding="utf-8",
)


# ============================================================
# 17. Confusion matrix
# ============================================================

matrix = confusion_matrix(
    test_participant_df[
        "True_Label"
    ],

    test_participant_df[
        "Predicted_Label"
    ],

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
    "Improved LSTM Test Confusion Matrix"
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
    OUTPUT_FOLDER
    / "test_participant_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    figure
)


# ============================================================
# 18. Training plots
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
    "Improved LSTM Loss History"
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
    "Improved LSTM F1 History"
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
# 19. Save final results
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
        "hidden_size":
            HIDDEN_SIZE,

        "number_of_lstm_layers":
            NUMBER_OF_LSTM_LAYERS,

        "classifier_dropout":
            CLASSIFIER_DROPOUT,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "participant_balanced_sampling":
            USE_PARTICIPANT_BALANCED_SAMPLING,

        "temporal_pooling":
            "mean_and_max",
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
# 20. Final output
# ============================================================

print("\n" + "=" * 70)
print("IMPROVED LSTM FINAL TEST RESULTS")
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


print("\nClassification report:")

print(
    test_report
)

print("\nConfusion matrix:")

print(
    matrix
)

print("\nOutputs saved in:")

print(
    OUTPUT_FOLDER
)