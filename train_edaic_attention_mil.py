from __future__ import annotations

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
# 1. Configuration
# ============================================================

SEED = 42
BATCH_SIZE = 4
EPOCHS = 40
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

WINDOWS_PER_PARTICIPANT = 24
EXPECTED_WINDOW_FRAMES = 270
EXPECTED_FEATURES = 28

GRU_HIDDEN_SIZE = 48
ATTENTION_HIDDEN_SIZE = 32
DROPOUT = 0.40

EARLY_STOPPING_PATIENCE = 8
NUM_WORKERS = 0
DECISION_THRESHOLD = 0.50


# ============================================================
# 2. Paths
# ============================================================

WINDOWS_FOLDER = Path(
    r"D:\Downloads\Depression_Anxiety_Body_Movement\data\processed"
    r"\e_daic_woz_visual_windows"
)

TRAIN_METADATA_PATH = WINDOWS_FOLDER / "train_windows_metadata.csv"
DEV_METADATA_PATH = WINDOWS_FOLDER / "dev_windows_metadata.csv"
TEST_METADATA_PATH = WINDOWS_FOLDER / "test_windows_metadata.csv"

OUTPUT_FOLDER = Path(
    r"D:\Flutterr\multimodal-depression-from-video"
    r"\model_outputs\edaic_attention_mil"
)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = OUTPUT_FOLDER / "best_attention_mil_model.pt"
TRAINING_HISTORY_PATH = OUTPUT_FOLDER / "training_history.csv"
FINAL_RESULTS_PATH = OUTPUT_FOLDER / "final_results.json"


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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 72)
print("E-DAIC-WOZ ATTENTION MULTIPLE INSTANCE LEARNING")
print("=" * 72)
print("\nDevice:")
print(DEVICE)

if DEVICE.type == "cuda":
    print("\nGPU:")
    print(torch.cuda.get_device_name(0))


# ============================================================
# 5. Metadata
# ============================================================

def read_metadata(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata was not found:\n{metadata_path}")

    dataframe = pd.read_csv(metadata_path)

    required_columns = [
        "Participant_ID",
        "Split",
        "Label",
        "Number_of_Windows",
        "Number_of_Features",
        "Windows_File",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{metadata_path.name} is missing columns: {missing_columns}"
        )

    dataframe["Participant_ID"] = pd.to_numeric(
        dataframe["Participant_ID"], errors="raise"
    ).astype(int)

    dataframe["Label"] = pd.to_numeric(
        dataframe["Label"], errors="raise"
    ).astype(int)

    dataframe["Number_of_Windows"] = pd.to_numeric(
        dataframe["Number_of_Windows"], errors="raise"
    ).astype(int)

    return dataframe


train_metadata_df = read_metadata(TRAIN_METADATA_PATH)
dev_metadata_df = read_metadata(DEV_METADATA_PATH)
test_metadata_df = read_metadata(TEST_METADATA_PATH)

print("\nParticipant counts:")
print(f"Train: {len(train_metadata_df)}")
print(f"Dev:   {len(dev_metadata_df)}")
print(f"Test:  {len(test_metadata_df)}")


# ============================================================
# 6. Participant-level MIL dataset
# ============================================================

class ParticipantBagDataset(Dataset):
    """
    Each item is one participant (one bag).

    Each bag contains a fixed number of 9-second windows.
    Training: random windows every epoch.
    Evaluation: deterministic evenly spaced windows.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        windows_per_participant: int,
        training: bool,
    ) -> None:
        self.metadata_df = metadata_df.reset_index(drop=True).copy()
        self.windows_per_participant = windows_per_participant
        self.training = training

        for row in self.metadata_df.itertuples(index=False):
            windows_file = Path(str(row.Windows_File))
            if not windows_file.exists():
                raise FileNotFoundError(
                    f"Windows file was not found:\n{windows_file}"
                )

    def __len__(self) -> int:
        return len(self.metadata_df)

    def _select_indices(self, number_of_windows: int) -> np.ndarray:
        target = self.windows_per_participant

        if self.training:
            return np.random.choice(
                number_of_windows,
                size=target,
                replace=(number_of_windows < target),
            )

        if number_of_windows >= target:
            return np.linspace(
                0,
                number_of_windows - 1,
                num=target,
                dtype=np.int64,
            )

        repeats = int(np.ceil(target / number_of_windows))
        tiled = np.tile(np.arange(number_of_windows), repeats)
        return tiled[:target]

    def __getitem__(self, index: int):
        row = self.metadata_df.iloc[index]

        participant_id = int(row["Participant_ID"])
        label = int(row["Label"])
        windows_file = Path(str(row["Windows_File"]))

        windows_array = np.load(windows_file, mmap_mode="r")

        if windows_array.ndim != 3:
            raise ValueError(
                f"Expected 3D array, found {windows_array.shape}"
            )

        if windows_array.shape[1:] != (
            EXPECTED_WINDOW_FRAMES,
            EXPECTED_FEATURES,
        ):
            raise ValueError(
                f"Unexpected window shape in {windows_file}: "
                f"{windows_array.shape}"
            )

        selected_indices = self._select_indices(
            windows_array.shape[0]
        )

        bag = np.asarray(
            windows_array[selected_indices],
            dtype=np.float32,
        )

        return (
            torch.from_numpy(bag),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(participant_id, dtype=torch.long),
        )


train_dataset = ParticipantBagDataset(
    train_metadata_df,
    WINDOWS_PER_PARTICIPANT,
    training=True,
)

dev_dataset = ParticipantBagDataset(
    dev_metadata_df,
    WINDOWS_PER_PARTICIPANT,
    training=False,
)

test_dataset = ParticipantBagDataset(
    test_metadata_df,
    WINDOWS_PER_PARTICIPANT,
    training=False,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)


# ============================================================
# 7. Attention MIL model
# ============================================================

class AttentionMIL(nn.Module):
    """
    1. Encode each 9-second window using bidirectional GRU.
    2. Pool frames inside every window using mean + max.
    3. Learn attention weights across participant windows.
    4. Predict one label for the entire participant.
    """

    def __init__(
        self,
        input_size: int,
        gru_hidden_size: int,
        attention_hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.window_encoder = nn.GRU(
            input_size=input_size,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.window_embedding_size = gru_hidden_size * 4

        self.window_normalization = nn.LayerNorm(
            self.window_embedding_size
        )

        self.attention = nn.Sequential(
            nn.Linear(
                self.window_embedding_size,
                attention_hidden_size,
            ),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attention_hidden_size, 1),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.window_embedding_size),
            nn.Dropout(dropout),
            nn.Linear(self.window_embedding_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(
        self,
        bags: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, number_of_windows, frames, features = bags.shape

        flattened_windows = bags.reshape(
            batch_size * number_of_windows,
            frames,
            features,
        )

        sequence_output, _ = self.window_encoder(
            flattened_windows
        )

        mean_pooled = sequence_output.mean(dim=1)
        max_pooled = sequence_output.max(dim=1).values

        window_embeddings = torch.cat(
            [mean_pooled, max_pooled],
            dim=1,
        )

        window_embeddings = self.window_normalization(
            window_embeddings
        )

        window_embeddings = window_embeddings.reshape(
            batch_size,
            number_of_windows,
            self.window_embedding_size,
        )

        attention_logits = self.attention(
            window_embeddings
        ).squeeze(-1)

        attention_weights = torch.softmax(
            attention_logits,
            dim=1,
        )

        participant_embedding = torch.sum(
            window_embeddings
            * attention_weights.unsqueeze(-1),
            dim=1,
        )

        logits = self.classifier(
            participant_embedding
        )

        return logits, attention_weights


model = AttentionMIL(
    input_size=EXPECTED_FEATURES,
    gru_hidden_size=GRU_HIDDEN_SIZE,
    attention_hidden_size=ATTENTION_HIDDEN_SIZE,
    dropout=DROPOUT,
).to(DEVICE)

print("\nModel:")
print(model)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("\nTrainable parameters:")
print(f"{trainable_parameters:,}")


# ============================================================
# 8. Loss and optimizer
# ============================================================

participant_counts = (
    train_metadata_df["Label"].value_counts().sort_index()
)

control_count = int(participant_counts.loc[0])
depression_count = int(participant_counts.loc[1])
total_count = control_count + depression_count

class_weights = torch.tensor(
    [
        total_count / (2 * control_count),
        total_count / (2 * depression_count),
    ],
    dtype=torch.float32,
    device=DEVICE,
)

print("\nTrain participants by class:")
print(f"Control:    {control_count}")
print(f"Depression: {depression_count}")

print("\nClass weights:")
print(f"Control:    {class_weights[0].item():.4f}")
print(f"Depression: {class_weights[1].item():.4f}")

loss_function = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.03,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
)


# ============================================================
# 9. Metrics
# ============================================================

def calculate_metrics(
    true_labels,
    predicted_labels,
) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "balanced_accuracy": balanced_accuracy_score(
            true_labels,
            predicted_labels,
        ),
        "precision": precision_score(
            true_labels,
            predicted_labels,
            zero_division=0,
        ),
        "recall": recall_score(
            true_labels,
            predicted_labels,
            zero_division=0,
        ),
        "f1": f1_score(
            true_labels,
            predicted_labels,
            zero_division=0,
        ),
    }


# ============================================================
# 10. Training and evaluation
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
) -> tuple[float, dict[str, float]]:
    model.train()

    total_loss = 0.0
    total_samples = 0

    true_labels = []
    predicted_labels = []

    for bags, labels, _ in loader:
        bags = bags.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits, _ = model(bags)
        loss = loss_function(logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=3.0,
        )

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        predictions = logits.argmax(dim=1)

        true_labels.extend(
            labels.detach().cpu().tolist()
        )
        predicted_labels.extend(
            predictions.detach().cpu().tolist()
        )

    return (
        total_loss / total_samples,
        calculate_metrics(
            true_labels,
            predicted_labels,
        ),
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
) -> dict:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    participant_ids_list = []
    true_labels = []
    predicted_labels = []
    depression_probabilities = []
    attention_rows = []

    for bags, labels, participant_ids in loader:
        bags = bags.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        logits, attention_weights = model(bags)
        loss = loss_function(logits, labels)

        probabilities = torch.softmax(logits, dim=1)

        predictions = (
            probabilities[:, 1] >= DECISION_THRESHOLD
        ).long()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        participant_ids_cpu = participant_ids.cpu().tolist()
        labels_cpu = labels.cpu().tolist()
        predictions_cpu = predictions.cpu().tolist()
        probabilities_cpu = probabilities[:, 1].cpu().tolist()
        attention_cpu = attention_weights.cpu().numpy()

        participant_ids_list.extend(participant_ids_cpu)
        true_labels.extend(labels_cpu)
        predicted_labels.extend(predictions_cpu)
        depression_probabilities.extend(probabilities_cpu)

        for participant_id, weights in zip(
            participant_ids_cpu,
            attention_cpu,
        ):
            row = {
                "Participant_ID": int(participant_id),
            }

            for window_index, weight in enumerate(weights):
                row[f"Attention_{window_index:02d}"] = float(weight)

            attention_rows.append(row)

    predictions_df = pd.DataFrame(
        {
            "Participant_ID": participant_ids_list,
            "True_Label": true_labels,
            "Predicted_Label": predicted_labels,
            "Depression_Probability": depression_probabilities,
        }
    )

    attention_df = pd.DataFrame(attention_rows)

    return {
        "loss": total_loss / total_samples,
        "metrics": calculate_metrics(
            true_labels,
            predicted_labels,
        ),
        "predictions": predictions_df,
        "attention": attention_df,
    }


# ============================================================
# 11. Training loop
# ============================================================

history_records = []

best_dev_f1 = -1.0
best_dev_balanced_accuracy = -1.0
epochs_without_improvement = 0

training_start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()

    train_loss, train_metrics = train_one_epoch(
        model,
        train_loader,
    )

    dev_results = evaluate_model(
        model,
        dev_loader,
    )

    dev_metrics = dev_results["metrics"]
    learning_rate = optimizer.param_groups[0]["lr"]
    epoch_time = time.time() - epoch_start_time

    history_records.append(
        {
            "Epoch": epoch,
            "Learning_Rate": learning_rate,
            "Train_Loss": train_loss,
            "Train_Accuracy": train_metrics["accuracy"],
            "Train_Balanced_Accuracy": train_metrics[
                "balanced_accuracy"
            ],
            "Train_Precision": train_metrics["precision"],
            "Train_Recall": train_metrics["recall"],
            "Train_F1": train_metrics["f1"],
            "Dev_Loss": dev_results["loss"],
            "Dev_Accuracy": dev_metrics["accuracy"],
            "Dev_Balanced_Accuracy": dev_metrics[
                "balanced_accuracy"
            ],
            "Dev_Precision": dev_metrics["precision"],
            "Dev_Recall": dev_metrics["recall"],
            "Dev_F1": dev_metrics["f1"],
            "Epoch_Time_Seconds": epoch_time,
        }
    )

    pd.DataFrame(history_records).to_csv(
        TRAINING_HISTORY_PATH,
        index=False,
    )

    print("\n" + "-" * 72)
    print(f"Epoch {epoch}/{EPOCHS}")

    print(
        f"Train loss: {train_loss:.4f} "
        f"| F1: {train_metrics['f1']:.4f} "
        f"| Recall: {train_metrics['recall']:.4f}"
    )

    print(
        f"Dev loss: {dev_results['loss']:.4f} "
        f"| F1: {dev_metrics['f1']:.4f} "
        f"| Recall: {dev_metrics['recall']:.4f} "
        f"| Balanced accuracy: "
        f"{dev_metrics['balanced_accuracy']:.4f}"
    )

    print(f"Learning rate: {learning_rate:.7f}")
    print(f"Epoch time: {epoch_time:.1f} seconds")

    scheduler.step(dev_metrics["f1"])

    improved = (
        dev_metrics["f1"] > best_dev_f1
        or (
            np.isclose(dev_metrics["f1"], best_dev_f1)
            and dev_metrics["balanced_accuracy"]
            > best_dev_balanced_accuracy
        )
    )

    if improved:
        best_dev_f1 = dev_metrics["f1"]
        best_dev_balanced_accuracy = dev_metrics[
            "balanced_accuracy"
        ]
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dev_f1": best_dev_f1,
                "best_dev_balanced_accuracy":
                    best_dev_balanced_accuracy,
                "configuration": {
                    "windows_per_participant":
                        WINDOWS_PER_PARTICIPANT,
                    "gru_hidden_size":
                        GRU_HIDDEN_SIZE,
                    "attention_hidden_size":
                        ATTENTION_HIDDEN_SIZE,
                    "dropout":
                        DROPOUT,
                    "decision_threshold":
                        DECISION_THRESHOLD,
                },
            },
            BEST_MODEL_PATH,
        )

        dev_results["predictions"].to_csv(
            OUTPUT_FOLDER
            / "best_dev_participant_predictions.csv",
            index=False,
        )

        dev_results["attention"].to_csv(
            OUTPUT_FOLDER
            / "best_dev_attention_weights.csv",
            index=False,
        )

        print("New best model saved.")

    else:
        epochs_without_improvement += 1

        print(
            "No Dev F1 improvement for "
            f"{epochs_without_improvement} epoch(s)."
        )

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print("\nEarly stopping activated.")
        break


training_duration = time.time() - training_start_time


# ============================================================
# 12. Load best model
# ============================================================

print("\n" + "=" * 72)
print("LOADING BEST ATTENTION MIL MODEL")
print("=" * 72)

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(f"Best epoch: {checkpoint['epoch']}")
print(f"Best Dev F1: {checkpoint['best_dev_f1']:.4f}")
print(
    "Best Dev balanced accuracy: "
    f"{checkpoint['best_dev_balanced_accuracy']:.4f}"
)


# ============================================================
# 13. Final evaluation
# ============================================================

final_dev_results = evaluate_model(
    model,
    dev_loader,
)

final_test_results = evaluate_model(
    model,
    test_loader,
)

final_dev_results["predictions"].to_csv(
    OUTPUT_FOLDER
    / "final_dev_participant_predictions.csv",
    index=False,
)

final_test_results["predictions"].to_csv(
    OUTPUT_FOLDER
    / "final_test_participant_predictions.csv",
    index=False,
)

final_dev_results["attention"].to_csv(
    OUTPUT_FOLDER
    / "final_dev_attention_weights.csv",
    index=False,
)

final_test_results["attention"].to_csv(
    OUTPUT_FOLDER
    / "final_test_attention_weights.csv",
    index=False,
)


# ============================================================
# 14. Reports and plots
# ============================================================

test_predictions_df = final_test_results["predictions"]

test_report = classification_report(
    test_predictions_df["True_Label"],
    test_predictions_df["Predicted_Label"],
    target_names=["Control", "Depression"],
    digits=4,
    zero_division=0,
)

(
    OUTPUT_FOLDER
    / "test_classification_report.txt"
).write_text(
    test_report,
    encoding="utf-8",
)

matrix = confusion_matrix(
    test_predictions_df["True_Label"],
    test_predictions_df["Predicted_Label"],
    labels=[0, 1],
)

figure, axis = plt.subplots(figsize=(6, 5))
image = axis.imshow(matrix)
figure.colorbar(image, ax=axis)

axis.set_title("Attention MIL Test Confusion Matrix")
axis.set_xlabel("Predicted label")
axis.set_ylabel("True label")

axis.set_xticks(
    [0, 1],
    labels=["Control", "Depression"],
)
axis.set_yticks(
    [0, 1],
    labels=["Control", "Depression"],
)

for row_index in range(2):
    for column_index in range(2):
        axis.text(
            column_index,
            row_index,
            str(matrix[row_index, column_index]),
            ha="center",
            va="center",
        )

figure.tight_layout()
figure.savefig(
    OUTPUT_FOLDER / "test_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(figure)


history_df = pd.DataFrame(history_records)

figure, axis = plt.subplots(figsize=(8, 5))
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
axis.set_xlabel("Epoch")
axis.set_ylabel("Loss")
axis.set_title("Attention MIL Loss History")
axis.legend()
figure.tight_layout()
figure.savefig(
    OUTPUT_FOLDER / "loss_history.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(figure)


figure, axis = plt.subplots(figsize=(8, 5))
axis.plot(
    history_df["Epoch"],
    history_df["Train_F1"],
    label="Train F1",
)
axis.plot(
    history_df["Epoch"],
    history_df["Dev_F1"],
    label="Dev F1",
)
axis.set_xlabel("Epoch")
axis.set_ylabel("F1-score")
axis.set_title("Attention MIL F1 History")
axis.legend()
figure.tight_layout()
figure.savefig(
    OUTPUT_FOLDER / "f1_history.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(figure)


# ============================================================
# 15. Save final results
# ============================================================

final_results = {
    "device": str(DEVICE),
    "best_epoch": int(checkpoint["epoch"]),
    "training_duration_seconds": float(training_duration),
    "configuration": {
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "windows_per_participant":
            WINDOWS_PER_PARTICIPANT,
        "gru_hidden_size": GRU_HIDDEN_SIZE,
        "attention_hidden_size":
            ATTENTION_HIDDEN_SIZE,
        "dropout": DROPOUT,
        "decision_threshold":
            DECISION_THRESHOLD,
    },
    "best_dev_metrics": {
        metric_name: float(metric_value)
        for metric_name, metric_value
        in final_dev_results["metrics"].items()
    },
    "test_metrics": {
        metric_name: float(metric_value)
        for metric_name, metric_value
        in final_test_results["metrics"].items()
    },
    "test_confusion_matrix": matrix.tolist(),
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
# 16. Final output
# ============================================================

print("\n" + "=" * 72)
print("ATTENTION MIL FINAL TEST RESULTS")
print("=" * 72)

print("\nParticipant-level Test metrics:")

for metric_name, metric_value in (
    final_test_results["metrics"].items()
):
    print(f"{metric_name}: {metric_value:.4f}")

print("\nClassification report:")
print(test_report)

print("\nConfusion matrix:")
print(matrix)

print("\nBaseline to beat:")
print("LSTM participant F1: 0.4444")
print("LSTM balanced accuracy: 0.5943")

print("\nOutputs saved in:")
print(OUTPUT_FOLDER)