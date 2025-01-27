import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from datasets import MIDIDataset
from model import NoteEventPredictor

def reduce_dataset(dataset, fraction=0.05):
    total_size = len(dataset)
    subset_size = int(total_size * fraction)
    indices = torch.randperm(total_size).tolist()
    subset_indices = indices[:subset_size]
    return Subset(dataset, subset_indices)

# Creating the DataLoader
def get_data_loader(file_paths, batch_size=64, context_size=196, overlapping=True, shuffle=True):
    dataset = MIDIDataset(file_paths, context_size, overlapping)
    # print(type(dataset))
    # print(len(dataset.data))
    fraction = 0.05  # 5%
    reduced_dataset = reduce_dataset(dataset, fraction)
    print(f"Reduced dataset size: {len(reduced_dataset)}")

    return DataLoader(reduced_dataset, batch_size=batch_size, shuffle=shuffle)


def split_dataloader(full_loader, val_split=0.2, batch_size=64, shuffle=True):
    full_dataset = full_loader.dataset

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader


def custom_loss(outputs, targets):
    mu_t, sigma_t, mu_d, sigma_d, log_pi_n, mu_v, sigma_v = (
        outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3],
        outputs[:, 4:132], outputs[:, 132], outputs[:, 133]
    )
    t, d, n, v = targets[:, 0], targets[:, 1], targets[:, 2], targets[:, 3]

    # NLL for time
    nll_t = 0.5 * torch.log(2 * torch.pi * sigma_t**2) + ((t - mu_t)**2) / (2 * sigma_t**2)

    # NLL for duration
    nll_d = 0.5 * torch.log(2 * torch.pi * sigma_d**2) + ((d - mu_d)**2) / (2 * sigma_d**2)

    # NLL for volume
    nll_v = 0.5 * torch.log(2 * torch.pi * sigma_v**2) + ((v - mu_v)**2) / (2 * sigma_v**2)

    # NLL for note value (categorical)
    nll_n = F.cross_entropy(log_pi_n, n.long())
    return nll_t, nll_d, nll_v, nll_n

def calculate_accuracy(predictions, targets):
    logits_n = predictions[:, 4:132]  # Extract logits for the note value
    predicted_notes = logits_n.argmax(dim=1)  # Predicted note indices
    actual_notes = targets[:, 2].long()  # Ground truth note indices
    correct = (predicted_notes == actual_notes).sum().item()  # Count correct predictions
    total = targets.size(0)
    return correct / total


def plot_train_val_metrics(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)

    # Plotting Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.show()

def evaluate_model(dataset_file="data/songs/test.pt", batch_size=64):
    # Load the test dataset
    dataset = torch.load(dataset_file)
    print(f"Loaded test dataset for evaluation from {dataset_file}...")

    # Create DataLoader for the test data
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize and load the model
    model = NoteEventPredictor()
    model.load_state_dict(torch.load("note_prediction_model.pth"))
    model.eval()

    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            context = batch.float()  # Convert the input tensor to float32

            # Check if context has the expected shape
            print(f"Original Context shape: {context.shape}")

            # Ensure context has three dimensions: [batch_size, sequence_length, 4]
            if context.dim() == 2:
                context = context.unsqueeze(0)  # Add a batch dimension

            output = model(context)
            predictions.append(output)

    # Concatenate the predictions
    predictions_tensor = torch.cat(predictions, dim=0)

    # Save the predictions to a file
    torch.save(predictions_tensor, "data/songs/note_predictions.pt")
    print("Predictions saved to note_predictions.pt")


# Training loop

def train_model(model, train_loader, val_loader, epochs=6, patience=5, save_path="note_prediction_model.pth"):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    best_val_loss = float('inf')
    patience_counter = 0

    best_model = model.state_dict()

    train_losses, val_losses = [], []
    total_training_time = 0.0
    total_prediction_time_train = 0.0
    total_prediction_time_val = 0.0
    total_context_windows_train = 0
    total_context_windows_val = 0

    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}/{epochs}")
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        start_time_epoch = time.time()

        for context, target in train_loader:
            optimizer.zero_grad()
            # Measure prediction time for this batch
            start_prediction = time.time()

            predictions = model(context.to(torch.float32))
            total_prediction_time_train += time.time() - start_prediction
            total_context_windows_train += context.size(0)

            loss_t, loss_d, loss_v, loss_n = custom_loss(predictions, target)
            loss = (loss_t + loss_d + loss_v + loss_n).mean()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            print("Train loss in # ", epoch+1, "=", loss.item())
            train_correct += calculate_accuracy(predictions, target) * target.size(0)
            train_total += target.size(0)

        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for context, target in val_loader:
                start_prediction = time.time()
                predictions = model(context.to(torch.float32))
                total_prediction_time_val += time.time() - start_prediction
                total_context_windows_val += context.size(0)

                # loss = custom_loss(predictions, target)
                # val_loss += loss.item()

                loss_t, loss_d, loss_v, loss_n = custom_loss(predictions, target)
                loss = (loss_t + loss_d + loss_v + loss_n).mean()
                val_loss += loss.item()

                print("Validation loss in # ", epoch+1, "=", loss.item())
                val_correct += calculate_accuracy(predictions, target) * target.size(0)
                val_total += target.size(0)

        scheduler.step(val_loss)
        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total

        epoch_time = time.time() - start_time_epoch
        total_training_time += epoch_time

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Printing loss values for the current epoch
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        print(f"  Epoch {epoch + 1} took {epoch_time:.2f} seconds.")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Save the best model
    torch.save(best_model, save_path)
    print(f"Model training complete and saved to {save_path}")

    # Calculate average prediction times
    avg_prediction_time_train = total_prediction_time_train / total_context_windows_train
    avg_prediction_time_val = total_prediction_time_val / total_context_windows_val

    # Plotting the metrics
    plot_train_val_metrics(train_losses, val_losses)

    # Displaying final Training and validation metrics
    summary = {
        "Metric": ["Loss"],
        "Train (Final)": [train_losses[-1]],
        "Validation (Final)": [val_losses[-1]],
    }
    summary_df = pd.DataFrame(summary)
    print(summary_df)

    # Display Hardware and Timing Details
    print("Training Hardware Details:")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Total Training Time: {total_training_time:.2f} seconds")
    print(f"Average Prediction Time (Train): {avg_prediction_time_train:.6f} seconds per context window")
    print(f"Average Prediction Time (Validation): {avg_prediction_time_val:.6f} seconds per context window")

    return model, train_losses, val_losses



# Running all the cells
# Loading the data
root_dir = "data/songs/train"
file_paths = [os.path.join(dp, f) for dp, _, filenames in os.walk(root_dir) for f in filenames if f.endswith('.pt')]

data_loader = get_data_loader(root_dir, batch_size=64, context_size=196, overlapping=True, shuffle=True)

train_loader, val_loader = split_dataloader(data_loader, val_split=0.2, batch_size=64, shuffle=True)

# Initialize the model
model = NoteEventPredictor()
print(model)
print("Model initialized")

# Train and save the model
model, train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=6, patience=5, save_path="note_prediction_model.pth")

dataset_file = "data/songs/test.pt"
evaluate_model(dataset_file=dataset_file, batch_size=64)