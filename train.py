import torch
import torch.nn as nn
import torch.optim as optim
import os
import tqdm
import numpy as np

from model_arch import LSTMModel
# Initialize LSTM model parameters
input_size = 10  # Example: 10 features for time-series data
hidden_size = 50
num_layers = 2
output_size = 1  # Predicting one pollutant level

# Instantiate the model
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping and checkpoint saving
best_loss = float('inf')
early_stopping_patience = 5
patience_counter = 0
checkpoint_path = "best_model.pth"

# Example training loop

def train_lstm(model, train_loader, num_epochs=10):
    global best_loss, patience_counter
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, (inputs, targets) in progress_bar:
            # Move data to the appropriate device
            inputs, targets = inputs.to(torch.float32), targets.to(torch.float32)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            epoch_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())

        # Calculate average loss for the epoch
        epoch_loss /= len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss:.4f}")

        # Early stopping and checkpoint saving
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model with loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}/{early_stopping_patience}")

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break
