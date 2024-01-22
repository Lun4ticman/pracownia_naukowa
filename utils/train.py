import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10, patience=3, checkpoint_path='checkpoint.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Initialize early stopping variables
    best_valid_loss = float('inf')
    epochs_without_improvement = 0

    # Initialize Tensorboard writer
    writer = SummaryWriter()


    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        with tqdm(train_loader, unit='batch') as pbar:
            for idx, (inputs, labels) in enumerate(pbar):
                pbar.set_description(f'Epoch...{epoch}/{num_epochs}')
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                running_loss = train_loss/(idx+1)

                pbar.set_postfix(loss=running_loss)

                # print(f'\rEpoch {epoch}, Running loss:{show_loss:.4f}', end="")

            # Calculate average training loss
            avg_train_loss = train_loss / len(train_loader)

            # Validation
            model.eval()
            valid_loss = 0.0

            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    valid_loss += loss.item()

            # Calculate average validation loss
            avg_valid_loss = valid_loss / len(valid_loader)

            # Log losses to Tensorboard
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('Loss/Validation', avg_valid_loss, epoch)

            # Check for early stopping
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                epochs_without_improvement = 0
                # Save the model checkpoint
                torch.save(model.state_dict(), checkpoint_path)
            else:
                epochs_without_improvement += 1

            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}')

            # Check for early stopping
            if epochs_without_improvement >= patience:
                print(f'Early stopping after {epoch+1} epochs without improvement.')
                break

    writer.close()

def test(model, dataloader, device='cuda'):
    # Set the model to evaluation mode
    model.eval()

    correct_predictions = 0
    total_samples = 0

    model = model.to(device)

    with torch.no_grad():
        for inputs, labels in dataloader:
            # Convert inputs and labels to torch tensors if they are not already
            inputs = inputs.float()
            labels = labels.long()

            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Calculate accuracy
            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = correct_predictions / total_samples
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
