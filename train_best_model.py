# train_best_model.py

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# --- Import the model definition ---
# Make sure AlexNet_evolvable.py is in the same directory
from AlexNet_evolvable import Net

# --- Define the Champion Chromosome ---
# COPY the best chromosome from your GA output here:
BEST_CHROMOSOME = {
    'conv1_filters': 96,
    'conv1_kernel': 9,
    'conv1_stride': 3,
    'conv2_filters': 192,
    'conv2_kernel': 3,
    'conv3_filters': 384,
    'conv4_filters': 384,
    'conv5_filters': 256,
    'fc1_neurons': 2048,
    'fc2_neurons': 4096,
    'lr': 0.01,
    'momentum': 0.95,
    'dropout': 0.4
}

# --- Training Parameters ---
FINAL_NUM_EPOCHS = 50  # <<< Increase this significantly >>>
BATCH_SIZE = 128
CHECKPOINT_DIR = "./final_model_checkpoints"
FINAL_MODEL_PATH = "./best_model_final.pth"

# Create checkpoint directory if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Loading (Full Train + Test Sets) ---
print("Loading CIFAR-10 dataset...")
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(), # Add common augmentation
    transforms.RandomCrop(32, padding=4), # Add common augmentation
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load full training set (for final training)
full_train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = DataLoader(full_train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# Load test set (for final evaluation)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Input/Output shapes for CIFAR-10
in_shape = (3, 32, 32)
out_shape = (10,)

# --- Create and Setup the Model ---
print("Creating the best model from the champion chromosome...")
model = Net(in_shape, out_shape, BEST_CHROMOSOME, device)
model.train_setup(prm=BEST_CHROMOSOME) # Setup optimizer using the chromosome's hyperparams

# --- Training Loop ---
print(f"Starting final training for {FINAL_NUM_EPOCHS} epochs...")
best_test_acc = 0.0
start_epoch = 0

# Optional: Load from a previous final training checkpoint if it exists
checkpoint_path = os.path.join(CHECKPOINT_DIR, "final_training_checkpoint.pth")
if os.path.exists(checkpoint_path):
    print(f"Resuming training from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # Assuming optimizer state is also saved if needed, load it too
    # model.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # Uncomment if optimizer state was saved
    start_epoch = checkpoint['epoch']
    best_test_acc = checkpoint.get('best_test_acc', 0.0) # Get previous best accuracy


for epoch in range(start_epoch, FINAL_NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{FINAL_NUM_EPOCHS}")
    model.train() # Set to training mode
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc='Training')
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        model.optimizer.zero_grad()
        outputs = model(inputs)
        loss = model.criteria[0](outputs, labels) # Use CrossEntropyLoss from criteria
        loss.backward()
        # Optional: Gradient clipping if used during evolution
        # nn.utils.clip_grad_norm_(model.parameters(), 3)
        model.optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

    # --- Evaluate on Test Set ---
    print("Evaluating on test set...")
    model.eval() # Set to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    print(f"Test Accuracy after Epoch {epoch+1}: {test_acc:.2f}%")

    # --- Save Checkpoint (and potentially the best model) ---
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    checkpoint_data = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': model.optimizer.state_dict(), # Uncomment if saving optimizer state
        'best_test_acc': best_test_acc,
         'test_acc': test_acc
    }
    torch.save(checkpoint_data, os.path.join(CHECKPOINT_DIR, "final_training_checkpoint.pth"))

    # Save the best model based on test accuracy
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), FINAL_MODEL_PATH)
        print(f"*** New best test accuracy: {best_test_acc:.2f}%. Model saved to {FINAL_MODEL_PATH} ***")

print("\n--- Final Training Complete ---")
print(f"Best Test Accuracy Achieved: {best_test_acc:.2f}%")
print(f"Final model saved to: {FINAL_MODEL_PATH}")
