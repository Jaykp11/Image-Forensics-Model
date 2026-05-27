import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
torch.set_num_threads(2)

import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageChops, ImageEnhance
import pandas as pd
from sklearn.model_selection import train_test_split
import io

# ADDED METRICS IMPORTS
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ELA FUNCTION
def convert_to_ela(image_path, quality=90):
    original = Image.open(image_path).convert("RGB")

    buffer = io.BytesIO()
    original.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)

    compressed = Image.open(buffer)
    ela_image = ImageChops.difference(original, compressed)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])

    if max_diff == 0:
        max_diff = 1

    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image

# LOAD DATA
df = pd.read_csv("dataset.csv")

train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=2/3,
    stratify=temp_df["label"],
    random_state=42
)

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

# DATASET CLASS
class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["image_path"]
        label = self.df.iloc[idx]["label"]

        image = convert_to_ela(img_path)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

# TRANSFORMS
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# DATALOADERS

train_dataset = ImageDataset(train_df, train_transform)
val_dataset = ImageDataset(val_df, val_test_transform)
test_dataset = ImageDataset(test_df, val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

# MODEL
model = models.resnet18(weights="IMAGENET1K_V1")

for param in model.parameters():
    param.requires_grad = False

for param in model.layer3.parameters():
    param.requires_grad = True

for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 2)
)

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# LOSS + OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.3
)

# TRAINING + VALIDATION
EPOCHS = 25

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total

    # VALIDATION
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total

    scheduler.step()

    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Acc: {train_acc:.2f}%")
    print(f"Validation Acc: {val_acc:.2f}%")
    print("-"*40)

# TESTING
model.eval()
correct = 0
total = 0

# ADDED FOR METRICS
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # STORE LABELS AND PREDICTIONS
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

test_accuracy = 100 * correct / total
print(f"\n🔥 Final Test Accuracy: {test_accuracy:.2f}%")

# PRECISION
precision = precision_score(
    all_labels,
    all_predictions,
    average='weighted'
)

# RECALL
recall = recall_score(
    all_labels,
    all_predictions,
    average='weighted'
)

# F1 SCORE
f1 = f1_score(
    all_labels,
    all_predictions,
    average='weighted'
)

# CONFUSION MATRIX
cm = confusion_matrix(all_labels, all_predictions)

# PRINT METRICS
print("\n📊 PERFORMANCE METRICS")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# CLASSIFICATION REPORT
print("\n📋 CLASSIFICATION REPORT")
print(classification_report(all_labels, all_predictions))

# CONFUSION MATRIX OUTPUT
print("\n🧩 CONFUSION MATRIX")
print(cm)

# SAVE MODEL
model_path = "image_forensics_model.pth"
torch.save(model.state_dict(), model_path)

print(f"\n✅ Model saved successfully at: {model_path}")