import pandas as pd
import numpy as np
import os
from scripts.preprocessing.dmrir_data_loader import DMRIRDataset 
from torch.utils.data import DataLoader
import random 
from pathlib import Path
import torch
import vicreg.resnet as resnet
import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import wandb
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
ROOT_DIR = Path(os.getenv("ROOT_DIR"))

lab_db_path = ROOT_DIR / "data" / "dmrir" / "lab_database" / "database"
label_1_path = "abnormal/"
label_0_path = "normal/"

batch_size = 64

def shuffle_data(img_paths, labels, ):
    data = list(zip(img_paths, labels))
    random.shuffle(data)
    img_paths_shuffled, labels_shuffled = zip(*data)
    return list(img_paths_shuffled), list(labels_shuffled)

def split_data(paths, labels, train_size=0.7, val_size=0.15):
    n = len(paths)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    
    train_paths = paths[:train_end]
    train_labels = labels[:train_end]
    
    val_paths = paths[train_end:val_end]
    val_labels = labels[train_end:val_end]
    
    test_paths = paths[val_end:]
    test_labels = labels[val_end:]
    
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

files_1 = os.listdir(os.path.join(lab_db_path, label_1_path))
files_0 = os.listdir(os.path.join(lab_db_path, label_0_path))
paths_images = [os.path.join(lab_db_path, label_1_path, f) for f in files_1] + \
                [os.path.join(lab_db_path, label_0_path, f) for f in files_0]

labels = [1] * len(files_1) + [0] * len(files_0)
print(len(labels), len(paths_images))
print(labels[:10])

paths_images, labels = shuffle_data(paths_images, labels)
print(len(labels), len(paths_images))   
print(labels[:10])

train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = split_data(paths_images, labels)
print("full dataset size:", len(paths_images))
print("Train size:", len(train_paths))
print("Validation size:", len(val_paths))
print("Test size:", len(test_paths))

print("Train labels 1:", sum(train_labels), "0:", len(train_labels) - sum(train_labels))
print("Validation labels 1:", sum(val_labels), "0:", len(val_labels) - sum(val_labels))
print("Test labels 1:", sum(test_labels), "0:", len(test_labels) - sum(test_labels))

train = DMRIRDataset(img_paths=train_paths, labels=train_labels, transform=None)
print(f"Numero di campioni nel dataset: {len(train)}")
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)

test = DMRIRDataset(img_paths=test_paths, labels=test_labels, transform=None)
print(f"Numero di campioni nel dataset di test: {len(test)}")
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

val = DMRIRDataset(img_paths=val_paths, labels=val_labels, transform=None)
print(f"Numero di campioni nel dataset di validation: {len(val)}")
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)

ckpt_path = Path("vicreg/outputs_cluster1/best_resnet50.pth")
backbone, _ = resnet.__dict__["resnet50"](zero_init_residual=True)
backbone.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
backbone.eval()

backbone.to(device)

class ClassificationHead(nn.Module):
    def __init__(self, backbone):
        super(ClassificationHead, self).__init__()
        self.backbone = backbone
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
    
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, train_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
model = ClassificationHead(backbone)
model.to(device)

for param in model.backbone.parameters():
    param.requires_grad = False


for param in model.fc.parameters():
    param.requires_grad = True

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.fc.parameters(), lr=1e-5)
early_stopping = EarlyStopping(patience=5, delta=0.001)

wandb.init(
    project="DownStreamDMRIR",
    name="prova_fulldf_1_only_class_layer",
    config={
        "learning_rate": 1e-5,
        "batch_size": 64,
        "epochs": 100,
        "model": "ResNet50 with Classification Head"
    }
)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
f1_scores = []
precisions = []
recalls = []
aucs = []

num_epochs = 200 
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    all_preds = []
    all_labels = []
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        if labels.dim() == 1:
            labels = labels.unsqueeze(1).float()
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(1)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        preds = (outputs > 0.0).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        loss.backward()
        optimizer.step()

    avg_train_loss = running_loss / len(train_loader)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    train_losses.append(avg_train_loss)
    f1_scores.append(f1)
    precisions.append(precision)
    recalls.append(recall)
    aucs.append(auc)
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            running_loss += loss.item()
            preds = (outputs > 0.0).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        avg_val_loss = running_loss / len(val_loader)
        f1 = f1_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_preds)
        val_losses.append(avg_val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} - Validation Loss: {avg_val_loss:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}")
    wandb.log({
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "val_f1": f1,
        "val_precision": precision,
        "val_recall": recall,
        "val_auc": auc,
        "epoch": epoch + 1
    })
    if early_stopping(avg_val_loss, avg_train_loss):
        print("Early stopping triggered. Stopping training.")
        break

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())
        running_loss += loss.item()
        preds = (outputs.squeeze() > 0.0).float() 
        acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())  
        running_accuracy += acc
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

f1 = f1_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_preds)

print(f"Test F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}")

print("Classification Report (Test Set):")
print(classification_report(all_labels, all_preds, target_names=["no_cancro", "cancro"]))
