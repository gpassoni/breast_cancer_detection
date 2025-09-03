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
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

import os 
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
ROOT_DIR = Path(os.getenv("ROOT_DIR"))
print("Root dir:", ROOT_DIR)


lab_db_path = ROOT_DIR / "data/dmrir/lab_database/database/"
label_1_path = "abnormal/"
label_0_path = "normal/"

def shuffle_data(img_paths, labels, ):
    data = list(zip(img_paths, labels))
    random.shuffle(data)
    img_paths_shuffled, labels_shuffled = zip(*data)
    return list(img_paths_shuffled), list(labels_shuffled)

def split_data(paths, labels, train_size=0.7, val_size=0.15, max_size=1000):
    if len(paths) > max_size:
        paths = paths[:max_size]
        labels = labels[:max_size]

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

files_1 = os.listdir(os.path.join(lab_db_path, label_1_path))
files_0 = os.listdir(os.path.join(lab_db_path, label_0_path))
paths_images = [os.path.join(lab_db_path, label_1_path, f) for f in files_1] + \
                [os.path.join(lab_db_path, label_0_path, f) for f in files_0]
labels = [1] * len(files_1) + [0] * len(files_0)

train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(paths_images, labels, test_size=0.3, random_state=42, stratify=labels)

columns = ["n_samples", "cv_fold", "f1", "precision", "recall", "auc"]
df_data_test_results = pd.DataFrame(columns=columns)

trials = [64, 128, 192, 256, 352, 416, 512]
cv = 5
skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

for n_samples in trials:
    for i, (train_index, val_index) in enumerate(skf.split(train_val_paths, train_val_labels)):
        train_paths = [train_val_paths[idx] for idx in train_index]
        train_labels = [train_val_labels[idx] for idx in train_index]
        val_paths = [train_val_paths[idx] for idx in val_index]
        val_labels = [train_val_labels[idx] for idx in val_index]
        
        print(f"Fold {i+1}/{cv} - Train size: {len(train_paths)}, Validation size: {len(val_paths)}, Test size: {len(test_paths)}")

        batch_size = 64 if n_samples > 64 else n_samples

        train_paths_temp = train_paths[:n_samples]
        train_labels_temp = train_labels[:n_samples]

        val_paths = val_paths[:100]
        val_labels = val_labels[:100]

        rateo_labels = sum(train_labels_temp) / len(train_labels_temp)
        print("Rateo labels:", rateo_labels)

        while rateo_labels < 0.2 or rateo_labels > 0.8:
            print("Rateo labels fuori dal range 0.2-0.8, rifacimento dello shuffle dei dati...")
            train_paths, train_labels = shuffle_data(train_paths, train_labels)
            train_paths_temp = train_paths[:n_samples]
            train_labels_temp = train_labels[:n_samples]
            rateo_labels = sum(train_labels_temp) / len(train_labels_temp)
            print("Rateo labels:", rateo_labels)

        train_paths, train_labels = train_paths_temp, train_labels_temp

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

        model = ClassificationHead(backbone)
        model.to(device)

        for param in model.backbone.parameters():
            param.requires_grad = False


        for param in model.fc.parameters():
            param.requires_grad = True

        criterion = nn.BCEWithLogitsLoss()

        optimizer = optim.Adam(model.fc.parameters(), lr=1e-5)
        early_stopping = EarlyStopping(patience=10, delta=0.001)

        wandb.init(
            project="DownStreamDMRIR_data_study_V2",
            name=f"DMRIR_Classification_{n_samples}_samples_cv_{i}_fulltest",
            config={
                "learning_rate": 1e-5,
                "batch_size": 64,
                "epochs": 300,
                "model": "ResNet50 with Classification Head"
            },
            reinit=True
        )

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        f1_scores = []
        precisions = []
        recalls = []
        aucs = []

        num_epochs = 300 
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
        try:
            print(classification_report(all_labels, all_preds, target_names=["no_cancro", "cancro"]))
        except ValueError as e:
            print(f"Error in classification report: {e}")
            print("This may be due to one of the classes not being present in the test set.")

        new_row = pd.DataFrame({
            "n_samples": [n_samples],
            "cv_fold": [i],
            "f1": [f1],
            "precision": [precision],
            "recall": [recall],
            "auc": [auc]
        })
        df_data_test_results = pd.concat([df_data_test_results, new_row], ignore_index=True)
        wandb.finish()

        df_data_test_results.to_csv("classification_fulltestset_resultsV2.csv", index=False)
