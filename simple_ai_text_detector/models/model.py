import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import mlflow


class BertTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


class BertClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_classes=2, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


class BertTextClassifier:
    def __init__(
        self,
        model_name="bert-base-uncased",
        max_length=128,
        batch_size=32,
        learning_rate=2e-5,
        num_epochs=1,
        dropout=0.3,
        device=None,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.dropout = dropout

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"Используется устройство: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None

    def _create_dataloader(self, texts, labels, shuffle=False):
        dataset = BertTextDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=0
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None, mlflow_tracking=False):
        self.model = BertClassifier(
            model_name=self.model_name, dropout=self.dropout
        ).to(self.device)

        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            pbar = tqdm(train_loader, desc=f"Эпоха {epoch + 1}/{self.num_epochs}")

            for batch_idx, batch in enumerate(pbar):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()

                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{100 * correct / total:.2f}%",
                    }
                )

                if mlflow_tracking:
                    step = epoch * len(train_loader) + batch_idx
                    mlflow.log_metric("batch_loss", loss.item(), step=step)
                    mlflow.log_metric(
                        "batch_accuracy", 100 * correct / total, step=step
                    )

            avg_loss = total_loss / len(train_loader)
            train_acc = 100 * correct / total

            if mlflow_tracking:
                mlflow.log_metric("epoch_loss", avg_loss, step=epoch)
                mlflow.log_metric("epoch_accuracy", train_acc, step=epoch)

            if X_val is not None and y_val is not None:
                val_metrics = self.evaluate(X_val, y_val)

                if mlflow_tracking:
                    mlflow.log_metric(
                        "epoch_val_accuracy", val_metrics["accuracy"], step=epoch
                    )
                    mlflow.log_metric(
                        "epoch_val_precision", val_metrics["precision"], step=epoch
                    )
                    mlflow.log_metric(
                        "epoch_val_recall", val_metrics["recall"], step=epoch
                    )
                    mlflow.log_metric("epoch_val_f1", val_metrics["f1"], step=epoch)

    def predict(self, X):
        self.model.eval()
        predictions = []

        data_loader = self._create_dataloader(X, [0] * len(X))

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Предсказание"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                _, predicted = torch.max(logits, 1)

                predictions.extend(predicted.cpu().numpy())

        return np.array(predictions)

    def predict_proba(self, X):
        self.model.eval()
        probabilities = []

        data_loader = self._create_dataloader(X, [0] * len(X))

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Предсказание вероятностей"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)

                probabilities.extend(probs.cpu().numpy())

        return np.array(probabilities)

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary"
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def save(self, path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_name": self.model_name,
                "max_length": self.max_length,
            },
            path,
        )
        print(f"Модель сохранена в {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        self.model_name = checkpoint["model_name"]
        self.max_length = checkpoint["max_length"]

        self.model = BertClassifier(
            model_name=self.model_name, dropout=self.dropout
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        print(f"Модель загружена из {path}")
