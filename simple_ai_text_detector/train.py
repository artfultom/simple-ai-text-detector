from omegaconf import DictConfig
from hydra import initialize, compose
from sklearn.metrics import classification_report, confusion_matrix
import random
import numpy as np
import torch
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import os
from pathlib import Path

from simple_ai_text_detector.data.dataset import TextClassificationDataModule
from simple_ai_text_detector.models.baseline import TfidfLogisticClassifier
from simple_ai_text_detector.models.model import BertTextClassifier


def baseline(total_size: int = None):
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")
        train("baseline", cfg, total_size)


def model(total_size: int = None):
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")
        train("bert", cfg, total_size)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model_name, cfg: DictConfig, total_size):
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(f"text_classification_{model_name}")

    with mlflow.start_run(run_name=f"{model_name}_experiment"):
        set_seed(cfg.random_state)

        mlflow.log_param("model_type", model_name)
        mlflow.log_param("random_state", cfg.random_state)
        mlflow.log_param(
            "total_size", total_size if total_size is not None else cfg.data.total_size
        )
        mlflow.log_param("train_size", cfg.data.train_size)
        mlflow.log_param("val_size", cfg.data.val_size)
        mlflow.log_param("test_size", cfg.data.test_size)

        dm = TextClassificationDataModule(
            data_path=cfg.data.data_path,
            total_size=total_size if total_size is not None else cfg.data.total_size,
            train_size=cfg.data.train_size,
            val_size=cfg.data.val_size,
            test_size=cfg.data.test_size,
            random_state=cfg.random_state,
        )
        dm.setup()

        mlflow.log_metric("train_samples", len(dm.train_dataset))
        mlflow.log_metric("val_samples", len(dm.val_dataset))
        mlflow.log_metric("test_samples", len(dm.test_dataset))

        X_train, y_train = dm.get_texts_labels(dm.train_dataset)
        X_val, y_val = dm.get_texts_labels(dm.val_dataset)
        X_test, y_test = dm.get_texts_labels(dm.test_dataset)

        if model_name == "baseline":
            mlflow.log_param("max_features", cfg.baseline.max_features)
            mlflow.log_param("ngram_range", cfg.baseline.ngram_range)
            mlflow.log_param("max_iter", cfg.baseline.max_iter)
            mlflow.log_param("C", cfg.baseline.C)
            mlflow.log_param("use_preprocessing", cfg.baseline.use_preprocessing)

            model = TfidfLogisticClassifier(
                max_features=cfg.baseline.max_features,
                ngram_range=tuple(cfg.baseline.ngram_range),
                max_iter=cfg.baseline.max_iter,
                C=cfg.baseline.C,
                random_state=cfg.random_state,
                use_preprocessing=cfg.baseline.use_preprocessing,
            )
            model.fit(X_train, y_train)

            mlflow.sklearn.log_model(model.classifier, "model")

        elif model_name == "bert":
            mlflow.log_param("model_name", cfg.model.model_name)
            mlflow.log_param("max_length", cfg.model.max_length)
            mlflow.log_param("batch_size", cfg.model.batch_size)
            mlflow.log_param("learning_rate", cfg.model.learning_rate)
            mlflow.log_param("num_epochs", cfg.model.num_epochs)

            model = BertTextClassifier(
                model_name=cfg.model.model_name,
                max_length=cfg.model.max_length,
                batch_size=cfg.model.batch_size,
                learning_rate=cfg.model.learning_rate,
                num_epochs=cfg.model.num_epochs,
            )

            model.fit(X_train, y_train, X_val, y_val, mlflow_tracking=True)

            model_dir = Path(cfg.model.model_path)
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save(model_dir / "model.pkl")

        print_results(X_train, y_train, X_test, y_test, X_val, y_val, model)


def print_results(X_train, y_train, X_test, y_test, X_val, y_val, model):
    train_metrics = model.evaluate(X_train, y_train)
    for metric, value in train_metrics.items():
        mlflow.log_metric(f"train_{metric}", value)

    val_metrics = model.evaluate(X_val, y_val)
    for metric, value in val_metrics.items():
        mlflow.log_metric(f"val_{metric}", value)

    test_metrics = model.evaluate(X_test, y_test)
    for metric, value in test_metrics.items():
        mlflow.log_metric(f"test_{metric}", value)

    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=["Human", "Generated"])
    cm = confusion_matrix(y_test, y_pred)

    # как пример отчетов
    classification_report_file = "reports/classification_report.txt"
    os.makedirs(os.path.dirname(classification_report_file), exist_ok=True)
    with open(classification_report_file, "w") as f:
        f.write(report)
    mlflow.log_artifact(classification_report_file)

    confusion_matrix_file = "reports/confusion_matrix.txt"
    np.savetxt(confusion_matrix_file, cm, fmt="%d")
    mlflow.log_artifact(confusion_matrix_file)
