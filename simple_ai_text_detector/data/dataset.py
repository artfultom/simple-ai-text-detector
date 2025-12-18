import lightning as L
import numpy as np
import pandas as pd


class TextDataset:
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": str(self.texts[idx]), "label": int(self.labels[idx])}


class TextClassificationDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: str = None,
        df: pd.DataFrame = None,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        total_size: int = None,
        random_state: int = 42,
    ):
        super().__init__()
        self.data_path = data_path
        self.df = df
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.total_size = total_size
        self.random_state = random_state
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        self.df = pd.read_csv(self.data_path, nrows=self.total_size)

    def setup(self, stage=None):
        if self.df is None:
            self.prepare_data()

        df_shuffled = self.df.sample(
            frac=1, random_state=self.random_state
        ).reset_index(drop=True)

        n = len(df_shuffled)
        train_end = int(n * self.train_size)
        val_end = int(n * (self.train_size + self.val_size))

        train_df = df_shuffled[:train_end]
        val_df = df_shuffled[train_end:val_end]
        test_df = df_shuffled[val_end:]

        self.train_dataset = TextDataset(
            train_df["text"].values, train_df["generated"].values
        )
        self.val_dataset = TextDataset(
            val_df["text"].values, val_df["generated"].values
        )
        self.test_dataset = TextDataset(
            test_df["text"].values, test_df["generated"].values
        )

    def get_texts_labels(self, dataset):
        texts = [dataset[i]["text"] for i in range(len(dataset))]
        labels = [dataset[i]["label"] for i in range(len(dataset))]
        return texts, np.array(labels)
