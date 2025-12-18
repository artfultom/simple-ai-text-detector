from pathlib import Path

from hydra import initialize, compose
from hydra.utils import to_absolute_path

from kaggle.api.kaggle_api_extended import KaggleApi


def download_data():
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="config")

    dataset = cfg.download_data.dataset
    output_dir = Path(to_absolute_path(cfg.download_data.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(dataset, path=str(output_dir), unzip=True)

    print(f"Dataset downloaded to {output_dir}")
