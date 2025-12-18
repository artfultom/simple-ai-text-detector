import fire

from simple_ai_text_detector.data.download_data import download_data
from simple_ai_text_detector.train import baseline, model

if __name__ == "__main__":
    fire.Fire(
        {
            "download_data": download_data,
            "baseline": baseline,
            "model": model,
        }
    )
