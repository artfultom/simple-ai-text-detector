import fire

from simple_ai_text_detector.infer import infer
from simple_ai_text_detector.train import train

if __name__ == "__main__":
    fire.Fire(
        {
            "train": train,
            "infer": infer,
        }
    )
