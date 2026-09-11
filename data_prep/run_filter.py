import os
from data_prep.filters import is_sharp_enough


def filter_directory(input_dir: str, threshold: float = 100.0) -> list[str]:
    """
    Return the list of image paths in input_dir that pass the sharpness
    threshold.
    """
    accepted = []
    for filename in os.listdir(input_dir):
        path = os.path.join(input_dir, filename)
        if is_sharp_enough(path, threshold=threshold):
            accepted.append(path)
    return accepted


if __name__ == "__main__":
    result = filter_directory("data/raw/dataset/colorful_sneaker")
    print(f"{len(result)} images passed the sharpness filter")
    for path in result:
        print(path)