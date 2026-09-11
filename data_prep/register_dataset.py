from db.session import SessionLocal
from db.models import Dataset


def register_dataset(subject_name: str, source_images_path: str, num_images: int) -> int:
    """
    Insert a record for a prepared dataset into the datasets table and
    return the new record's id.
    """
    session = SessionLocal()
    try:
        dataset = Dataset(
            subject_name=subject_name,
            source_images_path=source_images_path,
            num_images=num_images,
        )
        session.add(dataset)
        session.commit()
        session.refresh(dataset)
        return dataset.id
    finally:
        session.close()


if __name__ == "__main__":
    dataset_id = register_dataset(
        subject_name="colorful_sneaker",
        source_images_path="data/raw/dataset/colorful_sneaker",
        num_images=5,
    )
    print(f"Registered dataset with id: {dataset_id}")