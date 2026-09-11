# evaluation/register_training_runs.py
from db.session import SessionLocal
from db.models import TrainingRun, Dataset

session = SessionLocal()
dataset = session.query(Dataset).filter_by(subject_name="colorful_sneaker").first()

run_rank16 = TrainingRun(
    dataset_id=dataset.id,
    lora_rank=16,
    steps=1000,
    learning_rate=1e-4,
    checkpoint_path="training/checkpoints/pytorch_lora_weights.safetensors",
)
run_rank32 = TrainingRun(
    dataset_id=dataset.id,
    lora_rank=32,
    steps=1000,
    learning_rate=1e-4,
    checkpoint_path="checkpoints/lora_colorful_sneaker_rank32/pytorch_lora_weights.safetensors",
)
session.add_all([run_rank16, run_rank32])
session.commit()
print(f"Registered training run rank16 with id={run_rank16.id}")
print(f"Registered training run rank32 with id={run_rank32.id}")
session.close()