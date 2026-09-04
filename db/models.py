from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from db.session import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True)
    subject_name = Column(String, nullable=False)
    source_images_path = Column(String, nullable=False)
    num_images = Column(Integer)
    created_at = Column(DateTime, server_default=func.now())

    training_runs = relationship("TrainingRun", back_populates="dataset")


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    lora_rank = Column(Integer)
    steps = Column(Integer)
    learning_rate = Column(Float)
    checkpoint_path = Column(String)
    started_at = Column(DateTime, server_default=func.now())
    completed_at = Column(DateTime, nullable=True)

    dataset = relationship("Dataset", back_populates="training_runs")
    eval_results = relationship("EvalResult", back_populates="training_run")


class EvalResult(Base):
    __tablename__ = "eval_results"

    id = Column(Integer, primary_key=True)
    training_run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=False)
    checkpoint_step = Column(Integer)
    prompt = Column(String)
    clip_i = Column(Float)
    clip_t = Column(Float)
    dino_score = Column(Float, nullable=True)
    model_variant = Column(String)  # 'base', 'finetuned', 'knn_baseline'
    created_at = Column(DateTime, server_default=func.now())

    training_run = relationship("TrainingRun", back_populates="eval_results")


class InferenceRequest(Base):
    __tablename__ = "inference_requests"

    id = Column(Integer, primary_key=True)
    prompt = Column(String)
    model_version = Column(String)
    latency_ms = Column(Float)
    status = Column(String)
    created_at = Column(DateTime, server_default=func.now())