import json

from db.models import EvalResult
from db.session import SessionLocal


def log_eval_results(training_run_id: int, results_path: str, key: str, model_variant: str) -> None:
    """
    Read a results JSON file and insert one EvalResult row per prompt,
    tagged with the given model variant and associated training run.
    """
    with open(results_path) as f:
        data = json.load(f)

    entries = data[key]

    session = SessionLocal()
    try:
        for entry in entries:
            eval_result = EvalResult(
                training_run_id=training_run_id,
                checkpoint_step=None,
                prompt=entry["prompt"],
                clip_i=entry["clip_i"],
                clip_t=entry["clip_t"],
                model_variant=model_variant,
            )
            session.add(eval_result)
        session.commit()
        print(f"Logged {len(entries)} eval results for variant '{model_variant}' (run_id={training_run_id})")
    finally:
        session.close()


if __name__ == "__main__":
    # base and finetuned (rank16) came from the same eval run, attributed to run_id=1
    log_eval_results(1, "evaluation/results.json", "base", "base")
    log_eval_results(1, "evaluation/results.json", "finetuned", "finetuned_rank16")

    # rank32 results, attributed to run_id=2
    log_eval_results(2, "evaluation/results_rank32.json", "rank32", "finetuned_rank32")

    # knn baseline isn't tied to a specific training run, but the schema
    # requires a training_run_id — attach it to run_id=1 (rank16) since
    # that's the "primary" comparison run
    log_eval_results(1, "evaluation/results_knn.json", "knn_baseline", "knn_baseline")