"""Build explicit analysis inputs and reference labels from a completed run."""
import argparse
import json
from pathlib import Path

from .artifacts import manifest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    config = json.loads((run / "run.json").read_text())
    inputs, references = [], []
    for path in sorted(run.glob("episode_*/episode.json")):
        result = json.loads(path.read_text())
        # Infrastructure errors and diagnostic motion are not model-evaluation samples.
        if result["status"] not in ("success", "task_failure"):
            continue
        episode = path.parent
        inputs.append({"run_id": result["run_id"], "episode_id": result["episode_id"],
                       "source": "simulation", "instruction": result["instruction"],
                       "videos": [str(p.relative_to(run)) for p in sorted(episode.glob("*.mp4"))],
                       "question": "Was the task completed? Identify the failure stage if visible; otherwise report uncertainty."})
        references.append({"run_id": result["run_id"], "episode_id": result["episode_id"],
                           "simulator_success": result["status"] == "success",
                           "human_failure_stage": None, "human_assessment": None,
                           "subtasks": str((episode / "subtasks.jsonl").relative_to(run))})
    write_json(run / "analysis_inputs.json", inputs)
    write_json(run / "analysis_references.json", references)
    write_json(run / "handoff.json", {"run_id": config["run_id"], "environment": config["environment"],
                                     "analysis_samples": len(inputs),
                                     "reference_labels_are_not_model_inputs": True})
    manifest(run)
    print(f"Exported {len(inputs)} evaluation samples with checksums: {run}")


if __name__ == "__main__":
    main()
