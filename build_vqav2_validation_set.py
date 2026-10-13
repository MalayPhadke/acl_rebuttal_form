"""Build the fixed, blinded cohort used by experiment_app_vqav2.py.

The 100 trials follow three strict selection rules:

* 10 pinned questions that all five models answered incorrectly. These are
  drawn from the 13 unanimous failures in the form's original cohort.
* 8 questions per model (40 total) where that model alone was incorrect and
  the other four models were correct.
* 50 questions that all five models answered correctly.

Each 50-question part contains 25 incorrect and 25 correct trials. Images are
copied beside the app so deployment does not depend on experiment directories.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = APP_DIR / "vqav2_validation_100.csv"
OUTPUT_IMAGES = APP_DIR / "vqav2_validation_images"
SEED = 20261013

SHARED_ROOT = Path(
    "/home/debarpanb/acl/endpoint_amb_probe_488/matched_external_70_30"
)
MODEL_SOURCES = {
    "qwenvl": SHARED_ROOT / "features/vqav2/metadata_with_labels.csv",
    "gemma3": SHARED_ROOT
    / "multimodel/gemma3/vqav2/metadata_with_labels.csv",
    "llava": SHARED_ROOT / "multimodel/llava/vqav2/metadata_with_labels.csv",
    "phi4": SHARED_ROOT / "multimodel/phi4/vqav2/metadata_with_labels.csv",
    "pixtral": SHARED_ROOT
    / "multimodel/pixtral/vqav2/metadata_with_labels.csv",
}
MODELS = tuple(MODEL_SOURCES)

MODEL_DISPLAY_NAMES = {
    "qwenvl": "Qwen2.5-VL",
    "gemma3": "Gemma-3",
    "llava": "LLaVA-v1.6-Mistral-7B",
    "phi4": "Phi-4 Vision",
    "pixtral": "Pixtral-12B",
}

# Ten of the 13 unanimous failures found in the form's original cohort. Their
# IDs are pinned so rebuilding the cohort cannot silently replace them.
PINNED_UNANIMOUS_INCORRECT_IDS = (
    "154590000",
    "241174003",
    "267725003",
    "137085000",
    "548500000",
    "380842002",
    "578786005",
    "268375002",
    "382617001",
    "573134004",
)

BASE_OUTPUT_FIELDS = [
    "trial_uid",
    "part",
    "selection_group",
    "target_model",
    "allocation_model",
    "source_correctness",
    "question_id",
    "image_id",
    "question",
    "reference_answer",
    "source_split",
    "image_path",
]
MODEL_OUTPUT_FIELDS = [
    field
    for model in MODELS
    for field in (f"{model}_prediction", f"{model}_correctness")
]
OUTPUT_FIELDS = BASE_OUTPUT_FIELDS + MODEL_OUTPUT_FIELDS


def _read_index(path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    index = {row["sample_id"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError(f"Duplicate sample_id values in {path}")
    return index


def _load_shared_rows():
    by_model = {
        model: _read_index(path) for model, path in MODEL_SOURCES.items()
    }
    shared_ids = set.intersection(*(set(index) for index in by_model.values()))
    if len(shared_ids) != 3800:
        raise ValueError(
            f"Expected 3,800 shared VQAv2 samples, found {len(shared_ids)}"
        )
    return by_model, shared_ids


def _correctness(by_model, question_id, model):
    return int(by_model[model][question_id]["correctness_label"])


def _selected_row(
    by_model,
    question_id,
    selection_group,
    part,
    target_model="",
    allocation_model="",
):
    base = by_model["qwenvl"][question_id]
    image_paths = json.loads(base["image_paths_json"])
    if len(image_paths) != 1:
        raise ValueError(
            f"Expected one image for VQAv2 sample {question_id}, "
            f"found {len(image_paths)}"
        )

    source_correctness = "1" if selection_group == "unanimous_correct" else "0"
    row = {
        "trial_uid": f"vqav2:{question_id}",
        "part": str(part),
        "selection_group": selection_group,
        "target_model": target_model,
        "allocation_model": allocation_model,
        "source_correctness": source_correctness,
        "question_id": question_id,
        "image_id": base["group_id"],
        "question": base["question"],
        "reference_answer": base["gold_answer"],
        "source_split": base.get("outer_split", ""),
        "source_image": image_paths[0],
    }
    for model in MODELS:
        model_row = by_model[model][question_id]
        row[f"{model}_prediction"] = model_row["prediction"]
        row[f"{model}_correctness"] = model_row["correctness_label"]
    return row


def _sample_cohort():
    rng = random.Random(SEED)
    by_model, shared_ids = _load_shared_rows()
    used_ids = set()
    selected = []

    # Five pinned unanimous failures in each part.
    for index, question_id in enumerate(PINNED_UNANIMOUS_INCORRECT_IDS):
        if question_id not in shared_ids:
            raise ValueError(f"Pinned sample is missing: {question_id}")
        labels = [_correctness(by_model, question_id, model) for model in MODELS]
        if labels != [0] * len(MODELS):
            raise ValueError(
                f"Pinned sample {question_id} is no longer unanimously incorrect: "
                f"{labels}"
            )
        part = 1 if index < 5 else 2
        selected.append(
            _selected_row(
                by_model,
                question_id,
                selection_group="unanimous_incorrect",
                part=part,
            )
        )
        used_ids.add(question_id)

    # Eight strict model-specific failures per model: the target model is wrong
    # and every other model is correct. Split four into each part.
    for target_model in MODELS:
        candidates = [
            question_id
            for question_id in sorted(shared_ids)
            if question_id not in used_ids
            and _correctness(by_model, question_id, target_model) == 0
            and all(
                _correctness(by_model, question_id, other_model) == 1
                for other_model in MODELS
                if other_model != target_model
            )
        ]
        rng.shuffle(candidates)
        chosen = candidates[:8]
        if len(chosen) != 8:
            raise RuntimeError(
                f"Could only select {len(chosen)} strict failures for {target_model}"
            )
        for index, question_id in enumerate(chosen):
            part = 1 if index < 4 else 2
            selected.append(
                _selected_row(
                    by_model,
                    question_id,
                    selection_group="model_specific_incorrect",
                    part=part,
                    target_model=target_model,
                )
            )
            used_ids.add(question_id)

    # Fifty unanimous successes. allocation_model provides an even 10/model
    # bookkeeping split without changing their all-model-correct status.
    correct_candidates = [
        question_id
        for question_id in sorted(shared_ids)
        if question_id not in used_ids
        and all(
            _correctness(by_model, question_id, model) == 1 for model in MODELS
        )
    ]
    rng.shuffle(correct_candidates)
    chosen_correct = correct_candidates[:50]
    if len(chosen_correct) != 50:
        raise RuntimeError(
            f"Could only select {len(chosen_correct)} unanimous correct samples"
        )

    for model_index, allocation_model in enumerate(MODELS):
        model_rows = chosen_correct[model_index * 10 : (model_index + 1) * 10]
        for index, question_id in enumerate(model_rows):
            part = 1 if index < 5 else 2
            selected.append(
                _selected_row(
                    by_model,
                    question_id,
                    selection_group="unanimous_correct",
                    part=part,
                    allocation_model=allocation_model,
                )
            )
            used_ids.add(question_id)

    if len(selected) != 100 or len(used_ids) != 100:
        raise AssertionError("The generated cohort must contain 100 unique samples")

    for part in ("1", "2"):
        part_rows = [row for row in selected if row["part"] == part]
        rng.shuffle(part_rows)
        for order, row in enumerate(part_rows, start=1):
            row["part_order"] = order

    return sorted(selected, key=lambda row: (row["part"], row["part_order"]))


def _validate(rows):
    if len(rows) != 100:
        raise AssertionError(f"Expected 100 rows, found {len(rows)}")
    if len({row["question_id"] for row in rows}) != 100:
        raise AssertionError("Question IDs are not unique")

    for part in ("1", "2"):
        part_rows = [row for row in rows if row["part"] == part]
        if len(part_rows) != 50:
            raise AssertionError(f"Part {part} has {len(part_rows)} rows")
        if sum(row["source_correctness"] == "0" for row in part_rows) != 25:
            raise AssertionError(f"Part {part} does not have 25 incorrect rows")
        if sum(row["source_correctness"] == "1" for row in part_rows) != 25:
            raise AssertionError(f"Part {part} does not have 25 correct rows")
        if sum(
            row["selection_group"] == "unanimous_incorrect" for row in part_rows
        ) != 5:
            raise AssertionError(
                f"Part {part} does not have five unanimous failures"
            )
        for model in MODELS:
            count = sum(
                row["selection_group"] == "model_specific_incorrect"
                and row["target_model"] == model
                for row in part_rows
            )
            if count != 4:
                raise AssertionError(
                    f"Part {part} has {count} strict failures for {model}"
                )

    for row in rows:
        labels = [int(row[f"{model}_correctness"]) for model in MODELS]
        if row["selection_group"] == "unanimous_incorrect":
            expected = [0] * len(MODELS)
        elif row["selection_group"] == "unanimous_correct":
            expected = [1] * len(MODELS)
        else:
            expected = [
                0 if model == row["target_model"] else 1 for model in MODELS
            ]
        if labels != expected:
            raise AssertionError(
                f"Bad labels for {row['question_id']}: {labels} != {expected}"
            )


def build(force=False):
    if (OUTPUT_CSV.exists() or OUTPUT_IMAGES.exists()) and not force:
        raise FileExistsError(
            "Output already exists. Pass --force to rebuild the fixed cohort."
        )

    rows = _sample_cohort()
    _validate(rows)

    if OUTPUT_IMAGES.exists():
        if not force:
            raise FileExistsError(OUTPUT_IMAGES)
        # This directory contains only assets generated by this script.
        shutil.rmtree(OUTPUT_IMAGES)
    OUTPUT_IMAGES.mkdir(parents=True)

    output_rows = []
    for global_index, row in enumerate(rows, start=1):
        source = Path(row["source_image"])
        if not source.is_file():
            raise FileNotFoundError(source)
        suffix = source.suffix.lower() or ".jpg"
        filename = f"{global_index:03d}_{row['question_id']}{suffix}"
        shutil.copy2(source, OUTPUT_IMAGES / filename)

        output_row = {field: row.get(field, "") for field in OUTPUT_FIELDS}
        output_row["image_path"] = f"vqav2_validation_images/{filename}"
        output_rows.append(output_row)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} trials to {OUTPUT_CSV}")
    print(f"Copied {len(output_rows)} images to {OUTPUT_IMAGES}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true", help="replace an existing generated cohort"
    )
    args = parser.parse_args()
    build(force=args.force)
