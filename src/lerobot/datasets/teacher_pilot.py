"""Read indexed B601 recordings in place, with whole-run splits and per-step mode labels."""

import bisect
import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset

from .compute_stats import aggregate_stats
from .factory import _apply_imagenet_stats, make_dataset, selected_episode_stats


def _indexed_path(parent: Path, relative: str) -> Path:
    locator = Path(relative)
    if locator.is_absolute() or ".." in locator.parts:
        raise ValueError(f"Expected a relative Teacher index path: {relative}")
    path = (parent / locator).resolve(strict=True)
    if not path.is_relative_to(parent):
        raise ValueError(f"Recording outside data root: {path}")
    return path


class TeacherPilotDataset(TorchDataset):
    def __init__(self, cfg, entries, parent, task_names):
        self.parts = []
        self.ends = []
        self.hold = []
        self.run_uids = []
        episode_rows = []
        self.task_indices = []
        self.action_offsets = torch.tensor(cfg.trainable_config.action_delta_indices or [0])
        for entry in entries:
            root = _indexed_path(parent, entry["root"])
            manifest = json.loads(_indexed_path(root, entry["manifest"]).read_text(encoding="utf-8"))
            if (
                not manifest["complete"]
                or manifest["episode_uid"] != entry["episode_uid"]
                or manifest["episode_index"] != entry["episode_index"]
                or manifest["frame_count"] != entry["frames"]
                or manifest["config"]["task"] != entry["task"]
                or any(
                    manifest["files"][kind]["sha256"] != entry[f"{kind}_sha256"]
                    for kind in ("frames", "control")
                )
            ):
                raise ValueError(f"Teacher manifest disagrees with index: {root}")
            with _indexed_path(root, manifest["files"]["control"]["path"]).open(encoding="utf-8") as control:
                if json.loads(control.readline())["run_uid"] != entry["run_uid"]:
                    raise ValueError(f"Teacher run UID disagrees with index: {root}")
            data_cfg = replace(
                cfg.dataset,
                root=str(root),
                repo_id=manifest["config"]["repo_id"],
                episodes=[entry["episode_index"]],
                teacher_pilot_index=None,
                held_out_run_uids=[],
                use_imagenet_stats=False,
            )
            ds = make_dataset(replace(cfg, dataset=data_cfg))
            if len(ds) != entry["frames"] or not len(ds):
                raise ValueError(f"Frame count mismatch: {root}")
            if self.parts and (
                ds.meta.features != self.parts[0].meta.features or ds.meta.fps != self.parts[0].meta.fps
            ):
                raise ValueError(f"Inconsistent Teacher schema: {root}")
            order = ds.meta.features["observation.tactile"]["names"]
            if order != ["sensor.gripper_force.left.normal_force", "sensor.gripper_force.right.normal_force"]:
                raise ValueError(f"Unexpected tactile channels: {order}")
            # A successful contact reset already enters FORCE_TRACK. Its first feedback action is next.
            labels = torch.empty(len(ds), dtype=torch.bool)
            stop = 0
            for span in entry["gripper_stage_ranges"]:
                if span["start"] != stop or not stop < span["stop"] <= len(ds):
                    raise ValueError(f"Invalid mode ranges: {root}")
                stop = span["stop"]
                source = span["gripper_source"]
                if source not in {"teleop", "transition", "teacher"}:
                    raise ValueError(f"Unknown Teacher gripper source: {source}")
                labels[span["start"] : stop] = source in {"transition", "teacher"}
            if stop != len(ds):
                raise ValueError(f"Incomplete mode ranges: {root}")
            start = self.ends[-1] if self.ends else 0
            self.parts.append(ds)
            self.hold.append(labels)
            self.run_uids.append(entry["run_uid"])
            self.ends.append(start + len(ds))
            self.task_indices.append(task_names.index(entry["task"]))
            episode_rows.append(
                {
                    "episode_index": len(self.parts) - 1,
                    "episode_uid": entry["episode_uid"],
                    "run_uid": entry["run_uid"],
                    "length": len(ds),
                    "dataset_from_index": start,
                    "dataset_to_index": self.ends[-1],
                    "tasks": [entry["task"]],
                }
            )
        self.num_episodes = len(self.parts)
        self.num_frames = self.ends[-1]
        self.episodes = list(range(self.num_episodes))
        self.absolute_to_relative_idx = None
        self.meta = copy.copy(self.parts[0].meta)
        self.meta.info = copy.copy(self.meta.info)
        self.meta.info.total_frames = self.num_frames
        self.meta.info.total_episodes = self.num_episodes
        self.meta.info.total_tasks = len(task_names)
        self.meta.repo_id = cfg.dataset.repo_id
        self.meta.tasks = pd.DataFrame(
            {"task_index": range(len(task_names))}, index=pd.Index(task_names, name="task")
        )
        self.meta.episodes = Dataset.from_list(episode_rows)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        part = bisect.bisect_right(self.ends, index)
        local_index = index - (self.ends[part - 1] if part else 0)
        sample = self.parts[part][local_index]
        # The child reader pads within its own episode. Use the same coordinates for the mode target.
        labels = self.hold[part]
        steps = (local_index + self.action_offsets).clamp(0, len(labels) - 1)
        sample["gripper_hold"] = labels[steps]
        sample["index"] = torch.tensor(index)
        sample["episode_index"] = torch.tensor(part)
        sample["task_index"] = torch.tensor(self.task_indices[part])
        # The verbatim instruction already contains the requested force.
        sample.pop("observation.force_target", None)
        return sample

    def frame_task_indices(self):
        """Global task indices for the trainer's per-task validation subsampler."""
        return np.repeat(self.task_indices, [len(part) for part in self.parts])


def make_teacher_pilot_split(cfg):
    data = cfg.dataset
    parent = (
        Path(data.root).expanduser() if data.root else Path.home() / "work/force-vla/data/b601_demos"
    ).resolve(strict=True)
    entries = json.loads(Path(data.teacher_pilot_index).expanduser().read_text(encoding="utf-8-sig"))[
        "episodes"
    ]
    runs = {e["run_uid"] for e in entries}
    held = set(data.held_out_run_uids)
    if not held <= runs or runs == held or len({e["episode_uid"] for e in entries}) != len(entries):
        raise ValueError("Invalid Teacher episode identities or run split")
    tasks = list(dict.fromkeys(e["task"] for e in entries))
    train = TeacherPilotDataset(cfg, [e for e in entries if e["run_uid"] not in held], parent, tasks)
    train.meta.stats = aggregate_stats(
        [selected_episode_stats(part.root, part.episodes) for part in train.parts]
    )
    if data.use_imagenet_stats:
        _apply_imagenet_stats(train.meta)
    evaluation = None
    if held:
        eval_cfg = replace(
            cfg, dataset=replace(data, image_transforms=replace(data.image_transforms, enable=False))
        )
        evaluation = TeacherPilotDataset(
            eval_cfg, [e for e in entries if e["run_uid"] in held], parent, tasks
        )
        if evaluation.meta.features != train.meta.features or evaluation.meta.fps != train.meta.fps:
            raise ValueError("Inconsistent Teacher schema between training and held-out runs")
        evaluation.meta.stats = train.meta.stats
    return train, evaluation
