#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

import os
import uuid

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_sensor_stream import initialize_main, write_main_episode

from lerobot.datasets.sensor_stream import SensorDatasetWriterLock
from lerobot.datasets.sensor_transaction import (
    SensorTransaction,
    SensorTransactionError,
    TransactionState,
    parquet_file_record,
)


@pytest.fixture(autouse=True)
def legacy_writer_lock(tmp_path):
    with SensorDatasetWriterLock(tmp_path):
        yield


def prepare_transaction(root, file_count=1):
    uid = str(uuid.uuid4())
    records = []
    for index in range(file_count):
        staging = root / ".sensor-staging" / uid / f"file-{index}.parquet"
        staging.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([{"value": index}]), staging)
        final = root / "raw" / f"file-{index}.parquet"
        records.append(parquet_file_record(root, staging, final))
    return SensorTransaction.prepare(
        root,
        episode_uid=uid,
        episode_index=0,
        frame_count=2,
        dataset_from_index=0,
        files=records,
    )


def test_prepared_without_main_save_aborts_and_quarantines(tmp_path) -> None:
    initialize_main(tmp_path)
    transaction = prepare_transaction(tmp_path)
    assert transaction.replay() == TransactionState.ABORTED
    assert (tmp_path / ".sensor-quarantine" / transaction.episode_uid).exists()


def test_crash_after_main_save_is_deterministically_replayed(tmp_path) -> None:
    initialize_main(tmp_path)
    transaction = prepare_transaction(tmp_path)
    write_main_episode(tmp_path)

    assert transaction.replay() == TransactionState.COMMITTED
    assert (tmp_path / "raw" / "file-0.parquet").exists()
    transaction.replay()
    assert transaction.state == TransactionState.COMMITTED


def test_partial_sidecar_promotion_is_idempotent(tmp_path) -> None:
    initialize_main(tmp_path)
    transaction = prepare_transaction(tmp_path, file_count=2)
    write_main_episode(tmp_path)
    transaction.mark_main_saved()
    first = transaction.journal["files"][0]
    final = tmp_path / first["final_path"]
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp_path / first["staging_path"], final)

    assert transaction.replay() == TransactionState.COMMITTED
    assert all((tmp_path / record["final_path"]).exists() for record in transaction.journal["files"])


def test_sidecar_promoted_and_committed_replay_are_idempotent(tmp_path) -> None:
    initialize_main(tmp_path)
    transaction = prepare_transaction(tmp_path)
    write_main_episode(tmp_path)
    transaction.mark_main_saved()
    transaction.promote_sidecars()
    assert transaction.state == TransactionState.SIDECAR_PROMOTED

    assert transaction.replay() == TransactionState.COMMITTED
    assert transaction.replay() == TransactionState.COMMITTED

    leftover = tmp_path / ".sensor-staging" / transaction.episode_uid
    leftover.mkdir(parents=True)
    (leftover / "orphan.tmp").write_text("closed after journal commit", encoding="utf-8")
    assert transaction.replay() == TransactionState.COMMITTED
    assert not leftover.exists()


def test_episode_totals_without_exact_main_artifacts_are_quarantined(tmp_path) -> None:
    initialize_main(tmp_path)
    transaction = prepare_transaction(tmp_path)
    info = tmp_path / "meta" / "info.json"
    info.write_text(
        '{"total_episodes": 1, "total_frames": 2, '
        '"data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"}',
        encoding="utf-8",
    )

    assert transaction.replay() == TransactionState.QUARANTINED


def test_digest_conflict_is_quarantined_and_refused(tmp_path) -> None:
    initialize_main(tmp_path)
    transaction = prepare_transaction(tmp_path)
    write_main_episode(tmp_path)
    transaction.mark_main_saved()
    (tmp_path / transaction.journal["files"][0]["staging_path"]).write_bytes(b"corrupt")

    with pytest.raises(SensorTransactionError, match="quarantine"):
        transaction.replay()
    assert transaction.state == TransactionState.QUARANTINED
