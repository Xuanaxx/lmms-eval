#!/usr/bin/env python3
import argparse
import fcntl
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional


EVAL_SCRIPT = Path("/home/user/czx/open_source_proj/lmms-eval/scripts/h100/llava_1_5_hf_v33_mmr_score18_last_gqa_textvqa.sh")
OUT_ROOT = Path("/home/user/czx/open_source_proj/lmms-eval/outputs/logs/textvqa_v33_hparam_search_gap1000")

STAGE1_LAYERS = [2, 3, 4, 5]
RECOVER_LAYERS = [6, 7, 8, 9, 10, 11, 12, 13]
FINAL_LAYERS = [12, 13, 14, 15, 16, 17, 18, 19, 20]
STAGE1_COUNTS = [72, 144, 288]
RECOVER_COUNTS = [144, 288]
BASELINE = {
    "stage1_merge_layer_idx": 3,
    "recover_layer_idx": 7,
    "final_prune_layer_idx": 18,
    "stage1_target_count": 288,
    "recover_topk_target_count": 288,
}


@dataclass(frozen=True)
class Config:
    stage1_merge_layer_idx: int
    recover_layer_idx: int
    final_prune_layer_idx: int
    stage1_target_count: int
    recover_topk_target_count: int

    @property
    def name(self) -> str:
        return (
            f"s1l{self.stage1_merge_layer_idx}_rl{self.recover_layer_idx}_"
            f"fl{self.final_prune_layer_idx}_s1c{self.stage1_target_count}_"
            f"rtc{self.recover_topk_target_count}"
        )

    @property
    def extra_model_args(self) -> str:
        items = asdict(self)
        items["visual_token_target_count"] = 64
        items["scoring_layer_idx"] = self.final_prune_layer_idx
        return ",".join(f"{key}={value}" for key, value in items.items())


def is_valid(config: Config) -> bool:
    return (
        config.recover_layer_idx - config.stage1_merge_layer_idx >= 4
        and config.final_prune_layer_idx - config.recover_layer_idx >= 6
    )


def unique(configs: Iterable[Config]) -> list[Config]:
    seen = set()
    result = []
    for config in configs:
        if not is_valid(config):
            continue
        key = asdict(config)
        frozen_key = tuple(key[item] for item in sorted(key))
        if frozen_key in seen:
            continue
        seen.add(frozen_key)
        result.append(config)
    return result


def baseline_config() -> Config:
    return Config(**BASELINE)


def phase1_configs() -> list[Config]:
    base = baseline_config()
    configs = [base]
    for value in STAGE1_LAYERS:
        configs.append(Config(value, base.recover_layer_idx, base.final_prune_layer_idx, base.stage1_target_count, base.recover_topk_target_count))
    for value in RECOVER_LAYERS:
        configs.append(Config(base.stage1_merge_layer_idx, value, base.final_prune_layer_idx, base.stage1_target_count, base.recover_topk_target_count))
    for value in FINAL_LAYERS:
        configs.append(Config(base.stage1_merge_layer_idx, base.recover_layer_idx, value, base.stage1_target_count, base.recover_topk_target_count))
    for value in STAGE1_COUNTS:
        configs.append(Config(base.stage1_merge_layer_idx, base.recover_layer_idx, base.final_prune_layer_idx, value, base.recover_topk_target_count))
    for value in RECOVER_COUNTS:
        configs.append(Config(base.stage1_merge_layer_idx, base.recover_layer_idx, base.final_prune_layer_idx, base.stage1_target_count, value))
    return unique(configs)


def pick_top_values(records: list[dict], key: str, values: list[int], keep: int) -> list[int]:
    scores = {value: [] for value in values}
    for record in records:
        if record.get("limit") != 1000 or record.get("status") != "ok":
            continue
        cfg = record["config"]
        value = cfg[key]
        if value in scores:
            scores[value].append(record["exact_match"])
    ranked = sorted(
        values,
        key=lambda value: (max(scores[value]) if scores[value] else -1.0, -abs(value - BASELINE[key])),
        reverse=True,
    )
    selected = ranked[:keep]
    if BASELINE[key] not in selected:
        selected.append(BASELINE[key])
    return sorted(set(selected))


def phase2_configs(records: list[dict]) -> list[Config]:
    s1_layers = pick_top_values(records, "stage1_merge_layer_idx", STAGE1_LAYERS, 2)
    recover_layers = pick_top_values(records, "recover_layer_idx", RECOVER_LAYERS, 2)
    final_layers = pick_top_values(records, "final_prune_layer_idx", FINAL_LAYERS, 3)
    s1_counts = pick_top_values(records, "stage1_target_count", STAGE1_COUNTS, 2)
    recover_counts = pick_top_values(records, "recover_topk_target_count", RECOVER_COUNTS, 2)

    configs = []
    for s1_layer in s1_layers:
        for recover_layer in recover_layers:
            for final_layer in final_layers:
                for s1_count in s1_counts:
                    for recover_count in recover_counts:
                        configs.append(Config(s1_layer, recover_layer, final_layer, s1_count, recover_count))
    return unique(configs)


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_record(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def result_key(config: Config, limit: Optional[int]) -> tuple:
    return (config.name, limit)


def completed_keys(records: list[dict]) -> set[tuple]:
    keys = set()
    for record in records:
        if record.get("status") == "ok":
            keys.add((record["name"], record.get("limit")))
    return keys


def find_result_json(output_path: Path) -> Optional[Path]:
    result_files = sorted(output_path.glob("*/**/*_results.json"), key=lambda path: path.stat().st_mtime)
    return result_files[-1] if result_files else None


def parse_exact_match(output_path: Path) -> tuple[float, Optional[Path]]:
    result_path = find_result_json(output_path)
    if result_path is None:
        raise FileNotFoundError(f"No *_results.json found under {output_path}")
    data = json.loads(result_path.read_text(encoding="utf-8"))
    return float(data["results"]["textvqa_val"]["exact_match,none"]), result_path


def run_one(config: Config, gpu: int, limit: Optional[int], phase: str) -> dict:
    limit_name = f"limit{limit}" if limit is not None else "full"
    output_path = OUT_ROOT / limit_name / config.name
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / "test.log"
    existing_result = find_result_json(output_path)
    if existing_result is not None:
        score, result_path = parse_exact_match(output_path)
        return {
            "phase": phase,
            "name": config.name,
            "config": asdict(config),
            "limit": limit,
            "gpu": gpu,
            "elapsed_seconds": 0.0,
            "output_path": str(output_path),
            "log_path": str(log_path),
            "status": "ok",
            "exact_match": score,
            "result_path": str(result_path),
            "reused_result": True,
        }
    env = {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "TASKS": "textvqa_val",
        "OUTPUT_PATH": str(output_path),
        "LOG_PATH": str(log_path),
        "EXTRA_MODEL_ARGS": config.extra_model_args,
        "TRANSFORMERS_VERBOSITY": "error",
    }
    if limit is not None:
        env["LIMIT"] = str(limit)
    merged_env = {**dict(**os.environ), **env}
    start = time.time()
    proc = subprocess.run(["bash", str(EVAL_SCRIPT)], env=merged_env)
    elapsed = time.time() - start
    record = {
        "phase": phase,
        "name": config.name,
        "config": asdict(config),
        "limit": limit,
        "gpu": gpu,
        "elapsed_seconds": elapsed,
        "output_path": str(output_path),
        "log_path": str(log_path),
        "status": "ok" if proc.returncode == 0 else "failed",
    }
    result_path = find_result_json(output_path)
    if proc.returncode == 0 or result_path is not None:
        score, result_path = parse_exact_match(output_path)
        record["status"] = "ok"
        record["exact_match"] = score
        record["result_path"] = str(result_path)
    else:
        record["returncode"] = proc.returncode
    return record


def worker_loop(worker_id: int, gpu: int, jobs_path: Path, records_path: Path, claims_path: Path, max_jobs: Optional[int] = None) -> None:
    lock_path = claims_path.with_suffix(".lock")
    completed_by_worker = 0
    while True:
        if max_jobs is not None and completed_by_worker >= max_jobs:
            return
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle, fcntl.LOCK_EX)
            jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
            records = load_records(records_path)
            done = completed_keys(records)
            recorded = {(record["name"], record.get("limit")) for record in records}
            claims = load_records(claims_path)
            claimed = {(claim["name"], claim.get("limit")) for claim in claims} - recorded

            next_job = None
            for job in jobs:
                key = (job["name"], job.get("limit"))
                if key not in done and key not in claimed:
                    next_job = job
                    break
            if next_job is None:
                fcntl.flock(lock_handle, fcntl.LOCK_UN)
                return

            append_record(claims_path, {"worker": worker_id, "gpu": gpu, **next_job, "claimed_at": time.time()})
            fcntl.flock(lock_handle, fcntl.LOCK_UN)
        config = Config(**next_job["config"])
        record = run_one(config, gpu=gpu, limit=next_job.get("limit"), phase=next_job["phase"])
        append_record(records_path, record)
        completed_by_worker += 1


def write_jobs(path: Path, configs: list[Config], limit: Optional[int], phase: str, records: list[dict], append: bool = True) -> None:
    existing = []
    if append and path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing_keys = {(job["name"], job.get("limit")) for job in existing}
    done = completed_keys(records)
    jobs = existing[:]
    for config in configs:
        key = result_key(config, limit)
        if key in existing_keys or key in done:
            continue
        jobs.append({"phase": phase, "name": config.name, "config": asdict(config), "limit": limit})
    path.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary(records: list[dict], path: Path) -> None:
    ok = [record for record in records if record.get("status") == "ok"]
    ok_sorted = sorted(ok, key=lambda record: (record.get("limit") is None, record["exact_match"]), reverse=True)
    summary = {
        "num_ok": len(ok),
        "num_failed": len([record for record in records if record.get("status") == "failed"]),
        "best_limited": next((record for record in ok_sorted if record.get("limit") == 1000), None),
        "best_full": next((record for record in ok_sorted if record.get("limit") is None), None),
        "top10": ok_sorted[:10],
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--add-phase2", action="store_true")
    parser.add_argument("--add-full-top", type=int, default=0)
    parser.add_argument("--max-jobs", type=int)
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs_path = OUT_ROOT / "jobs.json"
    records_path = OUT_ROOT / "records.jsonl"
    claims_path = OUT_ROOT / "claims.jsonl"
    summary_path = OUT_ROOT / "summary.json"

    if args.prepare:
        records = load_records(records_path)
        write_jobs(jobs_path, phase1_configs(), limit=1000, phase="phase1_coordinate", records=records, append=False)
        write_summary(load_records(records_path), summary_path)
        return

    if args.add_phase2:
        records = load_records(records_path)
        write_jobs(jobs_path, phase2_configs(records), limit=1000, phase="phase2_neighborhood", records=records)
        write_summary(load_records(records_path), summary_path)
        return

    if args.add_full_top:
        records = [record for record in load_records(records_path) if record.get("limit") == 1000 and record.get("status") == "ok"]
        top_records = sorted(records, key=lambda record: record["exact_match"], reverse=True)[: args.add_full_top]
        configs = [Config(**record["config"]) for record in top_records]
        write_jobs(jobs_path, configs, limit=None, phase="phase3_full", records=load_records(records_path))
        write_summary(load_records(records_path), summary_path)
        return

    if args.worker_id is None or args.gpu is None:
        raise SystemExit("Use --prepare/--add-phase2/--add-full-top, or provide --worker-id and --gpu.")

    worker_loop(args.worker_id, args.gpu, jobs_path, records_path, claims_path, max_jobs=args.max_jobs)
    write_summary(load_records(records_path), summary_path)


if __name__ == "__main__":
    main()
