import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoProcessor
from transformers.models.llama.modeling_llama import create_causal_mask as llama_create_causal_mask


HELPER_SCRIPT = Path("/home/user/czx/MLLM_Token_Compression_Workdir/src/LLaVA_infer_lmm_evals/textvqa_layer18_mmr_limit_search.py")
MODEL_FILE = Path("/home/user/czx/MLLM_Token_Compression_Workdir/src/LLaVA_infer_lmm_evals/transformers/modeling_llavav28_visual_token_wo_skip_layer_mmr_attn_norm_simi_threhold_norm_angle_ffn_after_dynamic_scoring_layer_oneforward.py")


def load_helper_module(script_path: Path):
    module_name = f"limit_search_helper_{script_path.stem}"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load helper script: {script_path}")
        spec.loader.exec_module(module)
    return sys.modules[module_name]


HELPER = load_helper_module(HELPER_SCRIPT)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)

    sorted_values = values[order]
    start = 0
    while start < len(sorted_values):
        end = start + 1
        while end < len(sorted_values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_corr(rankdata(np.asarray(x, dtype=np.float64)), rankdata(np.asarray(y, dtype=np.float64)))


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.sqrt(np.mean((x - y) ** 2)))


def zscore_fit_transform(matrix: np.ndarray):
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (matrix - mean) / std, mean, std


def zscore_transform(matrix: np.ndarray, mean: np.ndarray, std: np.ndarray):
    std = np.where(std < 1e-12, 1.0, std)
    return (matrix - mean) / std


def fit_linear_regression(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.concatenate([np.ones((features.shape[0], 1)), features], axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    return coeffs


def predict_linear_regression(features: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    design = np.concatenate([np.ones((features.shape[0], 1)), features], axis=1)
    return design @ coeffs


def extract_layer18_scores(model, inputs: Dict[str, torch.Tensor], scoring_layer_idx: int, attn_anchor: str, pool_type: str):
    prepared = HELPER.prepare_multimodal_inputs(model, inputs)
    input_ids = prepared["input_ids"]
    attention_mask = prepared["attention_mask"]
    inputs_embeds = prepared["inputs_embeds"]
    image_start_idx = prepared["image_start_idx"]
    image_end_idx = prepared["image_end_idx"]

    seq_length = inputs_embeds.shape[1]
    device = inputs_embeds.device
    position_ids = torch.arange(seq_length, device=device, dtype=torch.long).unsqueeze(0)
    cache_position = torch.arange(seq_length, device=device, dtype=torch.long)

    valid_mask = torch.zeros(seq_length, device=device, dtype=torch.bool)
    if image_end_idx < seq_length:
        valid_mask[image_end_idx:] = True
    valid_mask = valid_mask & attention_mask[0].bool()
    query_indices = valid_mask.nonzero(as_tuple=False).squeeze(-1)

    layers = model.model.language_model.layers
    rotary_emb = model.model.language_model.rotary_emb
    hidden_states = inputs_embeds
    causal_mask = llama_create_causal_mask(
        config=model.model.language_model.config,
        input_embeds=hidden_states,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=position_ids,
    )
    pos_emb = rotary_emb(hidden_states, position_ids)

    with torch.no_grad():
        for layer_idx in range(scoring_layer_idx):
            hidden_states = layers[layer_idx](
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=pos_emb,
                output_attentions=False,
            )

        raw_attn_scores, importance_scores, attention_quality = model.model._get_visual_token_attention_scores(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            scoring_layer_idx=scoring_layer_idx,
            image_start_idx=image_start_idx,
            image_end_idx=image_end_idx,
            importance_mode="attn",
            ffn_propagation="jvp",
            attn_anchor=attn_anchor,
            pool_type=pool_type,
            query_indices=query_indices,
        )

    return (
        raw_attn_scores.detach().float().cpu(),
        importance_scores.detach().float().cpu(),
        float(attention_quality),
    )


def renyi_entropy(probs: torch.Tensor, alpha: float) -> float:
    probs = probs.float().clamp_min(1e-12)
    if abs(alpha - 1.0) < 1e-12:
        return float(-(probs * probs.log()).sum().item())
    power_sum = torch.pow(probs, alpha).sum().item()
    return float(math.log(power_sum) / (1.0 - alpha))


def spectral_flatness(probs: torch.Tensor) -> float:
    probs = probs.float().clamp_min(1e-12)
    geometric_mean = torch.exp(torch.log(probs).mean()).item()
    arithmetic_mean = probs.mean().item()
    if arithmetic_mean <= 1e-12:
        return 0.0
    return float(geometric_mean / arithmetic_mean)


def head_tail_ratio(sorted_probs: torch.Tensor, head_k: int, tail_k: int) -> float:
    head = sorted_probs[: min(head_k, sorted_probs.numel())]
    tail = sorted_probs[-min(tail_k, sorted_probs.numel()):]
    return float(head.mean().item() / (tail.mean().item() + 1e-12))


def topk_mass(sorted_probs: torch.Tensor, k: int) -> float:
    if k <= 0:
        return 0.0
    return float(sorted_probs[: min(k, sorted_probs.numel())].sum().item())


def tokens_to_mass(sorted_probs: torch.Tensor, mass: float) -> int:
    cumulative = torch.cumsum(sorted_probs, dim=0)
    target = torch.tensor(mass, dtype=sorted_probs.dtype)
    return int(torch.searchsorted(cumulative, target).item()) + 1


def log_rank_decay(sorted_probs: torch.Tensor, head_k: int) -> float:
    head_k = min(head_k, sorted_probs.numel())
    if head_k < 2:
        return 0.0
    y = torch.log(sorted_probs[:head_k].clamp_min(1e-12)).numpy()
    x = np.log(np.arange(1, head_k + 1, dtype=np.float64))
    x_centered = x - x.mean()
    denom = np.sum(x_centered * x_centered)
    if denom <= 1e-12:
        return 0.0
    slope = float(np.sum(x_centered * (y - y.mean())) / denom)
    return slope


def distribution_family_features(scores: torch.Tensor, prefix: str) -> dict:
    scores = scores.float().clamp_min(0)
    probs = scores / scores.sum().clamp_min(1e-12)
    sorted_probs = torch.sort(probs, descending=True).values
    num_tokens = sorted_probs.numel()
    features = {}

    for k in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        if k > num_tokens:
            continue
        features[f"{prefix}_top{k}_mass"] = topk_mass(sorted_probs, k)
        features[f"{prefix}_top{k}_mean"] = float(sorted_probs[:k].mean().item())

    for mass in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        keep_count = tokens_to_mass(sorted_probs, mass)
        features[f"{prefix}_cover_count_{int(round(mass * 100)):02d}"] = float(keep_count)
        features[f"{prefix}_cover_ratio_{int(round(mass * 100)):02d}"] = float(keep_count / num_tokens)

    features[f"{prefix}_max_share"] = float(sorted_probs[0].item())
    features[f"{prefix}_second_share"] = float(sorted_probs[1].item()) if num_tokens > 1 else 0.0
    features[f"{prefix}_top1_to_top2_ratio"] = float(sorted_probs[0].item() / (sorted_probs[1].item() + 1e-12)) if num_tokens > 1 else 0.0
    features[f"{prefix}_top1_to_top8_mass_ratio"] = float(sorted_probs[0].item() / (topk_mass(sorted_probs, 8) + 1e-12))
    features[f"{prefix}_head8_tail8_ratio"] = head_tail_ratio(sorted_probs, 8, 8)
    features[f"{prefix}_head16_tail16_ratio"] = head_tail_ratio(sorted_probs, 16, 16)
    features[f"{prefix}_herfindahl"] = float(torch.sum(probs * probs).item())
    features[f"{prefix}_participation_ratio"] = float(1.0 / features[f"{prefix}_herfindahl"])
    features[f"{prefix}_participation_ratio_norm"] = float(features[f"{prefix}_participation_ratio"] / num_tokens)
    features[f"{prefix}_renyi_entropy_05"] = renyi_entropy(probs, 0.5)
    features[f"{prefix}_renyi_entropy_20"] = renyi_entropy(probs, 2.0)
    features[f"{prefix}_spectral_flatness"] = spectral_flatness(probs)
    features[f"{prefix}_logrank_slope_8"] = log_rank_decay(sorted_probs, 8)
    features[f"{prefix}_logrank_slope_16"] = log_rank_decay(sorted_probs, 16)
    features[f"{prefix}_logrank_slope_32"] = log_rank_decay(sorted_probs, 32)
    return features


def search_best_single_feature(records: List[dict], feature_names: List[str], target: np.ndarray) -> dict:
    feature_rows = []
    for feature_name in feature_names:
        values = np.array([row.get(feature_name, 0.0) for row in records], dtype=np.float64)
        row = {
            "feature": feature_name,
            "transform": "raw",
            "pearson": pearson_corr(values, target),
            "spearman": spearman_corr(values, target),
        }
        feature_rows.append(row)
        if np.all(values >= 0.0):
            log_values = np.log1p(values)
            feature_rows.append(
                {
                    "feature": feature_name,
                    "transform": "log1p",
                    "pearson": pearson_corr(log_values, target),
                    "spearman": spearman_corr(log_values, target),
                }
            )
    feature_rows.sort(key=lambda item: abs(item["spearman"]), reverse=True)
    return {
        "best": feature_rows[0] if feature_rows else None,
        "all": feature_rows,
    }


def search_best_linear_combo(records: List[dict], feature_names: List[str], target: np.ndarray, max_features: int = 6) -> dict:
    base_views = []
    for feature_name in feature_names:
        values = np.array([row.get(feature_name, 0.0) for row in records], dtype=np.float64)
        base_views.append({"feature": feature_name, "transform": "raw", "values": values})
        if np.all(values >= 0.0):
            base_views.append({"feature": feature_name, "transform": "log1p", "values": np.log1p(values)})

    ranked = sorted(
        base_views,
        key=lambda row: abs(spearman_corr(row["values"], target)),
        reverse=True,
    )
    selected = []
    best_score = -1.0
    best_result = None

    rng = np.random.default_rng(42)
    indices = np.arange(len(records))
    rng.shuffle(indices)
    folds = np.array_split(indices, 5)

    for _ in range(max_features):
        best_candidate = None
        best_candidate_result = None
        for candidate in ranked[:40]:
            signature = (candidate["feature"], candidate["transform"])
            if signature in {(item["feature"], item["transform"]) for item in selected}:
                continue
            columns = selected + [candidate]
            matrix = np.stack([col["values"] for col in columns], axis=1)
            cv_pred = np.full(len(records), np.nan, dtype=np.float64)
            for fold in folds:
                fold_set = set(fold.tolist())
                train_idx = np.array([idx for idx in indices if idx not in fold_set], dtype=np.int64)
                test_idx = np.array(fold, dtype=np.int64)
                train_x, mean, std = zscore_fit_transform(matrix[train_idx])
                test_x = zscore_transform(matrix[test_idx], mean, std)
                coeffs = fit_linear_regression(train_x, target[train_idx])
                cv_pred[test_idx] = predict_linear_regression(test_x, coeffs)

            cv_pearson = pearson_corr(cv_pred, target)
            cv_spearman = spearman_corr(cv_pred, target)
            score = 0.5 * (abs(cv_pearson) + abs(cv_spearman))
            if score > best_score + 1e-9:
                best_candidate = candidate
                best_result = {
                    "features": [{"feature": col["feature"], "transform": col["transform"]} for col in columns],
                    "cv_pearson": cv_pearson,
                    "cv_spearman": cv_spearman,
                    "cv_rmse": rmse(cv_pred, target),
                    "score": score,
                }
                best_candidate_result = best_result
        if best_candidate is None:
            break
        selected.append(best_candidate)
        best_score = best_candidate_result["score"]
        best_result = best_candidate_result

    if best_result is None:
        return {"best_cv": None, "train_fit": None}

    matrix = np.stack(
        [
            next(
                row["values"]
                for row in base_views
                if row["feature"] == feature_spec["feature"] and row["transform"] == feature_spec["transform"]
            )
            for feature_spec in best_result["features"]
        ],
        axis=1,
    )
    full_x, mean, std = zscore_fit_transform(matrix)
    coeffs = fit_linear_regression(full_x, target)
    train_pred = predict_linear_regression(full_x, coeffs)
    train_fit = {
        "train_pearson": pearson_corr(train_pred, target),
        "train_spearman": spearman_corr(train_pred, target),
        "train_rmse": rmse(train_pred, target),
        "intercept": float(coeffs[0]),
        "coefficients": [
            {
                "feature": feature_spec["feature"],
                "transform": feature_spec["transform"],
                "weight": float(weight),
            }
            for feature_spec, weight in zip(best_result["features"], coeffs[1:])
        ],
    }
    return {"best_cv": best_result, "train_fit": train_fit}


def main():
    parser = argparse.ArgumentParser(description="Mine stronger layer-18 priors for zero-token TextVQA limit search.")
    parser.add_argument("--model-path", default="/home/user/czx/model/llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--sample-index-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--feature-jsonl", required=True)
    parser.add_argument("--scoring-layer", type=int, default=18)
    parser.add_argument("--attn-anchor", default="query", choices=["last", "query"])
    parser.add_argument("--pool-type", default="max", choices=["avg", "max"])
    args = parser.parse_args()

    records_path = Path(args.records_jsonl)
    sample_index_path = Path(args.sample_index_json)
    output_json = Path(args.output_json)
    feature_jsonl = Path(args.feature_jsonl)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    feature_jsonl.parent.mkdir(parents=True, exist_ok=True)

    all_records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    valid_records = [row for row in all_records if row.get("estimated_min_correct_keep_tokens") is not None]
    records_by_dataset = {int(row["dataset_index"]): row for row in valid_records}
    sampled_indices = json.loads(sample_index_path.read_text(encoding="utf-8"))
    target_indices = [idx for idx in sampled_indices if idx in records_by_dataset]

    processed_features = {}
    if feature_jsonl.exists():
        with feature_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                processed_features[int(row["dataset_index"])] = row

    device = torch.device("cuda")
    dataset = HELPER.load_textvqa_split("validation")
    LlavaForConditionalGeneration = HELPER.load_local_llava_class(MODEL_FILE)
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    for dataset_index in target_indices:
        if dataset_index in processed_features:
            continue
        doc = dataset[int(dataset_index)]
        prompt = HELPER.build_prompt(doc["question"])
        image = doc["image"].convert("RGB")
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}

        raw_scores, importance_scores, attention_quality = extract_layer18_scores(
            model=model,
            inputs=inputs,
            scoring_layer_idx=args.scoring_layer,
            attn_anchor=args.attn_anchor,
            pool_type=args.pool_type,
        )
        feature_row = {
            "dataset_index": int(dataset_index),
            "attention_quality": attention_quality,
        }
        feature_row.update(distribution_family_features(raw_scores, "raw"))
        feature_row.update(distribution_family_features(importance_scores, "importance"))
        feature_row["raw_importance_top1_gap"] = feature_row["raw_top1_mass"] - feature_row["importance_top1_mass"]
        feature_row["cover95_gap"] = feature_row["raw_cover_ratio_95"] - feature_row["importance_cover_ratio_95"]
        feature_row["cover99_gap"] = feature_row["raw_cover_ratio_99"] - feature_row["importance_cover_ratio_99"]
        feature_row["mass_top1_product"] = feature_row["raw_top1_mass"] * feature_row["importance_top1_mass"]
        feature_row["attention_top1_product"] = attention_quality * feature_row["raw_top1_mass"]
        feature_row["attention_cover95_product"] = attention_quality * feature_row["raw_cover_ratio_95"]
        feature_row["attention_cover99_product"] = attention_quality * feature_row["raw_cover_ratio_99"]

        with feature_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(feature_row, ensure_ascii=False) + "\n")
        processed_features[int(dataset_index)] = feature_row

    merged_records = []
    for dataset_index in target_indices:
        base_row = dict(records_by_dataset[dataset_index])
        base_row.update(processed_features[dataset_index])
        merged_records.append(base_row)

    target_log_keep = np.log1p(np.array([row["estimated_min_correct_keep_tokens"] for row in merged_records], dtype=np.float64))
    extra_feature_names = sorted(
        set(merged_records[0].keys())
        - {
            "dataset_index",
            "question_id",
            "question",
            "baseline_prediction",
            "baseline_exact_match",
            "search_status",
            "estimated_min_correct_keep_tokens",
            "estimated_limit_keep_ratio",
            "lower_wrong_keep_tokens",
            "upper_correct_keep_tokens",
            "num_search_evals",
            "search_trace",
        }
    )

    single_feature_result = search_best_single_feature(merged_records, extra_feature_names, target_log_keep)
    linear_combo_result = search_best_linear_combo(merged_records, extra_feature_names, target_log_keep)

    summary = {
        "num_valid_records": len(merged_records),
        "target": "log1p(min_correct_keep_tokens)",
        "best_single_feature": single_feature_result["best"],
        "top_single_features": single_feature_result["all"][:40],
        "best_linear_combo_cv": linear_combo_result["best_cv"],
        "best_linear_combo_train": linear_combo_result["train_fit"],
    }
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
