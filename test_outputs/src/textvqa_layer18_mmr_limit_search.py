import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor
from transformers.models.llama.modeling_llama import create_causal_mask as llama_create_causal_mask


REPO_ROOT = Path("/home/user/czx/open_source_proj/lmms-eval")
WORKDIR_ROOT = Path("/home/user/czx/MLLM_Token_Compression_Workdir")
MODEL_FILE = WORKDIR_ROOT / "src/LLaVA_infer_lmm_evals/transformers/modeling_llavav28_visual_token_wo_skip_layer_mmr_attn_norm_simi_threhold_norm_angle_ffn_after_dynamic_scoring_layer_oneforward.py"
LOCAL_TEXTVQA_CACHE_ROOT = Path("/home/user/czx/.cache/huggingface/datasets/lmms-lab___textvqa/default/0.0.0/9c0699cd19768ac5ab97568f6b3cbac4c0062884")
PROMPT_SUFFIX = "\nAnswer the question using a single word or phrase."


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor


def load_local_llava_class(model_file: Path):
    module_name = f"transformers.models.llava.{model_file.stem}"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, model_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load model file: {model_file}")
        spec.loader.exec_module(module)
    else:
        module = sys.modules[module_name]

    if not hasattr(module, "LlavaForConditionalGeneration"):
        raise AttributeError(f"{model_file} does not define LlavaForConditionalGeneration")
    return module.LlavaForConditionalGeneration


def build_prompt(question: str) -> str:
    return f"USER: <image>\n{question.capitalize()}{PROMPT_SUFFIX} ASSISTANT:"


def compute_textvqa_accuracy(doc: dict, prediction: str, answer_processor: EvalAIAnswerProcessor) -> float:
    pred = answer_processor(prediction)
    answers = [answer_processor(ans) for ans in doc["answers"]]
    scores = []
    for idx in range(len(answers)):
        others = [answers[j] for j in range(len(answers)) if j != idx]
        matches = sum(item == pred for item in others)
        scores.append(min(1.0, matches / 3.0))
    return float(sum(scores) / len(scores))


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


def normalized_entropy(probs: torch.Tensor) -> float:
    probs = probs.float().clamp_min(1e-12)
    probs = probs / probs.sum()
    entropy = -(probs * probs.log()).sum().item()
    return float(entropy / math.log(probs.numel()))


def gini_coefficient(values: torch.Tensor) -> float:
    x = values.float().flatten().clamp_min(0)
    if x.numel() == 0 or float(x.sum()) == 0:
        return 0.0
    x = torch.sort(x)[0]
    n = x.numel()
    index = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
    numerator = (2 * index - n - 1) * x
    return float(numerator.sum().item() / (n * x.sum().item()))


def effective_ratio(values: torch.Tensor) -> float:
    probs = values.float().clamp_min(1e-12)
    probs = probs / probs.sum()
    entropy = -(probs * probs.log()).sum().item()
    return float(math.exp(entropy) / probs.numel())


def topk_mass_ratio(values: torch.Tensor, topk_ratio: float) -> float:
    values = values.float().clamp_min(0)
    if values.numel() == 0:
        return 0.0
    total = float(values.sum().item())
    if total <= 1e-12:
        return 0.0
    keep_k = max(1, int(round(values.numel() * topk_ratio)))
    return float(torch.topk(values, k=keep_k).values.sum().item() / total)


def cumulative_mass_ratio(values: torch.Tensor, mass: float) -> float:
    values = values.float().clamp_min(1e-12)
    values = values / values.sum()
    sorted_values = torch.sort(values, descending=True)[0]
    cumulative = torch.cumsum(sorted_values, dim=0)
    keep = int(torch.searchsorted(cumulative, torch.tensor(mass, device=values.device)).item()) + 1
    return float(keep / values.numel())


def redundancy_metrics(features: torch.Tensor) -> Dict[str, float]:
    features = F.normalize(features.float(), dim=-1)
    similarity = features @ features.T
    similarity.fill_diagonal_(-1.0)
    valid_similarity = similarity[similarity > -0.5]
    mean_max = float(similarity.max(dim=-1).values.mean().item())
    mean_pair = float(valid_similarity.mean().item()) if valid_similarity.numel() > 0 else 0.0
    return {
        "meanmax": mean_max,
        "meanpair": mean_pair,
    }


def spatial_dispersion(values: torch.Tensor) -> float:
    values = values.float().clamp_min(1e-12)
    values = values / values.sum()
    num_tokens = values.numel()
    side = int(round(math.sqrt(num_tokens)))
    if side * side != num_tokens:
        return 0.0
    coords_y, coords_x = torch.meshgrid(
        torch.linspace(0.0, 1.0, steps=side, device=values.device),
        torch.linspace(0.0, 1.0, steps=side, device=values.device),
        indexing="ij",
    )
    coords = torch.stack([coords_x.reshape(-1), coords_y.reshape(-1)], dim=-1)
    centroid = (coords * values.unsqueeze(-1)).sum(dim=0)
    return float((((coords - centroid) ** 2).sum(dim=-1) * values).sum().item())


def sample_indices(dataset_len: int, sample_size: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    indices = list(range(dataset_len))
    rng.shuffle(indices)
    return sorted(indices[:sample_size])


def load_textvqa_split(split: str):
    local_arrow_files = sorted(LOCAL_TEXTVQA_CACHE_ROOT.glob(f"textvqa-{split}-*.arrow"))
    if local_arrow_files:
        return load_dataset(
            "arrow",
            data_files={split: [str(path) for path in local_arrow_files]},
            split=split,
            cache_dir="/tmp/hf_arrow_cache",
        )
    return load_dataset("lmms-lab/textvqa", split=split)


def keep_tokens_to_ratio(keep_tokens: int, num_visual_tokens: int) -> float:
    if keep_tokens <= 0:
        return 0.0
    if keep_tokens >= num_visual_tokens:
        return 1.0
    return float(keep_tokens / num_visual_tokens)


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


def build_target_linear_estimator(
    valid_records: List[dict],
    feature_names: List[str],
    target: np.ndarray,
    feature_ranking: List[dict],
    ranking_key: str,
) -> Optional[dict]:
    if len(valid_records) < 8 or not feature_ranking:
        return None

    selected_entries = feature_ranking[: min(4, len(feature_ranking))]
    matrix_columns = []
    feature_specs = []
    for row in selected_entries:
        feature_name = row["feature"]
        transform = row.get("selected_feature_view", "raw")
        values = np.array([record.get(feature_name, 0.0) for record in valid_records], dtype=np.float64)
        if transform == "log1p":
            values = np.log1p(np.clip(values, a_min=0.0, a_max=None))
        matrix_columns.append(values)
        feature_specs.append({"feature": feature_name, "transform": transform})

    feature_matrix = np.stack(matrix_columns, axis=1)
    rng = np.random.default_rng(42)
    indices = np.arange(len(valid_records))
    rng.shuffle(indices)
    folds = np.array_split(indices, min(5, len(valid_records)))

    cv_predictions = np.full(len(valid_records), np.nan, dtype=np.float64)
    for fold in folds:
        if len(fold) == 0 or len(fold) == len(valid_records):
            continue
        fold_set = set(fold.tolist())
        train_idx = np.array([idx for idx in indices if idx not in fold_set], dtype=np.int64)
        test_idx = np.array(fold, dtype=np.int64)
        train_x, mean, std = zscore_fit_transform(feature_matrix[train_idx])
        test_x = zscore_transform(feature_matrix[test_idx], mean, std)
        coeffs = fit_linear_regression(train_x, target[train_idx])
        cv_predictions[test_idx] = predict_linear_regression(test_x, coeffs)

    valid_cv = np.isfinite(cv_predictions)
    full_x, mean, std = zscore_fit_transform(feature_matrix)
    full_coeffs = fit_linear_regression(full_x, target)
    coefficient_rows = []
    for spec, weight in zip(feature_specs, full_coeffs[1:]):
        coefficient_rows.append(
            {
                "feature": spec["feature"],
                "transform": spec["transform"],
                "weight": float(weight),
            }
        )
    coefficient_rows.sort(key=lambda row: abs(row["weight"]), reverse=True)
    return {
        "ranking_key": ranking_key,
        "features": feature_specs,
        "train_coefficients": coefficient_rows,
        "train_intercept": float(full_coeffs[0]),
        "cv_num_predictions": int(valid_cv.sum()),
        "cv_pearson": pearson_corr(cv_predictions[valid_cv], target[valid_cv]) if valid_cv.any() else 0.0,
        "cv_spearman": spearman_corr(cv_predictions[valid_cv], target[valid_cv]) if valid_cv.any() else 0.0,
        "cv_rmse": rmse(cv_predictions[valid_cv], target[valid_cv]) if valid_cv.any() else 0.0,
    }


def summarize_records(records: List[dict], feature_names: List[str]) -> dict:
    valid_records = [
        row for row in records
        if row.get("estimated_limit_keep_ratio") is not None
    ]
    summary = {
        "num_records": len(records),
        "num_valid_limit_records": len(valid_records),
        "status_counts": {},
    }

    for row in records:
        status = row.get("search_status", "unknown")
        summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1

    if not valid_records:
        summary["feature_correlations"] = []
        summary["best_single_feature"] = None
        summary["linear_estimator"] = None
        summary["feature_correlations_log1p_keep_tokens"] = []
        summary["best_single_feature_log1p_keep_tokens"] = None
        summary["linear_estimator_log1p_keep_tokens"] = None
        return summary

    target = np.array([row["estimated_limit_keep_ratio"] for row in valid_records], dtype=np.float64)
    target_keep_tokens = np.array([row["estimated_min_correct_keep_tokens"] for row in valid_records], dtype=np.float64)
    target_log_keep_tokens = np.log1p(target_keep_tokens)
    compression_target = 1.0 - target
    summary["mean_limit_keep_ratio"] = float(target.mean())
    summary["median_limit_keep_ratio"] = float(np.median(target))
    summary["mean_limit_keep_tokens"] = float(target_keep_tokens.mean())
    summary["median_limit_keep_tokens"] = float(np.median(target_keep_tokens))
    summary["mean_compression_capacity"] = float(compression_target.mean())
    summary["zero_keep_correct_fraction"] = float((target_keep_tokens == 0).mean())

    feature_correlations = []
    feature_correlations_log = []
    for feature_name in feature_names:
        values = np.array([row.get(feature_name, 0.0) for row in valid_records], dtype=np.float64)
        feature_correlations.append(
            {
                "feature": feature_name,
                "pearson_keep_ratio": pearson_corr(values, target),
                "spearman_keep_ratio": spearman_corr(values, target),
                "pearson_compression_capacity": pearson_corr(values, compression_target),
                "spearman_compression_capacity": spearman_corr(values, compression_target),
            }
        )
        log_entry = {
            "feature": feature_name,
            "raw_feature_pearson_log1p_keep_tokens": pearson_corr(values, target_log_keep_tokens),
            "raw_feature_spearman_log1p_keep_tokens": spearman_corr(values, target_log_keep_tokens),
            "log1p_feature_pearson_log1p_keep_tokens": None,
            "log1p_feature_spearman_log1p_keep_tokens": None,
            "selected_feature_view": "raw",
        }
        selected_abs_spearman = abs(log_entry["raw_feature_spearman_log1p_keep_tokens"])
        if np.all(values >= 0.0):
            log_values = np.log1p(values)
            log_entry["log1p_feature_pearson_log1p_keep_tokens"] = pearson_corr(log_values, target_log_keep_tokens)
            log_entry["log1p_feature_spearman_log1p_keep_tokens"] = spearman_corr(log_values, target_log_keep_tokens)
            if abs(log_entry["log1p_feature_spearman_log1p_keep_tokens"]) > selected_abs_spearman:
                log_entry["selected_feature_view"] = "log1p"
                selected_abs_spearman = abs(log_entry["log1p_feature_spearman_log1p_keep_tokens"])
        log_entry["selected_abs_spearman_log1p_keep_tokens"] = float(selected_abs_spearman)
        feature_correlations_log.append(log_entry)

    feature_correlations.sort(key=lambda row: abs(row["spearman_keep_ratio"]), reverse=True)
    feature_correlations_log.sort(key=lambda row: row["selected_abs_spearman_log1p_keep_tokens"], reverse=True)
    summary["feature_correlations"] = feature_correlations
    summary["best_single_feature"] = feature_correlations[0] if feature_correlations else None
    summary["feature_correlations_log1p_keep_tokens"] = feature_correlations_log
    summary["best_single_feature_log1p_keep_tokens"] = feature_correlations_log[0] if feature_correlations_log else None
    summary["linear_estimator"] = build_target_linear_estimator(
        valid_records=valid_records,
        feature_names=feature_names,
        target=target,
        feature_ranking=feature_correlations,
        ranking_key="keep_ratio",
    )
    summary["linear_estimator_log1p_keep_tokens"] = build_target_linear_estimator(
        valid_records=valid_records,
        feature_names=feature_names,
        target=target_log_keep_tokens,
        feature_ranking=feature_correlations_log,
        ranking_key="log1p_keep_tokens",
    )

    return summary


def prepare_multimodal_inputs(model, inputs: Dict[str, torch.Tensor]):
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs.get("pixel_values")
    image_sizes = inputs.get("image_sizes")

    inputs_embeds = model.get_input_embeddings()(input_ids)
    image_features = None
    if pixel_values is not None:
        image_features = model.model.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=None,
            vision_feature_select_strategy=None,
            image_sizes=image_sizes,
        )
        image_features = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        special_image_mask = model.model.get_placeholder_mask(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_features,
        )
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

    placeholder_mask = input_ids[0] == model.config.image_token_index
    placeholder_positions = torch.nonzero(placeholder_mask, as_tuple=False).squeeze(-1)
    image_start_idx = int(placeholder_positions.min().item())
    image_end_idx = int(placeholder_positions.max().item()) + 1
    num_visual_tokens = image_end_idx - image_start_idx

    if attention_mask is None:
        attention_mask = torch.ones((1, inputs_embeds.shape[1]), dtype=torch.long, device=inputs_embeds.device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "inputs_embeds": inputs_embeds,
        "image_features": image_features,
        "image_start_idx": image_start_idx,
        "image_end_idx": image_end_idx,
        "num_visual_tokens": num_visual_tokens,
    }


def extract_layer18_priors(model, inputs: Dict[str, torch.Tensor], scoring_layer_idx: int, attn_anchor: str, pool_type: str) -> dict:
    prepared = prepare_multimodal_inputs(model, inputs)
    input_ids = prepared["input_ids"]
    attention_mask = prepared["attention_mask"]
    inputs_embeds = prepared["inputs_embeds"]
    image_start_idx = prepared["image_start_idx"]
    image_end_idx = prepared["image_end_idx"]
    num_visual_tokens = prepared["num_visual_tokens"]

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

    raw_metrics = model.model._compute_distribution_metrics(raw_attn_scores)
    importance_metrics = model.model._compute_distribution_metrics(importance_scores)
    input_visual_features = inputs_embeds[0, image_start_idx:image_end_idx, :]
    layer18_visual_features = hidden_states[0, image_start_idx:image_end_idx, :]
    input_redundancy = redundancy_metrics(input_visual_features)
    layer18_redundancy = redundancy_metrics(layer18_visual_features)
    question_positions = torch.nonzero(~(input_ids[0] == model.config.image_token_index), as_tuple=False).squeeze(-1)
    question_positions = question_positions[question_positions > (image_end_idx - 1)]

    priors = {
        "num_visual_tokens": int(num_visual_tokens),
        "attention_quality": float(attention_quality),
        "importance_entropy_norm": normalized_entropy(importance_scores),
        "importance_effective_ratio": effective_ratio(importance_scores),
        "importance_topk_mass_00625": topk_mass_ratio(importance_scores, 0.0625),
        "importance_topk_mass_0125": topk_mass_ratio(importance_scores, 0.125),
        "importance_topk_mass_025": topk_mass_ratio(importance_scores, 0.25),
        "importance_cover_ratio_050": cumulative_mass_ratio(importance_scores, 0.5),
        "importance_cover_ratio_075": cumulative_mass_ratio(importance_scores, 0.75),
        "importance_gini_manual": gini_coefficient(importance_scores),
        "importance_spatial_dispersion": spatial_dispersion(importance_scores),
        "raw_attn_entropy_norm": normalized_entropy(raw_attn_scores),
        "raw_attn_effective_ratio": effective_ratio(raw_attn_scores),
        "raw_attn_topk_mass_0125": topk_mass_ratio(raw_attn_scores, 0.125),
        "raw_attn_topk_mass_025": topk_mass_ratio(raw_attn_scores, 0.25),
        "raw_attn_cover_ratio_050": cumulative_mass_ratio(raw_attn_scores, 0.5),
        "raw_attn_gini_manual": gini_coefficient(raw_attn_scores),
        "raw_attn_spatial_dispersion": spatial_dispersion(raw_attn_scores),
        "input_redundancy_meanmax": input_redundancy["meanmax"],
        "input_redundancy_meanpair": input_redundancy["meanpair"],
        "layer18_redundancy_meanmax": layer18_redundancy["meanmax"],
        "layer18_redundancy_meanpair": layer18_redundancy["meanpair"],
        "ocr_token_count": int(len(inputs.get("ocr_tokens", []))) if isinstance(inputs, dict) else 0,
        "question_token_count": int(question_positions.numel()),
    }

    for prefix, metrics in (("importance", importance_metrics), ("raw_attn", raw_metrics)):
        for metric_name, metric_value in metrics.items():
            priors[f"{prefix}_{metric_name}"] = float(metric_value)

    return priors


def decode_prediction(processor, generated_ids: torch.Tensor) -> str:
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
    text = decoded[0]
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:")[-1]
    return text.strip()


def search_limit_keep_tokens(
    evaluate_keep_tokens,
    num_visual_tokens: int,
    min_keep_tokens: int,
    max_refine_steps: int = 8,
) -> dict:
    evaluated = {}
    floor_status = "zero_keep_still_correct" if min_keep_tokens == 0 else "min_keep_still_correct"

    def eval_once(keep_tokens: int):
        keep_tokens = int(max(min_keep_tokens, min(num_visual_tokens, keep_tokens)))
        if keep_tokens not in evaluated:
            evaluated[keep_tokens] = evaluate_keep_tokens(keep_tokens)
        return evaluated[keep_tokens]

    full_result = eval_once(num_visual_tokens)
    if full_result["exact_match"] <= 0:
        return {
            "search_status": "full_keep_incorrect",
            "evaluated_points": [evaluated[k] for k in sorted(evaluated)],
            "estimated_min_correct_keep_tokens": None,
            "estimated_limit_keep_ratio": None,
            "lower_wrong_keep_tokens": None,
            "upper_correct_keep_tokens": None,
        }

    high = num_visual_tokens
    low = None
    current = high
    while current > min_keep_tokens:
        next_keep = max(min_keep_tokens, current // 2)
        if next_keep == current:
            break
        result = eval_once(next_keep)
        if result["exact_match"] > 0:
            high = next_keep
            current = next_keep
            if next_keep == min_keep_tokens:
                break
        else:
            low = next_keep
            break

    if high == min_keep_tokens and eval_once(min_keep_tokens)["exact_match"] > 0:
        return {
            "search_status": floor_status,
            "evaluated_points": [evaluated[k] for k in sorted(evaluated)],
            "estimated_min_correct_keep_tokens": int(min_keep_tokens),
            "estimated_limit_keep_ratio": float(min_keep_tokens / num_visual_tokens),
            "lower_wrong_keep_tokens": None,
            "upper_correct_keep_tokens": int(min_keep_tokens),
        }

    if low is None:
        low = min_keep_tokens
        if eval_once(low)["exact_match"] > 0:
            return {
                "search_status": floor_status,
                "evaluated_points": [evaluated[k] for k in sorted(evaluated)],
                "estimated_min_correct_keep_tokens": int(min_keep_tokens),
                "estimated_limit_keep_ratio": float(min_keep_tokens / num_visual_tokens),
                "lower_wrong_keep_tokens": None,
                "upper_correct_keep_tokens": int(min_keep_tokens),
            }

    while (high - low) > 1:
        mid = (high + low) // 2
        result = eval_once(mid)
        if result["exact_match"] > 0:
            high = mid
        else:
            low = mid

    refine_floor = max(min_keep_tokens, high - max_refine_steps)
    best = high
    for keep_tokens in range(high - 1, refine_floor - 1, -1):
        result = eval_once(keep_tokens)
        if result["exact_match"] > 0:
            best = keep_tokens
        else:
            low = max(low, keep_tokens)
            break

    return {
        "search_status": "boundary_found",
        "evaluated_points": [evaluated[k] for k in sorted(evaluated)],
        "estimated_min_correct_keep_tokens": int(best),
        "estimated_limit_keep_ratio": float(best / num_visual_tokens),
        "lower_wrong_keep_tokens": int(low),
        "upper_correct_keep_tokens": int(best),
    }


def main():
    parser = argparse.ArgumentParser(description="Dynamic search for the limit keep ratio at fixed layer-18 MMR pruning on TextVQA.")
    parser.add_argument("--model-path", default="/home/user/czx/model/llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--dataset-split", default="validation")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--scoring-layer", type=int, default=18)
    parser.add_argument("--min-keep-tokens", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--attn-anchor", default="query", choices=["last", "query"])
    parser.add_argument("--pool-type", default="max", choices=["avg", "max"])
    parser.add_argument("--add-type", default="add", choices=["add", "mul"])
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--sample-index-json", required=True)
    args = parser.parse_args()

    output_jsonl = Path(args.output_jsonl)
    summary_json = Path(args.summary_json)
    sample_index_json = Path(args.sample_index_json)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    sample_index_json.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_textvqa_split(args.dataset_split)
    sampled_indices = sample_indices(len(dataset), args.sample_size, args.sample_seed)
    sample_index_json.write_text(json.dumps(sampled_indices, ensure_ascii=False, indent=2), encoding="utf-8")

    processed = {}
    if output_jsonl.exists():
        with output_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                processed[int(row["dataset_index"])] = row

    device = torch.device("cuda")
    LlavaForConditionalGeneration = load_local_llava_class(MODEL_FILE)
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    answer_processor = EvalAIAnswerProcessor()

    feature_names = [
        "attention_quality",
        "importance_entropy_norm",
        "importance_effective_ratio",
        "importance_topk_mass_00625",
        "importance_topk_mass_0125",
        "importance_topk_mass_025",
        "importance_cover_ratio_050",
        "importance_cover_ratio_075",
        "importance_gini_manual",
        "importance_var",
        "importance_iqr",
        "importance_hoyer",
        "importance_spatial_entropy",
        "importance_spatial_dispersion",
        "raw_attn_entropy_norm",
        "raw_attn_effective_ratio",
        "raw_attn_topk_mass_0125",
        "raw_attn_topk_mass_025",
        "raw_attn_cover_ratio_050",
        "raw_attn_gini_manual",
        "raw_attn_var",
        "raw_attn_iqr",
        "raw_attn_spatial_entropy",
        "raw_attn_spatial_dispersion",
        "input_redundancy_meanmax",
        "input_redundancy_meanpair",
        "layer18_redundancy_meanmax",
        "layer18_redundancy_meanpair",
        "ocr_token_count",
        "question_token_count",
    ]

    records = list(processed.values())
    for dataset_index in tqdm(sampled_indices, desc="Layer-18 MMR limit search"):
        if dataset_index in processed:
            continue

        doc = dataset[int(dataset_index)]
        prompt = build_prompt(doc["question"])
        image = doc["image"].convert("RGB")
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}

        with torch.no_grad():
            baseline_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                scoring_layer_idx=None,
            )

        baseline_prediction = decode_prediction(processor, baseline_ids)
        baseline_exact_match = compute_textvqa_accuracy(doc, baseline_prediction, answer_processor)

        priors = extract_layer18_priors(
            model=model,
            inputs=inputs,
            scoring_layer_idx=args.scoring_layer,
            attn_anchor=args.attn_anchor,
            pool_type=args.pool_type,
        )
        priors["ocr_token_count"] = int(len(doc.get("ocr_tokens", [])))

        num_visual_tokens = int(priors["num_visual_tokens"])
        min_keep_tokens = min(args.min_keep_tokens, num_visual_tokens)

        def evaluate_keep_tokens(keep_tokens: int) -> dict:
            keep_ratio = keep_tokens_to_ratio(keep_tokens, num_visual_tokens)
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    scoring_layer_idx=args.scoring_layer,
                    force_fixed_scoring_layer=True,
                    visual_token_keep_ratio=keep_ratio,
                    visual_token_min_keep=min_keep_tokens,
                    attn_anchor=args.attn_anchor,
                    pool_type=args.pool_type,
                    add_type=args.add_type,
                )
            prediction = decode_prediction(processor, generated_ids)
            exact_match = compute_textvqa_accuracy(doc, prediction, answer_processor)
            return {
                "keep_tokens": int(keep_tokens),
                "keep_ratio": float(keep_tokens / num_visual_tokens),
                "prediction": prediction,
                "exact_match": float(exact_match),
            }

        if baseline_exact_match <= 0:
            search_result = {
                "search_status": "baseline_incorrect",
                "evaluated_points": [],
                "estimated_min_correct_keep_tokens": None,
                "estimated_limit_keep_ratio": None,
                "lower_wrong_keep_tokens": None,
                "upper_correct_keep_tokens": None,
            }
        else:
            search_result = search_limit_keep_tokens(
                evaluate_keep_tokens=evaluate_keep_tokens,
                num_visual_tokens=num_visual_tokens,
                min_keep_tokens=min_keep_tokens,
            )

        record = {
            "dataset_index": int(dataset_index),
            "question_id": int(doc["question_id"]),
            "question": doc["question"],
            "baseline_prediction": baseline_prediction,
            "baseline_exact_match": float(baseline_exact_match),
            "search_status": search_result["search_status"],
            "estimated_min_correct_keep_tokens": search_result["estimated_min_correct_keep_tokens"],
            "estimated_limit_keep_ratio": search_result["estimated_limit_keep_ratio"],
            "lower_wrong_keep_tokens": search_result["lower_wrong_keep_tokens"],
            "upper_correct_keep_tokens": search_result["upper_correct_keep_tokens"],
            "num_search_evals": len(search_result["evaluated_points"]),
            "search_trace": search_result["evaluated_points"],
        }
        record.update({name: float(priors[name]) if isinstance(priors[name], (int, float)) else priors[name] for name in priors})

        with output_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        processed[int(dataset_index)] = record
        records.append(record)
        torch.cuda.empty_cache()

    ordered_records = [processed[idx] for idx in sampled_indices if idx in processed]
    summary = summarize_records(ordered_records, feature_names=feature_names)
    summary.update(
        {
            "sample_size": len(sampled_indices),
            "sample_seed": args.sample_seed,
            "scoring_layer": args.scoring_layer,
            "min_keep_tokens": args.min_keep_tokens,
            "attn_anchor": args.attn_anchor,
            "pool_type": args.pool_type,
            "add_type": args.add_type,
            "model_file": str(MODEL_FILE),
        }
    )
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
