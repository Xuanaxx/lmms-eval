import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers.models.llama.modeling_llama import create_causal_mask as llama_create_causal_mask

from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor


RATIO_GRID = (0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0)
PROMPT_SUFFIX = "\nAnswer the question using a single word or phrase."


def build_prompt(question: str) -> str:
    return f"USER: <image>\n{question.capitalize()}{PROMPT_SUFFIX} ASSISTANT:"


def normalized_entropy(probs: torch.Tensor) -> float:
    probs = probs.float().clamp_min(1e-12)
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


def cumulative_mass_ratio(probs: torch.Tensor, mass: float) -> float:
    sorted_probs = torch.sort(probs.float(), descending=True)[0]
    cumulative = torch.cumsum(sorted_probs, dim=0)
    keep = int(torch.searchsorted(cumulative, torch.tensor(mass, device=probs.device)).item()) + 1
    return float(keep / probs.numel())


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
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_corr(rankdata(x), rankdata(y))


def compute_textvqa_accuracy(doc: dict, prediction: str, answer_processor: EvalAIAnswerProcessor) -> float:
    pred = answer_processor(prediction)
    answers = [answer_processor(ans) for ans in doc["answers"]]
    scores = []
    for idx in range(len(answers)):
        others = [answers[j] for j in range(len(answers)) if j != idx]
        matches = sum(item == pred for item in others)
        scores.append(min(1.0, matches / 3.0))
    return float(sum(scores) / len(scores))


def sample_indices(dataset_len: int, sample_size: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    indices = list(range(dataset_len))
    rng.shuffle(indices)
    return sorted(indices[:sample_size])


def build_left_padded_batch(
    sequence_embeddings: list[torch.Tensor],
    pad_embedding: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(seq.shape[0] for seq in sequence_embeddings)
    hidden_size = sequence_embeddings[0].shape[-1]
    batch_size = len(sequence_embeddings)

    embeds = torch.zeros(
        (batch_size, max_len, hidden_size),
        dtype=sequence_embeddings[0].dtype,
        device=device,
    )
    masks = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

    for row, seq in enumerate(sequence_embeddings):
        seq_len = seq.shape[0]
        offset = max_len - seq_len
        if offset > 0:
            embeds[row, :offset] = pad_embedding.unsqueeze(0).expand(offset, -1)
        embeds[row, offset:] = seq
        masks[row, offset:] = 1

    return embeds, masks


def build_full_multimodal_embeddings(
    model: LlavaForConditionalGeneration,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    text_embeddings = model.get_input_embeddings()(input_ids)
    image_features = model.model.get_image_features(pixel_values=pixel_values)
    if isinstance(image_features, (list, tuple)):
        image_features = torch.cat(image_features, dim=0)
    elif image_features.dim() == 3:
        image_features = image_features[0]

    image_features = image_features.to(device=text_embeddings.device, dtype=text_embeddings.dtype)
    image_mask = input_ids[0] == model.config.image_token_index
    full_embeddings = text_embeddings.clone()
    full_embeddings[0, image_mask] = image_features
    return full_embeddings, image_features


def compute_visual_token_deletion_cost_scores(
    layer,
    layer_input: torch.Tensor,
    hidden_normed: torch.Tensor,
    attn_output: torch.Tensor,
    attn_weights: torch.Tensor,
    image_start_idx: int,
    image_end_idx: int,
    query_indices: torch.Tensor,
    norm_weight: float = 1.0,
    combine_type: str = "add",
) -> torch.Tensor:
    self_attn = layer.self_attn
    batch_size, seq_len, hidden_size = hidden_normed.shape
    if batch_size != 1:
        raise ValueError("This experiment assumes batch_size=1.")

    num_heads = getattr(self_attn, "num_heads", getattr(self_attn.config, "num_attention_heads"))
    num_kv_heads = getattr(self_attn, "num_key_value_heads", getattr(self_attn.config, "num_key_value_heads"))
    head_dim = self_attn.head_dim

    value_proj = self_attn.v_proj(hidden_normed)
    value_states = value_proj.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    if num_kv_heads != num_heads:
        repeats = num_heads // num_kv_heads
        value_states = (
            value_states.unsqueeze(2)
            .expand(batch_size, num_kv_heads, repeats, seq_len, head_dim)
            .reshape(batch_size, num_heads, seq_len, head_dim)
        )

    query_indices = query_indices.to(dtype=torch.long, device=hidden_normed.device)
    query_attn = attn_weights[:, :, query_indices, :]  # [1, H, Q, S]
    query_attn = query_attn[0].permute(1, 0, 2).contiguous()  # [Q, H, S]

    value_states = value_states[0]  # [H, S, D]
    z_heads = torch.einsum("qhs,hsd->qhd", query_attn, value_states)  # [Q, H, D]

    visual_values = value_states[:, image_start_idx:image_end_idx, :]  # [H, V, D]
    visual_attn = query_attn[:, :, image_start_idx:image_end_idx]  # [Q, H, V]

    beta = visual_attn / (1.0 - visual_attn).clamp_min(1e-6)
    delta_z = beta.unsqueeze(-1) * (z_heads.unsqueeze(2) - visual_values.unsqueeze(0))  # [Q, H, V, D]
    delta_z_cat = delta_z.permute(0, 2, 1, 3).contiguous().view(-1, num_heads * head_dim)
    delta_y = self_attn.o_proj(delta_z_cat).view(query_indices.numel(), -1, hidden_size)  # [Q, V, D]

    post_norm = layer.post_attention_layernorm
    mlp = layer.mlp

    residual_states = layer_input[0, query_indices, :]  # [Q, D]
    attn_states = attn_output[0, query_indices, :]  # [Q, D]
    u_states = residual_states + attn_states

    normed_u = post_norm(u_states.view(-1, 1, hidden_size)).view(-1, hidden_size)
    base_ffn = mlp(normed_u.view(-1, 1, hidden_size)).view(-1, hidden_size)
    base_output = u_states + base_ffn  # [Q, D]

    delta_u = delta_y
    norm_weight_tensor = getattr(post_norm, "weight", None)
    if norm_weight_tensor is None:
        raise AttributeError("Expected RMSNorm-like post_attention_layernorm with weight.")
    eps = getattr(post_norm, "eps", getattr(post_norm, "variance_epsilon", 1e-6))

    d = u_states.shape[-1]
    r2 = u_states.pow(2).mean(dim=-1) + eps  # [Q]
    r = torch.sqrt(r2)

    dot = (delta_u * u_states.unsqueeze(1)).sum(dim=-1, keepdim=True)  # [Q, V, 1]
    proj = u_states.unsqueeze(1) * (dot / (d * r2.view(-1, 1, 1)))
    delta_normed = (norm_weight_tensor / r.view(-1, 1)).unsqueeze(1) * (delta_u - proj)

    gate_base = mlp.gate_proj(normed_u)
    up_base = mlp.up_proj(normed_u)
    act_base = F.silu(gate_base)
    sig = torch.sigmoid(gate_base)
    silu_prime = sig + gate_base * sig * (1.0 - sig)

    delta_normed_flat = delta_normed.reshape(-1, hidden_size)
    delta_gate = mlp.gate_proj(delta_normed_flat).view(query_indices.numel(), -1, gate_base.shape[-1])
    delta_up = mlp.up_proj(delta_normed_flat).view(query_indices.numel(), -1, up_base.shape[-1])

    delta_act = delta_gate * silu_prime.unsqueeze(1)
    delta_mlp_hidden = delta_act * up_base.unsqueeze(1) + act_base.unsqueeze(1) * delta_up
    delta_ffn = mlp.down_proj(delta_mlp_hidden.reshape(-1, gate_base.shape[-1])).view(query_indices.numel(), -1, hidden_size)

    delta_x = delta_u + delta_ffn  # [Q, V, D]
    delta_x_norm = delta_x.norm(dim=-1)

    base_norm = base_output.norm(p=2, dim=-1).clamp_min(1e-6)  # [Q]
    x_dot_delta = (delta_x * base_output.unsqueeze(1)).sum(dim=-1)
    new_norm_sq = base_norm.unsqueeze(1) ** 2 + 2.0 * x_dot_delta + delta_x_norm**2
    new_norm = torch.sqrt(new_norm_sq.clamp_min(1e-12))
    cos_theta = (base_norm.unsqueeze(1) ** 2 + x_dot_delta) / (base_norm.unsqueeze(1) * new_norm + 1e-6)
    cos_theta = cos_theta.clamp(min=-1.0, max=1.0)
    angle_loss = 1.0 - cos_theta

    if combine_type == "add":
        importance = norm_weight * delta_x_norm + (1.0 - norm_weight) * angle_loss
    elif combine_type == "mul":
        importance = (delta_x_norm.clamp_min(1e-12) ** norm_weight) * (angle_loss.clamp_min(1e-12) ** (1.0 - norm_weight))
    else:
        raise ValueError(f"Unsupported combine_type: {combine_type}")

    return importance.mean(dim=0)


def compute_scoring_signals(
    model: LlavaForConditionalGeneration,
    full_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    image_start_idx: int,
    image_end_idx: int,
    query_indices: torch.Tensor,
    layer_idx: int,
    norm_weight: float = 1.0,
    combine_type: str = "add",
) -> dict:
    language_model = model.model.language_model
    hidden_states = full_embeddings
    device = hidden_states.device
    seq_len = hidden_states.shape[1]
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    cache_position = torch.arange(seq_len, device=device, dtype=torch.long)

    for idx in range(layer_idx):
        layer = language_model.layers[idx]
        position_embeddings = language_model.rotary_emb(hidden_states, position_ids)
        causal_mask = llama_create_causal_mask(
            config=language_model.config,
            input_embeds=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )
        hidden_states = layer(
            hidden_states=hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    scoring_layer = language_model.layers[layer_idx]
    position_embeddings = language_model.rotary_emb(hidden_states, position_ids)
    causal_mask = llama_create_causal_mask(
        config=language_model.config,
        input_embeds=hidden_states,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=position_ids,
    )

    hidden_normed = scoring_layer.input_layernorm(hidden_states)
    attn_output, attn_weights = scoring_layer.self_attn(
        hidden_states=hidden_normed,
        attention_mask=causal_mask,
        past_key_values=None,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
    )

    layer_attn = attn_weights[0].float()
    question_to_all = layer_attn[:, query_indices].mean(dim=(0, 1))
    image_positions = torch.arange(image_start_idx, image_end_idx, device=device)
    question_to_vision = question_to_all[image_positions].clamp_min(1e-12)
    question_to_vision = question_to_vision / question_to_vision.sum()
    vision_share = float((question_to_all[image_positions].sum() / question_to_all.sum()).item())

    deletion_cost = compute_visual_token_deletion_cost_scores(
        layer=scoring_layer,
        layer_input=hidden_states,
        hidden_normed=hidden_normed,
        attn_output=attn_output,
        attn_weights=attn_weights,
        image_start_idx=image_start_idx,
        image_end_idx=image_end_idx,
        query_indices=query_indices,
        norm_weight=norm_weight,
        combine_type=combine_type,
    ).clamp_min(1e-12)
    deletion_cost_probs = deletion_cost / deletion_cost.sum()

    return {
        "vision_share": vision_share,
        "attn_probs": question_to_vision,
        "deletion_cost_probs": deletion_cost_probs,
    }


def summarize_records(records: list[dict], ratio_grid: tuple[float, ...], attn_layer: int) -> dict:
    target = np.array([row["oracle_keep_ratio"] for row in records], dtype=np.float64)
    correlation_records = [
        row for row in records if any(result["exact_match"] > 0 for result in row["ratio_results"])
    ]
    correlation_target = np.array([row["oracle_keep_ratio"] for row in correlation_records], dtype=np.float64)
    metric_names = [
        "vision_share",
        "delete_cost_entropy_norm",
        "delete_cost_effective_ratio",
        "delete_cost_topk_mass_0125",
        "delete_cost_gini",
        "delete_cost_cover_ratio_050",
        "attn_entropy_norm",
        "attn_effective_ratio",
        "attn_topk_mass_0125",
        "attn_gini",
        "attn_cover_ratio_050",
        "spatial_dispersion",
        "redundancy_meanmax",
        "redundancy_meanpair",
        "ocr_token_count",
        "question_token_count",
    ]

    correlations = []
    for metric in metric_names:
        values = np.array([row[metric] for row in correlation_records], dtype=np.float64)
        correlations.append(
            {
                "metric": metric,
                "pearson": pearson_corr(values, correlation_target),
                "spearman": spearman_corr(values, correlation_target),
            }
        )

    correlations.sort(key=lambda row: abs(row["spearman"]), reverse=True)

    composite = []
    for row in correlation_records:
        value = (
            0.40 * row["delete_cost_cover_ratio_050"]
            + 0.25 * (1.0 - row["delete_cost_topk_mass_0125"])
            + 0.20 * row["question_token_count"]
            + 0.15 * (1.0 - row["redundancy_meanmax"])
        )
        composite.append(value)
    composite = np.array(composite, dtype=np.float64)

    ratio_distribution = {str(ratio): 0 for ratio in ratio_grid}
    for row in records:
        ratio_distribution[str(row["oracle_keep_ratio"])] += 1

    summary = {
        "num_records": len(records),
        "num_metric_correlation_records": len(correlation_records),
        "metric_correlation_filter": "at_least_one_correct_ratio",
        "attn_layer": attn_layer,
        "ratio_grid": list(ratio_grid),
        "mean_oracle_keep_ratio": float(target.mean()),
        "oracle_keep_ratio_distribution": ratio_distribution,
        "metric_correlations": correlations,
        "composite_metric": {
            "name": "0.40*delete_cost_cover_ratio_050 + 0.25*(1-delete_cost_topk_mass_0125) + 0.20*question_token_count + 0.15*(1-redundancy_meanmax)",
            "pearson": pearson_corr(composite, correlation_target),
            "spearman": spearman_corr(composite, correlation_target),
        },
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Search oracle keep ratios on TextVQA with the original llava_hf path.")
    parser.add_argument("--model-path", default="/home/user/czx/model/llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--dataset-split", default="validation")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--attn-layer", type=int, default=17)
    parser.add_argument("--norm-weight", type=float, default=1.0)
    parser.add_argument("--combine-type", default="add", choices=["add", "mul"])
    parser.add_argument("--max-new-tokens", type=int, default=8)
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

    dataset = load_dataset("lmms-lab/textvqa", split=args.dataset_split)
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
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    answer_processor = EvalAIAnswerProcessor()
    pad_embedding = model.get_input_embeddings().weight[processor.tokenizer.pad_token_id].detach().to(device)

    ratio_grid = tuple(float(ratio) for ratio in RATIO_GRID)
    records = list(processed.values())

    for dataset_index in tqdm(sampled_indices, desc="TextVQA oracle keep-ratio search"):
        if dataset_index in processed:
            continue

        doc = dataset[int(dataset_index)]
        prompt = build_prompt(doc["question"])
        inputs = processor(images=doc["image"].convert("RGB"), text=prompt, return_tensors="pt")
        inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}

        input_ids = inputs["input_ids"][0]
        image_mask = input_ids == model.config.image_token_index
        image_positions = torch.nonzero(image_mask, as_tuple=False).squeeze(-1)
        question_positions = torch.nonzero(~image_mask, as_tuple=False).squeeze(-1)
        question_positions = question_positions[question_positions > image_positions[-1]]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            full_embeddings, image_features = build_full_multimodal_embeddings(
                model=model,
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
            )
            score_payload = compute_scoring_signals(
                model=model,
                full_embeddings=full_embeddings,
                attention_mask=attention_mask,
                image_start_idx=int(image_positions[0].item()),
                image_end_idx=int(image_positions[-1].item()) + 1,
                query_indices=question_positions,
                layer_idx=args.attn_layer,
                norm_weight=args.norm_weight,
                combine_type=args.combine_type,
            )

        question_to_vision = score_payload["attn_probs"]
        deletion_cost_probs = score_payload["deletion_cost_probs"]
        vision_share = score_payload["vision_share"]

        feature_norm = F.normalize(image_features.float(), dim=-1)
        similarity = feature_norm @ feature_norm.T
        similarity.fill_diagonal_(-1.0)
        redundancy_meanmax = float(similarity.max(dim=-1).values.mean().item())
        valid_similarity = similarity[similarity > -0.5]
        redundancy_meanpair = float(valid_similarity.mean().item())

        grid_size = int(round(math.sqrt(question_to_vision.numel())))
        coords_y, coords_x = torch.meshgrid(
            torch.linspace(0.0, 1.0, steps=grid_size, device=device),
            torch.linspace(0.0, 1.0, steps=grid_size, device=device),
            indexing="ij",
        )
        coords = torch.stack([coords_x.reshape(-1), coords_y.reshape(-1)], dim=-1)
        centroid = (coords * question_to_vision.unsqueeze(-1)).sum(dim=0)
        spatial_dispersion = float((((coords - centroid) ** 2).sum(dim=-1) * question_to_vision).sum().item())

        question_token_count = int(question_positions.numel())
        ocr_token_count = int(len(doc.get("ocr_tokens", [])))

        text_embeddings = model.get_input_embeddings()(inputs["input_ids"])
        prefix_embeddings = text_embeddings[0, : image_positions[0]]
        suffix_embeddings = text_embeddings[0, image_positions[-1] + 1 :]

        variant_sequences = []
        variant_meta = []
        for ratio in ratio_grid:
            keep_tokens = max(1, int(round(ratio * image_features.shape[0])))
            keep_indices = torch.topk(deletion_cost_probs, k=keep_tokens).indices.sort().values
            sequence = torch.cat([prefix_embeddings, image_features[keep_indices], suffix_embeddings], dim=0)
            variant_sequences.append(sequence)
            variant_meta.append((ratio, keep_tokens))

        batch_embeds, batch_masks = build_left_padded_batch(variant_sequences, pad_embedding, device)

        with torch.no_grad():
            generations = model.generate(
                inputs_embeds=batch_embeds,
                attention_mask=batch_masks,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        predictions = processor.batch_decode(generations, skip_special_tokens=True)

        ratio_results = []
        successful_ratios = []
        best_ratio = None
        best_em = -1.0
        for (ratio, keep_tokens), prediction in zip(variant_meta, predictions):
            exact_match = compute_textvqa_accuracy(doc, prediction, answer_processor)
            ratio_results.append(
                {
                    "ratio": ratio,
                    "keep_tokens": keep_tokens,
                    "prediction": prediction,
                    "exact_match": exact_match,
                }
            )
            if exact_match > 0:
                successful_ratios.append(ratio)
            if exact_match > best_em or (exact_match == best_em and (best_ratio is None or ratio < best_ratio)):
                best_em = exact_match
                best_ratio = ratio

        record = {
            "dataset_index": int(dataset_index),
            "question_id": int(doc["question_id"]),
            "question": doc["question"],
            "oracle_keep_ratio": float(min(successful_ratios) if successful_ratios else 1.0),
            "smallest_best_ratio": float(best_ratio),
            "best_exact_match": float(best_em),
            "score_mode": "deletion_cost_ffn_jvp",
            "vision_share": vision_share,
            "delete_cost_entropy_norm": normalized_entropy(deletion_cost_probs),
            "delete_cost_effective_ratio": float(math.exp(-(deletion_cost_probs * deletion_cost_probs.log()).sum().item()) / deletion_cost_probs.numel()),
            "delete_cost_topk_mass_0125": float(torch.topk(deletion_cost_probs, k=max(1, int(round(0.125 * deletion_cost_probs.numel())))).values.sum().item()),
            "delete_cost_gini": gini_coefficient(deletion_cost_probs),
            "delete_cost_cover_ratio_050": cumulative_mass_ratio(deletion_cost_probs, 0.5),
            "attn_entropy_norm": normalized_entropy(question_to_vision),
            "attn_effective_ratio": float(math.exp(-(question_to_vision * question_to_vision.log()).sum().item()) / question_to_vision.numel()),
            "attn_topk_mass_0125": float(torch.topk(question_to_vision, k=max(1, int(round(0.125 * question_to_vision.numel())))).values.sum().item()),
            "attn_gini": gini_coefficient(question_to_vision),
            "attn_cover_ratio_050": cumulative_mass_ratio(question_to_vision, 0.5),
            "spatial_dispersion": spatial_dispersion,
            "redundancy_meanmax": redundancy_meanmax,
            "redundancy_meanpair": redundancy_meanpair,
            "ocr_token_count": ocr_token_count,
            "question_token_count": question_token_count,
            "ratio_results": ratio_results,
        }

        with output_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        processed[int(dataset_index)] = record
        records.append(record)

        del full_embeddings, question_to_vision, deletion_cost_probs, image_features, text_embeddings, batch_embeds, batch_masks, generations
        torch.cuda.empty_cache()

    ordered_records = [processed[idx] for idx in sampled_indices if idx in processed]
    summary = summarize_records(ordered_records, ratio_grid=ratio_grid, attn_layer=args.attn_layer)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
