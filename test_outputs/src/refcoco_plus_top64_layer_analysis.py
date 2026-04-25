import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import ImageDraw
from tqdm import tqdm
from transformers import AutoProcessor
from transformers.models.llama.modeling_llama import create_causal_mask as llama_create_causal_mask


REPO_ROOT = Path("/home/user/czx/open_source_proj/lmms-eval")
WORKDIR_ROOT = Path("/home/user/czx/MLLM_Token_Compression_Workdir")
MODEL_FILE = WORKDIR_ROOT / "src/LLaVA_infer_lmm_evals/transformers/modeling_llavav27_visual_token_wo_skip_layer_mmr_attn_norm_simi_threhold_norm_angle_ffn_after_dynamic_scoring_layer.py"
REFCOCO_UTILS_FILE = REPO_ROOT / "lmms_eval/tasks/refcoco+/utils.py"

PROMPT_PREFIX = (
    "Bounding box coordinates are specified in the format "
    "(top-left x, top-left y, bottom-right x, bottom-right y). "
    "All values are floating point numbers bounded between 0 and 1. "
    "Please provide the bounding box coordinate of the region this sentence describes: "
)


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_local_llava_class(model_file: Path):
    module_name = f"transformers.models.llava.{model_file.stem}"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, model_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load model file: {model_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        module = sys.modules[module_name]

    if not hasattr(module, "LlavaForConditionalGeneration"):
        raise AttributeError(f"{model_file} does not define LlavaForConditionalGeneration")
    return module.LlavaForConditionalGeneration


def load_refcoco_caption_utils(utils_file: Path):
    module_name = "lmms_eval_refcoco_plus_utils_for_outputs"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, utils_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load RefCOCO+ utils file: {utils_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        module = sys.modules[module_name]
    return module.COCO_METRICS, module.refcoco_aggregation_result


def build_prompt(doc: dict, prompt_source: str) -> str:
    if prompt_source == "question":
        text = str(doc.get("question", "")).strip()
    elif prompt_source == "expression":
        text = str(doc.get("query_text", "")).strip()
    elif prompt_source == "bbox_rec":
        text = f"{PROMPT_PREFIX}{str(doc.get('query_text', '')).strip()}"
    elif prompt_source == "lmms_eval_bbox":
        text = "Provide a short description for this region."
    else:
        raise ValueError(f"Unsupported prompt_source: {prompt_source}")

    if not text:
        raise ValueError(f"Empty prompt text for prompt_source={prompt_source}")
    return f"USER: <image>\n{text} ASSISTANT:"


def make_lmms_eval_bbox_visual(image, raw_bbox: Sequence[float]):
    boxed = image.copy().convert("RGB")
    x, y, w, h = [float(value) for value in raw_bbox]
    draw = ImageDraw.Draw(boxed)
    draw.rectangle([x, y, x + w, y + h], outline="red")
    return boxed


def parse_layers(layer_spec: str) -> List[int]:
    layers: List[int] = []
    for chunk in layer_spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            start, end = int(start_s), int(end_s)
            step = 1 if end >= start else -1
            layers.extend(range(start, end + step, step))
        else:
            layers.append(int(chunk))
    seen = set()
    ordered = []
    for layer in layers:
        if layer not in seen:
            ordered.append(layer)
            seen.add(layer)
    return ordered


def sample_indices(dataset_len: int, sample_size: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    indices = list(range(dataset_len))
    rng.shuffle(indices)
    return sorted(indices[: min(sample_size, dataset_len)])


def normalize_bbox_xywh(raw_bbox: Sequence[float], width: int, height: int) -> List[float]:
    x, y, w, h = [float(v) for v in raw_bbox]
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: width={width}, height={height}")
    return [
        max(0.0, min(1.0, x / width)),
        max(0.0, min(1.0, y / height)),
        max(0.0, min(1.0, (x + w) / width)),
        max(0.0, min(1.0, (y + h) / height)),
    ]


def project_bbox_xywh_to_processed_square(raw_bbox: Sequence[float], width: int, height: int, image_processor) -> List[float]:
    """Map original-image xywh bbox to the image processor's resized+center-cropped square coordinates."""
    x, y, w, h = [float(v) for v in raw_bbox]
    size = getattr(image_processor, "size", {}) or {}
    crop_size = getattr(image_processor, "crop_size", {}) or {}

    shortest_edge = size.get("shortest_edge")
    if shortest_edge is None:
        if "height" in size and "width" in size:
            shortest_edge = min(float(size["height"]), float(size["width"]))
        else:
            shortest_edge = float(min(width, height))
    crop_h = float(crop_size.get("height", shortest_edge))
    crop_w = float(crop_size.get("width", shortest_edge))

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: width={width}, height={height}")

    if getattr(image_processor, "do_resize", True):
        scale = float(shortest_edge) / float(min(width, height))
    else:
        scale = 1.0
    resized_w = float(width) * scale
    resized_h = float(height) * scale

    if getattr(image_processor, "do_center_crop", True):
        crop_left = max(0.0, (resized_w - crop_w) * 0.5)
        crop_top = max(0.0, (resized_h - crop_h) * 0.5)
    else:
        crop_left = 0.0
        crop_top = 0.0
        crop_w = resized_w
        crop_h = resized_h

    x1 = x * scale - crop_left
    y1 = y * scale - crop_top
    x2 = (x + w) * scale - crop_left
    y2 = (y + h) * scale - crop_top

    x1 = max(0.0, min(crop_w, x1))
    x2 = max(0.0, min(crop_w, x2))
    y1 = max(0.0, min(crop_h, y1))
    y2 = max(0.0, min(crop_h, y2))
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return [
        x1 / max(crop_w, 1e-12),
        y1 / max(crop_h, 1e-12),
        x2 / max(crop_w, 1e-12),
        y2 / max(crop_h, 1e-12),
    ]


def load_refcoco_plus_bbox_rec_rows(dataset_path: str, split: str, prompt_source: str) -> List[dict]:
    dataset = load_dataset(dataset_path, split=split)
    rows: List[dict] = []
    for source_index, doc in enumerate(tqdm(dataset, desc=f"Loading {dataset_path}/{split}")):
        image = doc["image"].convert("RGB")
        raw_bbox = [float(value) for value in doc["bbox"]]
        bbox = normalize_bbox_xywh(raw_bbox, image.width, image.height)
        answers = doc.get("answer", [])
        if isinstance(answers, str):
            answers = [answers]
        if prompt_source in {"question", "lmms_eval_bbox"}:
            row = dict(doc)
            row["image"] = make_lmms_eval_bbox_visual(image, raw_bbox) if prompt_source == "lmms_eval_bbox" else image
            row["bbox"] = bbox
            row["raw_bbox_xywh"] = raw_bbox
            row["answer"] = [str(answer) for answer in answers]
            row["query_text"] = "Provide a short description for this region." if prompt_source == "lmms_eval_bbox" else str(doc.get("question", ""))
            row["source_index"] = int(source_index)
            row["answer_index"] = None
            row["row_id"] = len(rows)
            row["image_width"] = int(image.width)
            row["image_height"] = int(image.height)
            rows.append(row)
        else:
            for answer_index, answer in enumerate(answers):
                row = dict(doc)
                row["image"] = image
                row["bbox"] = bbox
                row["raw_bbox_xywh"] = raw_bbox
                row["answer"] = str(answer)
                row["query_text"] = str(answer)
                row["source_index"] = int(source_index)
                row["answer_index"] = int(answer_index)
                row["row_id"] = len(rows)
                row["image_width"] = int(image.width)
                row["image_height"] = int(image.height)
                rows.append(row)
    return rows


def get_image_token_id(config) -> int:
    if hasattr(config, "image_token_id"):
        return int(config.image_token_id)
    if hasattr(config, "image_token_index"):
        return int(config.image_token_index)
    raise AttributeError("Model config has neither image_token_id nor image_token_index")


def prepare_multimodal_inputs(model, inputs: Dict[str, torch.Tensor]) -> dict:
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs.get("pixel_values")
    image_sizes = inputs.get("image_sizes")

    inputs_embeds = model.get_input_embeddings()(input_ids)
    image_features = None
    if pixel_values is not None:
        image_kwargs = {}
        if image_sizes is not None:
            image_kwargs["image_sizes"] = image_sizes
        image_features = model.model.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=None,
            vision_feature_select_strategy=None,
            **image_kwargs,
        )
        image_features = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        special_image_mask = model.model.get_placeholder_mask(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_features,
        )
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

    image_token_id = get_image_token_id(model.config)
    placeholder_mask = input_ids[0] == image_token_id
    placeholder_positions = torch.nonzero(placeholder_mask, as_tuple=False).squeeze(-1)
    if placeholder_positions.numel() == 0:
        raise ValueError("No image placeholder tokens found in input_ids.")

    image_start_idx = int(placeholder_positions.min().item())
    image_end_idx = int(placeholder_positions.max().item()) + 1
    if attention_mask is None:
        attention_mask = torch.ones((1, inputs_embeds.shape[1]), dtype=torch.long, device=inputs_embeds.device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "inputs_embeds": inputs_embeds,
        "image_start_idx": image_start_idx,
        "image_end_idx": image_end_idx,
        "num_visual_tokens": int(image_end_idx - image_start_idx),
    }


def get_query_indices(prepared: dict, attn_anchor: str) -> Optional[torch.Tensor]:
    if attn_anchor != "query":
        return None
    attention_mask = prepared["attention_mask"]
    seq_len = int(prepared["inputs_embeds"].shape[1])
    image_end_idx = int(prepared["image_end_idx"])
    valid_mask = torch.zeros(seq_len, device=attention_mask.device, dtype=torch.bool)
    if image_end_idx < seq_len:
        valid_mask[image_end_idx:] = True
    valid_mask = valid_mask & attention_mask[0].bool()
    query_indices = valid_mask.nonzero(as_tuple=False).squeeze(-1)
    return query_indices if query_indices.numel() > 0 else None


def run_full_language_logits(model, prepared: dict) -> torch.Tensor:
    inputs_embeds = prepared["inputs_embeds"]
    attention_mask = prepared["attention_mask"]
    seq_len = int(inputs_embeds.shape[1])
    device = inputs_embeds.device
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    outputs = model.model.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
    return model.lm_head(hidden_states[:, -1, :]).float()


def run_topk_pruned_language_logits(model, prepared: dict, keep_indices: torch.Tensor) -> torch.Tensor:
    inputs_embeds = prepared["inputs_embeds"]
    image_start_idx = int(prepared["image_start_idx"])
    image_end_idx = int(prepared["image_end_idx"])
    seq_len = int(inputs_embeds.shape[1])
    device = inputs_embeds.device

    keep_indices = keep_indices.to(device=device, dtype=torch.long).sort().values
    pre_visual = inputs_embeds[:, :image_start_idx, :]
    visual = inputs_embeds[:, image_start_idx:image_end_idx, :][:, keep_indices, :]
    post_visual = inputs_embeds[:, image_end_idx:, :]
    pruned_embeds = torch.cat([pre_visual, visual, post_visual], dim=1)

    pre_positions = torch.arange(image_start_idx, device=device, dtype=torch.long)
    visual_positions = keep_indices + image_start_idx
    post_positions = torch.arange(image_end_idx, seq_len, device=device, dtype=torch.long)
    position_ids = torch.cat([pre_positions, visual_positions, post_positions]).unsqueeze(0)
    attention_mask = torch.ones((1, pruned_embeds.shape[1]), device=device, dtype=torch.long)

    outputs = model.model.language_model(
        inputs_embeds=pruned_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
    return model.lm_head(hidden_states[:, -1, :]).float()


def build_pruned_embedding_batch(prepared: dict, keep_indices_by_layer: Dict[int, torch.Tensor]) -> Tuple[List[int], torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs_embeds = prepared["inputs_embeds"]
    image_start_idx = int(prepared["image_start_idx"])
    image_end_idx = int(prepared["image_end_idx"])
    seq_len = int(inputs_embeds.shape[1])
    device = inputs_embeds.device

    layer_ids = sorted(keep_indices_by_layer)
    pruned_embeds = []
    pruned_position_ids = []
    for layer_idx in layer_ids:
        keep_indices = keep_indices_by_layer[layer_idx].to(device=device, dtype=torch.long).sort().values
        pre_visual = inputs_embeds[:, :image_start_idx, :]
        visual = inputs_embeds[:, image_start_idx:image_end_idx, :][:, keep_indices, :]
        post_visual = inputs_embeds[:, image_end_idx:, :]
        pruned_embeds.append(torch.cat([pre_visual, visual, post_visual], dim=1).squeeze(0))

        pre_positions = torch.arange(image_start_idx, device=device, dtype=torch.long)
        visual_positions = keep_indices + image_start_idx
        post_positions = torch.arange(image_end_idx, seq_len, device=device, dtype=torch.long)
        pruned_position_ids.append(torch.cat([pre_positions, visual_positions, post_positions]))

    inputs_embeds_batch = torch.stack(pruned_embeds, dim=0)
    position_ids_batch = torch.stack(pruned_position_ids, dim=0)
    attention_mask = torch.ones(
        (len(layer_ids), inputs_embeds_batch.shape[1]),
        device=device,
        dtype=prepared["attention_mask"].dtype,
    )
    return layer_ids, inputs_embeds_batch, attention_mask, position_ids_batch


def greedy_generate_from_embeds(
    model,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int],
    pad_token_id: Optional[int],
) -> torch.Tensor:
    batch_size, seq_len = inputs_embeds.shape[:2]
    device = inputs_embeds.device
    if pad_token_id is None:
        pad_token_id = eos_token_id if eos_token_id is not None else 0

    cache_position = torch.arange(seq_len, device=device, dtype=torch.long)
    outputs = model.model.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
        cache_position=cache_position,
    )
    logits = model.lm_head(outputs.last_hidden_state[:, -1, :])
    next_tokens = torch.argmax(logits, dim=-1)
    generated = []
    finished = torch.zeros(batch_size, device=device, dtype=torch.bool)
    past_key_values = outputs.past_key_values
    max_prompt_position = position_ids.max(dim=1).values

    for step in range(max_new_tokens):
        tokens_to_store = torch.where(
            finished,
            torch.full_like(next_tokens, int(pad_token_id)),
            next_tokens,
        )
        generated.append(tokens_to_store)
        if eos_token_id is not None:
            finished = finished | (next_tokens == int(eos_token_id))
        if step == max_new_tokens - 1 or bool(finished.all().item()):
            break

        decode_embeds = model.get_input_embeddings()(tokens_to_store).unsqueeze(1)
        decode_attention_mask = torch.ones(
            (batch_size, seq_len + step + 1),
            device=device,
            dtype=attention_mask.dtype,
        )
        decode_position_ids = (max_prompt_position + step + 1).view(batch_size, 1)
        decode_cache_position = torch.tensor([seq_len + step], device=device, dtype=torch.long)
        outputs = model.model.language_model(
            inputs_embeds=decode_embeds,
            attention_mask=decode_attention_mask,
            position_ids=decode_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=decode_cache_position,
        )
        logits = model.lm_head(outputs.last_hidden_state[:, -1, :])
        next_tokens = torch.argmax(logits, dim=-1)
        past_key_values = outputs.past_key_values

    return torch.stack(generated, dim=1)


def decode_generated(processor, generated_ids: torch.Tensor) -> List[str]:
    return [
        text.strip()
        for text in processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    ]


def extract_layer_scores(model, prepared: dict, scoring_layers: Sequence[int], attn_anchor: str, pool_type: str) -> Dict[int, dict]:
    requested = sorted(set(int(layer) for layer in scoring_layers))
    layers = model.model.language_model.layers
    num_layers = len(layers)
    requested = [layer for layer in requested if 0 <= layer < num_layers]
    if not requested:
        raise ValueError(f"No valid scoring layers in {scoring_layers}; model has {num_layers} layers")

    inputs_embeds = prepared["inputs_embeds"]
    attention_mask = prepared["attention_mask"]
    image_start_idx = int(prepared["image_start_idx"])
    image_end_idx = int(prepared["image_end_idx"])
    seq_len = int(inputs_embeds.shape[1])
    device = inputs_embeds.device
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    cache_position = torch.arange(seq_len, device=device, dtype=torch.long)
    causal_mask = llama_create_causal_mask(
        config=model.model.language_model.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=position_ids,
    )
    pos_emb = model.model.language_model.rotary_emb(inputs_embeds, position_ids)
    query_indices = get_query_indices(prepared, attn_anchor)

    hidden_states = inputs_embeds
    layer_outputs: Dict[int, dict] = {}
    max_requested = max(requested)
    requested_set = set(requested)
    for layer_idx in range(max_requested + 1):
        if layer_idx in requested_set:
            raw_attn, importance_scores, attention_quality = model.model._get_visual_token_attention_scores(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                scoring_layer_idx=layer_idx,
                image_start_idx=image_start_idx,
                image_end_idx=image_end_idx,
                importance_mode="attn",
                ffn_propagation="jvp",
                attn_anchor=attn_anchor,
                pool_type=pool_type,
                query_indices=query_indices,
            )
            layer_outputs[layer_idx] = {
                "raw_attn": raw_attn.detach(),
                "importance_scores": importance_scores.detach(),
                "attention_quality": float(attention_quality.detach().float().item()),
            }

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
    return layer_outputs


def token_bbox_geometry(num_visual_tokens: int, bbox: Sequence[float]) -> dict:
    side = int(round(math.sqrt(num_visual_tokens)))
    if side * side != num_visual_tokens:
        raise ValueError(f"Expected square visual grid, got {num_visual_tokens} tokens")

    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    xs = torch.arange(side, dtype=torch.float32)
    ys = torch.arange(side, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    cell_x1 = (grid_x.reshape(-1) / side).numpy()
    cell_y1 = (grid_y.reshape(-1) / side).numpy()
    cell_x2 = ((grid_x.reshape(-1) + 1.0) / side).numpy()
    cell_y2 = ((grid_y.reshape(-1) + 1.0) / side).numpy()
    center_x = ((cell_x1 + cell_x2) * 0.5)
    center_y = ((cell_y1 + cell_y2) * 0.5)

    inter_w = np.maximum(0.0, np.minimum(cell_x2, x2) - np.maximum(cell_x1, x1))
    inter_h = np.maximum(0.0, np.minimum(cell_y2, y2) - np.maximum(cell_y1, y1))
    overlap_area = inter_w * inter_h
    center_inside = (center_x >= x1) & (center_x <= x2) & (center_y >= y1) & (center_y <= y2)
    overlap_positive = overlap_area > 0.0

    return {
        "side": side,
        "cell_area": float(1.0 / (side * side)),
        "bbox_area": float(bbox_area),
        "overlap_area": overlap_area.astype(np.float64),
        "overlap_positive": overlap_positive.astype(bool),
        "center_inside": center_inside.astype(bool),
    }


def binary_auc(scores: np.ndarray, positives: np.ndarray) -> Optional[float]:
    positives = positives.astype(bool)
    pos_scores = scores[positives]
    neg_scores = scores[~positives]
    if pos_scores.size == 0 or neg_scores.size == 0:
        return None
    greater = (pos_scores[:, None] > neg_scores[None, :]).sum(dtype=np.float64)
    equal = (pos_scores[:, None] == neg_scores[None, :]).sum(dtype=np.float64)
    return float((greater + 0.5 * equal) / (pos_scores.size * neg_scores.size))


def score_bbox_alignment(scores_tensor: torch.Tensor, geometry: dict, topk: int, prefix: str) -> dict:
    scores = scores_tensor.detach().float().cpu().numpy().astype(np.float64)
    if scores.size == 0:
        return {}
    keep_k = int(min(max(1, topk), scores.size))
    top_indices = np.argsort(-scores, kind="mergesort")[:keep_k]
    nonnegative_scores = np.maximum(scores, 0.0)
    score_total = float(nonnegative_scores.sum())
    overlap_area = geometry["overlap_area"]
    overlap_positive = geometry["overlap_positive"]
    center_inside = geometry["center_inside"]
    bbox_area = max(float(geometry["bbox_area"]), 1e-12)
    cell_area = float(geometry["cell_area"])
    top_overlap_area = float(overlap_area[top_indices].sum())

    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[np.argsort(-scores, kind="mergesort")] = np.arange(1, scores.size + 1, dtype=np.float64)
    positive_ranks = ranks[overlap_positive]
    center_ranks = ranks[center_inside]

    metrics = {
        f"{prefix}_topk_precision_overlap": float(overlap_positive[top_indices].mean()),
        f"{prefix}_topk_precision_center": float(center_inside[top_indices].mean()),
        f"{prefix}_topk_bbox_recall": float(min(1.0, top_overlap_area / bbox_area)),
        f"{prefix}_topk_cell_iou": float(top_overlap_area / max(bbox_area + keep_k * cell_area - top_overlap_area, 1e-12)),
        f"{prefix}_positive_token_count": int(overlap_positive.sum()),
        f"{prefix}_center_token_count": int(center_inside.sum()),
        f"{prefix}_mean_positive_rank_pct": float(positive_ranks.mean() / scores.size) if positive_ranks.size else None,
        f"{prefix}_mean_center_rank_pct": float(center_ranks.mean() / scores.size) if center_ranks.size else None,
        f"{prefix}_auc_overlap": binary_auc(scores, overlap_positive),
        f"{prefix}_auc_center": binary_auc(scores, center_inside),
    }
    if score_total > 1e-12:
        metrics[f"{prefix}_score_mass_overlap"] = float(nonnegative_scores[overlap_positive].sum() / score_total)
        metrics[f"{prefix}_score_mass_center"] = float(nonnegative_scores[center_inside].sum() / score_total)
    else:
        metrics[f"{prefix}_score_mass_overlap"] = None
        metrics[f"{prefix}_score_mass_center"] = None
    return metrics


def distribution_metrics(full_logits: torch.Tensor, pruned_logits: torch.Tensor) -> dict:
    full = full_logits.float().squeeze(0)
    pruned = pruned_logits.float().squeeze(0)
    full_logp = F.log_softmax(full, dim=-1)
    pruned_logp = F.log_softmax(pruned, dim=-1)
    full_p = full_logp.exp()
    pruned_p = pruned_logp.exp()
    mixture = 0.5 * (full_p + pruned_p)
    mixture_logp = mixture.clamp_min(1e-12).log()

    full_top1 = int(torch.argmax(full).item())
    pruned_top1 = int(torch.argmax(pruned).item())
    k = min(5, full.numel())
    full_top5 = set(torch.topk(full, k=k).indices.tolist())
    pruned_top5 = set(torch.topk(pruned, k=k).indices.tolist())

    return {
        "dist_kl_full_to_pruned": float((full_p * (full_logp - pruned_logp)).sum().item()),
        "dist_kl_pruned_to_full": float((pruned_p * (pruned_logp - full_logp)).sum().item()),
        "dist_js": float(0.5 * (full_p * (full_logp - mixture_logp)).sum().item() + 0.5 * (pruned_p * (pruned_logp - mixture_logp)).sum().item()),
        "dist_total_variation": float(0.5 * torch.abs(full_p - pruned_p).sum().item()),
        "dist_logit_cosine": float(F.cosine_similarity(full, pruned, dim=0).item()),
        "dist_top1_match": bool(full_top1 == pruned_top1),
        "dist_top5_overlap": int(len(full_top5 & pruned_top5)),
        "dist_full_top1_id": full_top1,
        "dist_pruned_top1_id": pruned_top1,
        "dist_full_top1_prob": float(full_p[full_top1].item()),
        "dist_pruned_prob_on_full_top1": float(pruned_p[full_top1].item()),
    }


def rounded_float_list(tensor: torch.Tensor, decimals: int = 8) -> List[float]:
    values = tensor.detach().float().cpu().numpy().astype(np.float32)
    return np.round(values, decimals=decimals).tolist()


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    arr = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(arr)) if arr else None


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
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def pearson_corr(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 2 or np.std(x_arr) < 1e-12 or np.std(y_arr) < 1e-12:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 2:
        return None
    return pearson_corr(rankdata(x_arr[mask]), rankdata(y_arr[mask]))


def as_answer_list(answer):
    if isinstance(answer, list):
        return [str(item) for item in answer]
    return [str(answer)]


def build_caption_results(records: List[dict], layer_idx: Optional[int] = None) -> List[dict]:
    results = []
    for record in records:
        if layer_idx is None:
            pred = record.get("baseline_prediction")
        else:
            pred = None
            for row in record.get("layer_metrics", []):
                if int(row["layer_idx"]) == int(layer_idx):
                    pred = row.get("prediction")
                    break
        if pred is None:
            continue
        results.append(
            {
                "answer": as_answer_list(record.get("answer", [])),
                "pred": str(pred),
                "ann_id": record.get("question_id", record.get("dataset_index")),
            }
        )
    return results


def compute_caption_scores(records: List[dict], caption_metrics: Sequence[str], aggregation_fn, layer_idx: Optional[int] = None) -> dict:
    results = build_caption_results(records, layer_idx=layer_idx)
    if not results:
        return {}
    scores = {}
    for metric_name in caption_metrics:
        scores[metric_name] = float(aggregation_fn(results, metric_name))
    return scores


def summarize_records(records: List[dict], caption_metrics: Optional[Sequence[str]] = None, aggregation_fn=None) -> dict:
    by_layer: Dict[int, List[dict]] = {}
    for record in records:
        for layer_row in record.get("layer_metrics", []):
            by_layer.setdefault(int(layer_row["layer_idx"]), []).append(layer_row)

    numeric_metric_names = [
        "attention_quality",
        "importance_topk_precision_overlap",
        "importance_topk_precision_center",
        "importance_topk_bbox_recall",
        "importance_topk_cell_iou",
        "importance_mean_positive_rank_pct",
        "importance_auc_overlap",
        "importance_auc_center",
        "importance_score_mass_overlap",
        "raw_attn_topk_precision_overlap",
        "raw_attn_topk_bbox_recall",
        "raw_attn_auc_overlap",
        "raw_attn_score_mass_overlap",
        "dist_kl_full_to_pruned",
        "dist_js",
        "dist_total_variation",
        "dist_logit_cosine",
        "dist_pruned_prob_on_full_top1",
    ]
    layer_summaries = []
    for layer_idx in sorted(by_layer):
        rows = by_layer[layer_idx]
        item = {
            "layer_idx": int(layer_idx),
            "num_records": int(len(rows)),
            "dist_top1_match_rate": mean_or_none([1.0 if row.get("dist_top1_match") else 0.0 for row in rows]),
            "dist_mean_top5_overlap": mean_or_none([row.get("dist_top5_overlap") for row in rows]),
        }
        for metric_name in numeric_metric_names:
            item[f"mean_{metric_name}"] = mean_or_none([row.get(metric_name) for row in rows])
        if caption_metrics and aggregation_fn is not None:
            caption_scores = compute_caption_scores(records, caption_metrics, aggregation_fn, layer_idx=layer_idx)
            for metric_name, score in caption_scores.items():
                item[f"performance_{metric_name}"] = score
        layer_summaries.append(item)

    def best_by(metric_name: str, higher: bool = True) -> Optional[dict]:
        valid = [row for row in layer_summaries if row.get(metric_name) is not None]
        if not valid:
            return None
        return max(valid, key=lambda row: row[metric_name]) if higher else min(valid, key=lambda row: row[metric_name])

    mean_importance_recall = [row.get("mean_importance_topk_bbox_recall") for row in layer_summaries]
    mean_importance_auc = [row.get("mean_importance_auc_overlap") for row in layer_summaries]
    mean_js = [row.get("mean_dist_js") for row in layer_summaries]
    mean_kl = [row.get("mean_dist_kl_full_to_pruned") for row in layer_summaries]

    baseline_caption_scores = {}
    if caption_metrics and aggregation_fn is not None:
        baseline_caption_scores = compute_caption_scores(records, caption_metrics, aggregation_fn, layer_idx=None)

    summary = {
        "num_records": int(len(records)),
        "layer_summaries": layer_summaries,
        "best_query_alignment_by_importance_top64_bbox_recall": best_by("mean_importance_topk_bbox_recall", True),
        "best_query_alignment_by_importance_auc_overlap": best_by("mean_importance_auc_overlap", True),
        "best_distribution_preservation_by_js": best_by("mean_dist_js", False),
        "best_distribution_preservation_by_kl": best_by("mean_dist_kl_full_to_pruned", False),
        "layer_level_spearman_importance_recall_vs_negative_js": spearman_corr(mean_importance_recall, [-v if v is not None else np.nan for v in mean_js]),
        "layer_level_spearman_importance_auc_vs_negative_js": spearman_corr(mean_importance_auc, [-v if v is not None else np.nan for v in mean_js]),
        "layer_level_spearman_importance_recall_vs_negative_kl": spearman_corr(mean_importance_recall, [-v if v is not None else np.nan for v in mean_kl]),
        "primary_query_alignment_metric": "mean_importance_topk_bbox_recall",
        "primary_distribution_preservation_metric": "mean_dist_js",
    }
    for metric_name, score in baseline_caption_scores.items():
        summary[f"baseline_performance_{metric_name}"] = score
        summary[f"best_performance_by_{metric_name}"] = best_by(f"performance_{metric_name}", True)
    return summary


def load_processed_records(output_jsonl: Path) -> Dict[int, dict]:
    processed: Dict[int, dict] = {}
    if not output_jsonl.exists():
        return processed
    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            processed[int(row["dataset_index"])] = row
    return processed


def main():
    parser = argparse.ArgumentParser(
        description=(
            "RefCOCO+ layer-wise top-64 visual-token pruning analysis. "
            "For each layer, compute importance_scores with combine_type=add, norm_weight=1.0, "
            "importance_mode=attn; take direct top-k text-to-visual tokens and compare them with the ground-truth bbox."
        )
    )
    parser.add_argument("--model-path", default="/home/user/czx/model/llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--dataset-path", default="lmms-lab/RefCOCOPlus")
    parser.add_argument("--dataset-split", default="val")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--layers", default="0-31")
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--caption-metrics", default="Bleu_4,Bleu_3,Bleu_2,Bleu_1,METEOR,ROUGE_L,CIDEr")
    parser.add_argument("--prompt-source", default="question", choices=["question", "expression", "bbox_rec", "lmms_eval_bbox"])
    parser.add_argument("--attn-anchor", default="query", choices=["last", "query"])
    parser.add_argument("--pool-type", default="max", choices=["avg", "max"])
    parser.add_argument(
        "--store-score-distribution",
        action="store_true",
        help="Store per-layer full importance score vectors and top-k indices in records.jsonl.",
    )
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

    rows = load_refcoco_plus_bbox_rec_rows(args.dataset_path, args.dataset_split, args.prompt_source)
    sampled_indices = sample_indices(len(rows), args.sample_size, args.sample_seed)
    sample_index_json.write_text(json.dumps(sampled_indices, ensure_ascii=False, indent=2), encoding="utf-8")

    requested_layers = parse_layers(args.layers)
    processed = load_processed_records(output_jsonl)
    caption_metrics = [item.strip() for item in args.caption_metrics.split(",") if item.strip()]
    supported_caption_metrics, refcoco_aggregation = load_refcoco_caption_utils(REFCOCO_UTILS_FILE)
    unsupported = [metric_name for metric_name in caption_metrics if metric_name not in supported_caption_metrics]
    if unsupported:
        raise ValueError(f"Unsupported RefCOCO+ caption metrics: {unsupported}; supported={supported_caption_metrics}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LlavaForConditionalGeneration = load_local_llava_class(MODEL_FILE)
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
        attn_implementation="eager",
    )
    if device.type == "cpu":
        model.to(device)
    model.eval()

    for dataset_index in tqdm(sampled_indices, desc="RefCOCO+ top-64 layer analysis"):
        if int(dataset_index) in processed:
            continue

        doc = rows[int(dataset_index)]
        prompt = build_prompt(doc, args.prompt_source)
        inputs = processor(images=doc["image"], text=prompt, return_tensors="pt")
        inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}

        with torch.no_grad():
            prepared = prepare_multimodal_inputs(model, inputs)
            num_visual_tokens = int(prepared["num_visual_tokens"])
            keep_tokens = min(max(1, int(args.topk)), num_visual_tokens)
            processed_bbox = project_bbox_xywh_to_processed_square(
                doc["raw_bbox_xywh"],
                int(doc["image_width"]),
                int(doc["image_height"]),
                processor.image_processor,
            )
            geometry = token_bbox_geometry(num_visual_tokens, processed_bbox)
            layer_scores = extract_layer_scores(
                model=model,
                prepared=prepared,
                scoring_layers=requested_layers,
                attn_anchor=args.attn_anchor,
                pool_type=args.pool_type,
            )
            full_logits = run_full_language_logits(model, prepared)
            full_position_ids = torch.arange(
                prepared["inputs_embeds"].shape[1],
                device=prepared["inputs_embeds"].device,
                dtype=torch.long,
            ).unsqueeze(0)
            baseline_generated_ids = greedy_generate_from_embeds(
                model=model,
                inputs_embeds=prepared["inputs_embeds"],
                attention_mask=prepared["attention_mask"],
                position_ids=full_position_ids,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
            baseline_prediction = decode_generated(processor, baseline_generated_ids)[0]

            layer_metrics = []
            keep_indices_by_layer = {}
            for layer_idx in sorted(layer_scores):
                raw_attn = layer_scores[layer_idx]["raw_attn"]
                importance_scores = layer_scores[layer_idx]["importance_scores"]
                keep_indices = torch.topk(importance_scores.float(), k=keep_tokens).indices
                keep_indices_by_layer[int(layer_idx)] = keep_indices
                pruned_logits = run_topk_pruned_language_logits(model, prepared, keep_indices)

                metric_row = {
                    "layer_idx": int(layer_idx),
                    "num_visual_tokens": int(num_visual_tokens),
                    "keep_tokens": int(keep_tokens),
                    "attention_quality": float(layer_scores[layer_idx]["attention_quality"]),
                }
                metric_row.update(score_bbox_alignment(importance_scores, geometry, keep_tokens, "importance"))
                metric_row.update(score_bbox_alignment(raw_attn, geometry, keep_tokens, "raw_attn"))
                metric_row.update(distribution_metrics(full_logits, pruned_logits))
                if args.store_score_distribution:
                    metric_row["importance_topk_indices"] = sorted(int(idx) for idx in keep_indices.detach().cpu().tolist())
                    metric_row["importance_scores"] = rounded_float_list(importance_scores)
                    metric_row["raw_attn_scores"] = rounded_float_list(raw_attn)
                layer_metrics.append(metric_row)
                del pruned_logits

            if keep_indices_by_layer and args.max_new_tokens > 0:
                batch_layer_ids, pruned_embeds, pruned_attention_mask, pruned_position_ids = build_pruned_embedding_batch(
                    prepared,
                    keep_indices_by_layer,
                )
                pruned_generated_ids = greedy_generate_from_embeds(
                    model=model,
                    inputs_embeds=pruned_embeds,
                    attention_mask=pruned_attention_mask,
                    position_ids=pruned_position_ids,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
                pruned_predictions = decode_generated(processor, pruned_generated_ids)
                prediction_by_layer = {
                    int(layer_idx): prediction
                    for layer_idx, prediction in zip(batch_layer_ids, pruned_predictions)
                }
                for metric_row in layer_metrics:
                    metric_row["prediction"] = prediction_by_layer.get(int(metric_row["layer_idx"]), "")

        record = {
            "dataset_index": int(dataset_index),
            "row_id": int(doc["row_id"]),
            "source_index": int(doc["source_index"]),
            "answer_index": int(doc["answer_index"]) if doc["answer_index"] is not None else None,
            "question_id": int(doc["question_id"]) if "question_id" in doc else None,
            "question": doc.get("question"),
            "query_text": doc.get("query_text"),
            "answer": doc["answer"],
            "baseline_prediction": baseline_prediction,
            "bbox": [float(v) for v in doc["bbox"]],
            "processed_bbox": [float(v) for v in processed_bbox],
            "raw_bbox_xywh": [float(v) for v in doc["raw_bbox_xywh"]],
            "image_width": int(doc["image_width"]),
            "image_height": int(doc["image_height"]),
            "num_visual_tokens": int(num_visual_tokens),
            "topk": int(keep_tokens),
            "attn_anchor": args.attn_anchor,
            "pool_type": args.pool_type,
            "prompt_source": args.prompt_source,
            "layer_metrics": layer_metrics,
        }
        with output_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        processed[int(dataset_index)] = record
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ordered_records = [processed[idx] for idx in sampled_indices if idx in processed]
    summary = summarize_records(ordered_records, caption_metrics=caption_metrics, aggregation_fn=refcoco_aggregation)
    summary.update(
        {
            "dataset_path": args.dataset_path,
            "dataset_split": args.dataset_split,
            "sample_size": int(len(sampled_indices)),
            "sample_seed": int(args.sample_seed),
            "requested_layers": requested_layers,
            "topk": int(args.topk),
            "prompt_source": args.prompt_source,
            "attn_anchor": args.attn_anchor,
            "pool_type": args.pool_type,
            "model_path": args.model_path,
            "model_file": str(MODEL_FILE),
            "max_new_tokens": args.max_new_tokens,
            "caption_metrics": caption_metrics,
            "importance_mode": "attn",
            "combine_type": "add",
            "norm_weight": 1.0,
            "pruning_rule": "direct_topk_importance_scores",
            "score_distribution_stored": bool(args.store_score_distribution),
            "forward_scoring": "use_cache_false_prefix_forward_to_score_layer",
            "forward_pruned": "topk_visual_tokens_then_full_use_cache_false_forward_for_logits_and_greedy_generation",
            "full_distribution_reference": "unpruned_use_cache_false_forward",
            "bbox_alignment_coordinates": "processor_resized_center_cropped_image",
        }
    )
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
