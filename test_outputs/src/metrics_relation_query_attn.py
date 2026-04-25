import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

def load_data(layer_idx, task="gqa", base_path="/home/user/czx/open_source_proj/lmms-eval/outputs/logs/llava_dynamic_gqa"):
    """
    加载指定层的统计数据和评估结果。
    假设目录结构为: base_path/scoring_layer_{idx}/...
    """
    layer_dir = os.path.join(base_path, f"scoring_layer_{layer_idx}")
    
    # 1. 加载 Metrics Stats (var, cv, iqr)
    metrics_path = os.path.join(layer_dir, f"metrics_layer_{layer_idx}.json")
    if not os.path.exists(metrics_path):
        print(f"Warning: Metrics file not found: {metrics_path}")
        return None

    with open(metrics_path, 'r') as f:
        metrics_data = json.load(f)
    
    # 提取需要的统计量 (这里以 mean 为例，你也可以提取 median 或 var)
    stats = metrics_data.get(f"layer_{layer_idx}_stats", {})
    
    # 2. 加载 Evaluation Results (exact_match)
    # 假设结果文件在 scoring_layer_{idx}/llava-hf__llava-1.5-7b-hf/ 下
    # 并且文件名包含 _results.json。这里我们需要找到对应的结果文件。
    # 简单起见，假设你有一个固定的结果文件或者在该目录下只有一个 results 文件
    results_dir = os.path.join(layer_dir, "llava-hf__llava-1.5-7b-hf")
    results_path = None
    
    if os.path.exists(results_dir):
        for fname in os.listdir(results_dir):
            if fname.endswith("_results.json"):
                results_path = os.path.join(results_dir, fname)
                break
    
    if not results_path or not os.path.exists(results_path):
        print(f"Warning: Results file not found in: {results_dir}")
        return None

    with open(results_path, 'r') as f:
        results_data = json.load(f)
    
    # 提取性能指标
    # 注意：根据提供的 results.json 示例，结构是 "results": { "gqa": { "exact_match,none": ... } }
    try:
        # 尝试自动获取对应的 metric key，优先匹配 pope 和 exact_match
        res_dict = results_data["results"][task]
        if "pope_accuracy,none" in res_dict:
            performance = res_dict["pope_accuracy,none"]
        elif "exact_match,none" in res_dict:
            performance = res_dict["exact_match,none"]
        elif "mme_perception_score,none" in res_dict:
            performance = res_dict["mme_perception_score,none"]
            # 如果没有常用key，尝试获取第一个value
        elif "gpt_eval_score,none" in res_dict:
            performance = res_dict["gpt_eval_score,none"]
        else:
            raise KeyError
    except KeyError:
        print(f"Warning: Could not extract performance metric from {results_path}")
        return None

    # 定义基础指标后缀
    metric_suffixes = [
        "var_mean", "cv_mean", "iqr_mean", 
        "gini_mean", "hoyer_mean", "kurtosis_mean", 
        "entropy_mean", "spatial_entropy_mean"
    ]

    result = {
        "layer_idx": layer_idx,
        "performance": performance
    }

    # 提取 attention_quality 和 top_ratio metrics
    if "attention_quality_mean" in stats:
        result["attention_quality_mean"] = stats["attention_quality_mean"]
    if "top_ratio_attention_quality_mean" in stats:
        result["top_ratio_attention_quality_mean"] = stats["top_ratio_attention_quality_mean"]
    if "top_ratio_importance_quality_mean" in stats:
        result["top_ratio_importance_quality_mean"] = stats["top_ratio_importance_quality_mean"]

    # 提取 importance 和 raw_attn 统计量
    for suffix in metric_suffixes:
        # Handle importance (优先使用带前缀的 key，如果不存在则尝试兼容旧格式)
        imp_key = f"importance_{suffix}"
        if imp_key in stats:
            result[imp_key] = stats[imp_key]
        elif suffix in stats:
            result[imp_key] = stats[suffix]
        else:
            result[imp_key] = np.nan
            
        # Handle raw_attn
        raw_key = f"raw_attn_{suffix}"
        result[raw_key] = stats.get(raw_key, np.nan)

    return result

def plot_correlations(df, setting_name, task, prefixes, metric_suffixes, enable_log=False, additional_metrics=None):
    """通用相关性绘图函数"""
    correlation_stats = {}

    for prefix in prefixes:
        rows = 2
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(24, 10))
        axes = axes.flatten()

        plot_exists = False
        for i, suffix in enumerate(metric_suffixes):
            if i >= len(axes):
                break
            metric = f"{prefix}_{suffix}"
            if metric not in df.columns:
                continue

            ax = axes[i]
            x = df[metric].astype(float)
            y = df["performance"].astype(float)

            xlabel_text = metric
            if enable_log:
                # 取 Log 前处理非正值，避免 error
                x = x.where(x > 0, np.nan)
                x = np.log(x)
                xlabel_text = f"Log({metric})"

            valid_mask = ~np.isnan(x) & ~np.isnan(y)
            x_clean = x[valid_mask]
            y_clean = y[valid_mask]

            if len(x_clean) < 1:
                ax.set_title(f"{metric} (No Data)")
                continue

            plot_exists = True

            if len(x_clean) > 1:
                corr, p_value = pearsonr(x_clean, y_clean)
                correlation_stats[metric] = {"r": corr, "p": p_value}
                title = f"{metric} vs Performance\nR={corr:.3f}, p={p_value:.3f}"
            else:
                title = f"{metric} vs Performance"

            ax.scatter(x, y, alpha=0.7)
            ax.set_xlabel(xlabel_text)
            ax.set_ylabel("Performance Score")
            ax.set_title(title)
            ax.grid(True)

            for j, txt in enumerate(df["layer_idx"]):
                if j < len(x) and not np.isnan(x[j]) and not np.isnan(y[j]):
                    ax.annotate(txt, (x[j], y[j]))

        for j in range(len(metric_suffixes), len(axes)):
            fig.delaxes(axes[j])

        if plot_exists:
            plt.tight_layout()
            log_suffix = "_log" if enable_log else ""
            output_plot_path = f"/home/user/czx/open_source_proj/lmms-eval/outputs/src_output/{setting_name}/correlation_plots_{prefix}_{task}{log_suffix}.png"
            os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
            plt.savefig(output_plot_path)
            print(f"Correlation plots for {setting_name} - {prefix} saved to {output_plot_path}")
        plt.close(fig)

    # 处理 Additional Metrics (如 Quality, Ratios 等)
    if additional_metrics:
        rows = 1
        cols = 3  # 假设指标数量不多，用 1x3 网格
        fig, axes = plt.subplots(rows, cols, figsize=(18, 6))
        axes = axes.flatten()

        plot_exists = False
        for i, config in enumerate(additional_metrics):
            if i >= len(axes):
                break
            
            # 解析配置 (支持元组或仅列名)
            if isinstance(config, (tuple, list)):
                metric = config[0]
                title_alias = config[1] if len(config) > 1 else metric
                color = config[2] if len(config) > 2 else None
            else:
                metric = config
                title_alias = metric
                color = None

            if metric not in df.columns:
                continue

            ax = axes[i]
            x = df[metric].astype(float)
            y = df["performance"].astype(float)

            xlabel_text = title_alias
            
            # 处理 Additional Metrics 时，如果开启了 log，则先取 log
            if enable_log:
                x = x.where(x > 0, np.nan)
                x = np.log(x)
                xlabel_text = f"Log({title_alias})"

            valid_mask = ~np.isnan(x) & ~np.isnan(y)
            x_clean = x[valid_mask]
            y_clean = y[valid_mask]

            if len(x_clean) < 1:
                ax.set_title(f"{title_alias} (No Data)")
                continue

            plot_exists = True

            if len(x_clean) > 1:
                corr, p_value = pearsonr(x_clean, y_clean)
                correlation_stats[metric] = {"r": corr, "p": p_value}
                title = f"{title_alias} vs Performance\nR={corr:.3f}, p={p_value:.3f}"
            else:
                title = f"{title_alias} vs Performance"

            ax.scatter(x, y, alpha=0.7, c=color if color else 'blue')
            ax.set_xlabel(xlabel_text)
            ax.set_ylabel("Performance Score")
            ax.set_title(title)
            ax.grid(True)

            for j, txt in enumerate(df["layer_idx"]):
                if j < len(x) and not np.isnan(x[j]) and not np.isnan(y[j]):
                    ax.annotate(txt, (x[j], y[j]))

        for j in range(len(additional_metrics), len(axes)):
            fig.delaxes(axes[j])

        if plot_exists:
            plt.tight_layout()
            # 保存额外的相关性图
            output_plot_path = f"/home/user/czx/open_source_proj/lmms-eval/outputs/src_output/{setting_name}/correlation_plots_quality_metrics_{task}.png"
            os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
            plt.savefig(output_plot_path)
            print(f"Correlation plots for {setting_name} - Additional Metrics saved to {output_plot_path}")
        plt.close(fig)

    return correlation_stats

def _to_serializable_float(value):
    if value is None:
        return None
    value = float(value)
    if np.isnan(value) or np.isinf(value):
        return None
    return value

def compute_pearson_stats(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid_mask = np.isfinite(x) & np.isfinite(y)
    sample_size = int(valid_mask.sum())

    if sample_size < 2:
        return {"r": None, "p": None, "n": sample_size}

    try:
        corr, p_value = pearsonr(x[valid_mask], y[valid_mask])
    except Exception:
        corr, p_value = np.nan, np.nan

    return {
        "r": _to_serializable_float(corr),
        "p": _to_serializable_float(p_value),
        "n": sample_size,
    }

def compute_linear_residuals(target, covariate):
    target = np.asarray(target, dtype=float)
    covariate = np.asarray(covariate, dtype=float)

    valid_mask = np.isfinite(target) & np.isfinite(covariate)
    residuals = np.full(target.shape, np.nan, dtype=float)
    if int(valid_mask.sum()) < 2:
        return residuals

    slope, intercept = np.polyfit(covariate[valid_mask], target[valid_mask], 1)
    fitted = slope * covariate[valid_mask] + intercept
    residuals[valid_mask] = target[valid_mask] - fitted
    return residuals

def plot_layer_depth_ablation(df, task, setting_name, ablation_stats, metric_col="importance_iqr_mean"):
    df_sorted = df.sort_values(by="layer_idx").copy()
    layer_idx = df_sorted["layer_idx"].astype(int).to_numpy()
    performance = df_sorted["performance"].astype(float).to_numpy()
    layer_log = np.log1p(df_sorted["layer_idx"].astype(float).to_numpy())

    metric_raw = df_sorted[metric_col].astype(float)
    metric_log = np.log(metric_raw.where(metric_raw > 0, np.nan)).to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    panels = [
        (
            axes[0],
            metric_log,
            performance,
            "Log(importance_iqr_mean)",
            "Performance",
            "Performance vs Log(IQR)",
            ablation_stats.get("performance_vs_log_importance_iqr_mean", {}),
        ),
        (
            axes[1],
            layer_log,
            performance,
            "Log1p(scoring_layer_idx)",
            "Performance",
            "Performance vs Log1p(Layer Index)",
            ablation_stats.get("performance_vs_log1p_layer_idx", {}),
        ),
    ]

    for ax, x, y, xlabel, ylabel, title_prefix, stats in panels:
        valid_mask = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[valid_mask], y[valid_mask], alpha=0.75, color="tab:blue")
        for idx, x_val, y_val in zip(layer_idx[valid_mask], x[valid_mask], y[valid_mask]):
            ax.annotate(int(idx), (x_val, y_val), fontsize=8, alpha=0.75)

        corr = stats.get("r")
        p_value = stats.get("p")
        if corr is not None and p_value is not None:
            title = f"{title_prefix}\nR={corr:.3f}, p={p_value:.3f}"
        else:
            title = title_prefix
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.6)

    joint_valid_mask = np.isfinite(metric_log) & np.isfinite(layer_log) & np.isfinite(performance)
    metric_residuals = compute_linear_residuals(metric_log[joint_valid_mask], layer_log[joint_valid_mask])
    performance_residuals = compute_linear_residuals(performance[joint_valid_mask], layer_log[joint_valid_mask])
    partial_stats = ablation_stats.get("partial_corr_performance_log_importance_iqr_mean_given_log1p_layer_idx", {})

    axes[2].scatter(metric_residuals, performance_residuals, alpha=0.75, color="tab:orange")
    for idx, x_val, y_val in zip(layer_idx[joint_valid_mask], metric_residuals, performance_residuals):
        if np.isfinite(x_val) and np.isfinite(y_val):
            axes[2].annotate(int(idx), (x_val, y_val), fontsize=8, alpha=0.75)

    partial_corr = partial_stats.get("r")
    partial_p = partial_stats.get("p")
    if partial_corr is not None and partial_p is not None:
        partial_title = f"Partial Corr After Controlling Log1p(Layer)\nR={partial_corr:.3f}, p={partial_p:.3f}"
    else:
        partial_title = "Partial Corr After Controlling Log1p(Layer)"
    axes[2].set_xlabel("Residual Log(importance_iqr_mean)")
    axes[2].set_ylabel("Residual Performance")
    axes[2].set_title(partial_title)
    axes[2].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_plot_path = f"/home/user/czx/open_source_proj/lmms-eval/outputs/src_output/{setting_name}/layer_depth_ablation_{task}.png"
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path)
    print(f"Layer-depth ablation plot saved to {output_plot_path}")
    plt.close(fig)

def build_layer_depth_ablation(df, task, setting_name, metric_col="importance_iqr_mean"):
    if metric_col not in df.columns:
        print(f"Warning: {metric_col} not found in dataframe for task {task}, skipping layer-depth ablation.")
        return None

    df_sorted = df.sort_values(by="layer_idx").copy()
    performance = df_sorted["performance"].astype(float).to_numpy()
    layer_log = np.log1p(df_sorted["layer_idx"].astype(float).to_numpy())
    metric_raw = df_sorted[metric_col].astype(float)
    metric_log = np.log(metric_raw.where(metric_raw > 0, np.nan)).to_numpy()

    ablation_stats = {
        "metric_name": metric_col,
        "metric_transform": "log",
        "layer_transform": "log1p",
        "performance_vs_log_importance_iqr_mean": compute_pearson_stats(metric_log, performance),
        "performance_vs_log1p_layer_idx": compute_pearson_stats(layer_log, performance),
        "log_importance_iqr_mean_vs_log1p_layer_idx": compute_pearson_stats(metric_log, layer_log),
    }

    joint_valid_mask = np.isfinite(metric_log) & np.isfinite(layer_log) & np.isfinite(performance)
    joint_sample_size = int(joint_valid_mask.sum())
    ablation_stats["joint_sample_size"] = joint_sample_size

    if joint_sample_size >= 3:
        metric_residuals = compute_linear_residuals(metric_log[joint_valid_mask], layer_log[joint_valid_mask])
        performance_residuals = compute_linear_residuals(performance[joint_valid_mask], layer_log[joint_valid_mask])
        layer_residuals = compute_linear_residuals(layer_log[joint_valid_mask], metric_log[joint_valid_mask])
        performance_residuals_from_metric = compute_linear_residuals(performance[joint_valid_mask], metric_log[joint_valid_mask])

        ablation_stats["partial_corr_performance_log_importance_iqr_mean_given_log1p_layer_idx"] = compute_pearson_stats(
            metric_residuals,
            performance_residuals,
        )
        ablation_stats["partial_corr_performance_log1p_layer_idx_given_log_importance_iqr_mean"] = compute_pearson_stats(
            layer_residuals,
            performance_residuals_from_metric,
        )
    else:
        ablation_stats["partial_corr_performance_log_importance_iqr_mean_given_log1p_layer_idx"] = {"r": None, "p": None, "n": joint_sample_size}
        ablation_stats["partial_corr_performance_log1p_layer_idx_given_log_importance_iqr_mean"] = {"r": None, "p": None, "n": joint_sample_size}

    plot_layer_depth_ablation(df_sorted, task, setting_name, ablation_stats, metric_col=metric_col)
    return ablation_stats

def main():
    # 设定你要分析的层索引列表
    # 例如：layer_indices = [0, 1, 2, ..., 31]
    # 请根据你的实际生成数据情况修改这里。
    # 如果你目前只有 layer 0 的数据，相关性计算需要至少两个点才有意义。
    # 这里为了演示，我先只填入 0，你需要添加更多层的索引。
    layer_indices = list(range(32))  # 假设有32层，从0到31
    
    enable_log = True  # 是否对统计指标取 Log 后计算相关性

    tasks = ["gqa", "textvqa_val", "pope", "mme", "mmbench_en_dev"]
    # tasks = ["gqa"]
    
    for task in tasks:
        print(f"\n{'='*30}\nProcessing Task: {task}\n{'='*30}")
        
        json_output = {
            "task": task,
            "correlations": {},
            "performance": {},
            "iqr_mean": {},
            "ablation": {}
        }
    
        # 1. 加载 wo_rss 数据 (Primary Data for Correlation)
        data_list_wo_rss = []
        base_path_wo_rss = f"/home/user/czx/open_source_proj/lmms-eval/outputs/logs/llava_wo_rss_dynamic_{task}"
        print(f"Loading WO RSS data from: {base_path_wo_rss}")
        for idx in layer_indices:
            data = load_data(idx, task=task, base_path=base_path_wo_rss)
            if data:
                data_list_wo_rss.append(data)
        
        # 2. 加载 dynamic 数据 (Comparison Data)
        data_list_dynamic = []
        base_path_dynamic = f"/home/user/czx/open_source_proj/lmms-eval/outputs/logs/llava_dynamic_{task}"
        print(f"Loading Dynamic data from: {base_path_dynamic}")
        for idx in layer_indices:
            data = load_data(idx, task=task, base_path=base_path_dynamic)
            if data:
                data_list_dynamic.append(data)

        # 3. 加载 raw_attn 数据
        data_list_raw_attn = []
        base_path_raw_attn = f"/home/user/czx/open_source_proj/lmms-eval/outputs/logs/llava_raw_attn_dynamic_{task}"
        print(f"Loading Raw Attn data from: {base_path_raw_attn}")
        for idx in layer_indices:
            data = load_data(idx, task=task, base_path=base_path_raw_attn)
            if data:
                data_list_raw_attn.append(data)

        # 4. 加载 ffn 数据
        data_list_ffn = []
        base_path_ffn = f"/home/user/czx/open_source_proj/lmms-eval/outputs/logs/llava_ffn_dynamic_{task}"
        print(f"Loading FFN data from: {base_path_ffn}")
        for idx in layer_indices:
            data = load_data(idx, task=task, base_path=base_path_ffn)
            if data:
                data_list_ffn.append(data)

        # 5. 加载 query attn 数据
        data_list_query_attn = []
        base_path_query_attn = f"/home/user/czx/open_source_proj/lmms-eval/outputs/logs/llava_query_attn_dynamic_{task}"
        print(f"Loading Query Attn data from: {base_path_query_attn}")
        for idx in layer_indices:
            data = load_data(idx, task=task, base_path=base_path_query_attn)
            if data:
                data_list_query_attn.append(data)
        
        # 6. 加载 query attn w rss 数据
        data_list_query_attn_w_rss = []
        base_path_query_attn_w_rss = f"/home/user/czx/open_source_proj/lmms-eval/outputs/logs/llava_query_attn_w_rss_dynamic_{task}"
        print(f"Loading Query Attn w RSS data from: {base_path_query_attn_w_rss}")
        for idx in layer_indices:
            data = load_data(idx, task=task, base_path=base_path_query_attn_w_rss)
            if data:
                data_list_query_attn_w_rss.append(data)

        # if len(data_list_wo_rss) < 2:
        #     print(f"Error: Need at least 2 data points (layers) in wo_rss to calculate correlation for task {task}.")
        #     if len(data_list_wo_rss) == 1:
        #         print("Current data:", data_list_wo_rss[0])
        #     continue

        df = pd.DataFrame(data_list_wo_rss)
        print(f"WO RSS Data extracted for {task}:")
        print(df)

        metric_suffixes = [
            "var_mean", "cv_mean", "iqr_mean", 
            "gini_mean", "hoyer_mean", "kurtosis_mean", 
            "entropy_mean", "spatial_entropy_mean"
        ]
        
        prefixes = ["importance", "raw_attn"]
        
        # # 相关性：wo_rss
        # plot_correlations(df, "wo_rss", task, prefixes, metric_suffixes, enable_log=enable_log)

        # 相关性：query_attn
        if len(data_list_query_attn) >= 2:
            df_query_attn = pd.DataFrame(data_list_query_attn)
            print(f"Query Attn Data extracted for {task}:")
            print(df_query_attn)
            
            # 定义额外的指标用于绘图
            additional_metrics = [
                ("attention_quality_mean", "Attention Quality", "purple", "d"),
                ("top_ratio_attention_quality_mean", "Top Ratio Attention Quality", "orange", "o"),
                ("top_ratio_importance_quality_mean", "Top Ratio Importance", "teal", "s")
            ]
            
            # 使用 plot_correlations 统一绘制相关性图
            stats_query_attn = plot_correlations(df_query_attn, "query_attn", task, prefixes, metric_suffixes, enable_log=enable_log, additional_metrics=additional_metrics)
            json_output["correlations"]["query_attn"] = stats_query_attn
            layer_depth_ablation = build_layer_depth_ablation(df_query_attn, task, "query_attn", metric_col="importance_iqr_mean")
            if layer_depth_ablation is not None:
                json_output["ablation"]["query_attn"] = layer_depth_ablation

        else:
            print(f"Warning: Not enough query_attn data for correlation plotting in task {task}.")

       # --- 性能对比折线图 (wo RSS vs w RSS vs Raw Attn vs FFN vs Query Attn vs Query Attn w RSS) ---
        plt.figure(figsize=(12, 7))

        # WO RSS
        if data_list_wo_rss:
            df_sorted = df.sort_values(by="layer_idx")
            # json_output["performance"]["wo_rss"] = df_sorted[["layer_idx", "performance"]].to_dict(orient="records")
            # if "importance_iqr_mean" in df_sorted.columns:
            #     json_output["iqr_mean"]["wo_rss"] = df_sorted[["layer_idx", "importance_iqr_mean"]].to_dict(orient="records")
            plt.plot(df_sorted["layer_idx"], df_sorted["performance"], marker='o', linestyle='-', color='b', label=f'WO RSS ({task})', alpha=0.8)
            
            # 标注 WO RSS 最高点
            if not df_sorted.empty:
                max_point = df_sorted.loc[df_sorted["performance"].idxmax()]
                plt.annotate(f"L{int(max_point['layer_idx'])}:{max_point['performance']:.2f}", 
                            (max_point['layer_idx'], max_point['performance']), 
                            textcoords="offset points", xytext=(0, 10), ha='center', color='b', fontweight='bold')

        # w RSS
        if data_list_dynamic:
            df_dynamic = pd.DataFrame(data_list_dynamic)
            print(f"Dynamic Data extracted for {task}:")
            print(df_dynamic)
            df_dynamic_sorted = df_dynamic.sort_values(by="layer_idx")
            # json_output["performance"]["dynamic"] = df_dynamic_sorted[["layer_idx", "performance"]].to_dict(orient="records")
            # if "importance_iqr_mean" in df_dynamic_sorted.columns:
            #     json_output["iqr_mean"]["dynamic"] = df_dynamic_sorted[["layer_idx", "importance_iqr_mean"]].to_dict(orient="records")
            plt.plot(df_dynamic_sorted["layer_idx"], df_dynamic_sorted["performance"], marker='s', linestyle='--', color='r', label=f'w RSS ({task})', alpha=0.8)
            
            # 标注 Dynamic 最高点
            if not df_dynamic_sorted.empty:
                max_point_dyn = df_dynamic_sorted.loc[df_dynamic_sorted["performance"].idxmax()]
                plt.annotate(f"L{int(max_point_dyn['layer_idx'])}:{max_point_dyn['performance']:.2f}",
                             (max_point_dyn['layer_idx'], max_point_dyn['performance']),
                             textcoords="offset points", xytext=(0, 10), ha='center', color='r', fontweight='bold')
        else:
            print(f"Warning: No data found for llava_dynamic_{task}, skipping dynamic in comparison plot.")

        # Raw Attn
        if data_list_raw_attn:
            df_raw_sorted = pd.DataFrame(data_list_raw_attn).sort_values(by="layer_idx")
            # json_output["performance"]["raw_attn"] = df_raw_sorted[["layer_idx", "performance"]].to_dict(orient="records")
            # if "importance_iqr_mean" in df_raw_sorted.columns:
            #     json_output["iqr_mean"]["raw_attn"] = df_raw_sorted[["layer_idx", "importance_iqr_mean"]].to_dict(orient="records")
            plt.plot(df_raw_sorted["layer_idx"], df_raw_sorted["performance"], marker='^', linestyle='-.', color='g', label=f'Raw Attn ({task})', alpha=0.8)
            
            # 标注 Raw Attn 最高点
            if not df_raw_sorted.empty:
                max_point_raw = df_raw_sorted.loc[df_raw_sorted["performance"].idxmax()]
                plt.annotate(f"L{int(max_point_raw['layer_idx'])}:{max_point_raw['performance']:.2f}",
                             (max_point_raw['layer_idx'], max_point_raw['performance']),
                             textcoords="offset points", xytext=(0, 10), ha='center', color='g', fontweight='bold')
        else:
            print(f"Warning: No data found for llava_raw_attn_dynamic_{task}, skipping raw_attn in comparison plot.")
        # FFN
        if data_list_ffn:
            df_ffn_sorted = pd.DataFrame(data_list_ffn).sort_values(by="layer_idx")
            # json_output["performance"]["ffn"] = df_ffn_sorted[["layer_idx", "performance"]].to_dict(orient="records")
            # if "importance_iqr_mean" in df_ffn_sorted.columns:
            #     json_output["iqr_mean"]["ffn"] = df_ffn_sorted[["layer_idx", "importance_iqr_mean"]].to_dict(orient="records")
            plt.plot(df_ffn_sorted["layer_idx"], df_ffn_sorted["performance"], marker='x', linestyle=':', color='m', label=f'FFN ({task})', alpha=0.8)
            
            # 标注 FFN 最高点
            if not df_ffn_sorted.empty:
                max_point_ffn = df_ffn_sorted.loc[df_ffn_sorted["performance"].idxmax()]
                plt.annotate(f"L{int(max_point_ffn['layer_idx'])}:{max_point_ffn['performance']:.2f}",
                             (max_point_ffn['layer_idx'], max_point_ffn['performance']),
                             textcoords="offset points", xytext=(0, 10), ha='center', color='m', fontweight='bold')
        else:
            print(f"Warning: No data found for llava_ffn_dynamic_{task}, skipping ffn in comparison plot.")
        
        # Query Attn
        if data_list_query_attn:
            df_query_sorted = pd.DataFrame(data_list_query_attn).sort_values(by="layer_idx")
            json_output["performance"]["query_attn"] = df_query_sorted[["layer_idx", "performance"]].to_dict(orient="records")
            if "importance_iqr_mean" in df_query_sorted.columns:
                json_output["iqr_mean"]["query_attn"] = df_query_sorted[["layer_idx", "importance_iqr_mean"]].to_dict(orient="records")
            plt.plot(df_query_sorted["layer_idx"], df_query_sorted["performance"], marker='d', linestyle='-', color='c', label=f'Query Attn ({task})', alpha=0.8)
            
            # 标注 Query Attn 最高点
            if not df_query_sorted.empty:
                max_point_query = df_query_sorted.loc[df_query_sorted["performance"].idxmax()]
                plt.annotate(f"L{int(max_point_query['layer_idx'])}:{max_point_query['performance']:.2f}",
                             (max_point_query['layer_idx'], max_point_query['performance']),
                             textcoords="offset points", xytext=(0, 10), ha='center', color='c', fontweight='bold')
        else:
            print(f"Warning: No data found for llava_query_attn_dynamic_{task}, skipping query_attn in comparison plot.")

        # Query Attn w RSS
        if data_list_query_attn_w_rss:
            df_query_sorted = pd.DataFrame(data_list_query_attn_w_rss).sort_values(by="layer_idx")
            # json_output["performance"]["query_attn_w_rss"] = df_query_sorted[["layer_idx", "performance"]].to_dict(orient="records")
            # if "importance_iqr_mean" in df_query_sorted.columns:
            #     json_output["iqr_mean"]["query_attn_w_rss"] = df_query_sorted[["layer_idx", "importance_iqr_mean"]].to_dict(orient="records")
            plt.plot(df_query_sorted["layer_idx"], df_query_sorted["performance"], marker='d', linestyle='-', color='m', label=f'Query Attn w RSS ({task})', alpha=0.8)
            
            # 标注 Query Attn w RSS 最高点
            if not df_query_sorted.empty:
                max_point_query = df_query_sorted.loc[df_query_sorted["performance"].idxmax()]
                plt.annotate(f"L{int(max_point_query['layer_idx'])}:{max_point_query['performance']:.2f}",
                             (max_point_query['layer_idx'], max_point_query['performance']),
                             textcoords="offset points", xytext=(0, 10), ha='center', color='m', fontweight='bold')
        else:
            print(f"Warning: No data found for llava_query_attn_dynamic_{task}, skipping query_attn in comparison plot.")

        plt.xlabel("Scoring Layer Index")
        plt.ylabel("Performance")
        plt.title(f"Model Performance Comparison: WO RSS vs Dynamic vs Raw Attn vs FFN vs Query Attn vs Query Attn w RSS ({task})")
        plt.grid(True)
        plt.legend()

        output_perf_path = f"/home/user/czx/open_source_proj/lmms-eval/outputs/src_output/query_attn/performance_vs_layer_comparison_{task}.png"
        os.makedirs(os.path.dirname(output_perf_path), exist_ok=True)
        plt.savefig(output_perf_path)
        print(f"Performance vs Layer comparison plot saved to {output_perf_path}")
        plt.close()

        # Save Analysis JSON
        json_output_path = f"/home/user/czx/open_source_proj/lmms-eval/outputs/src_output/query_attn/analysis_stats_{task}.json"
        with open(json_output_path, 'w') as f:
            json.dump(json_output, f, indent=4)
        print(f"Analysis stats saved to {json_output_path}")


if __name__ == "__main__":
    main()
