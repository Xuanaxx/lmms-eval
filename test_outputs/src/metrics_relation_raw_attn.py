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

def plot_correlations(df, setting_name, task, prefixes, metric_suffixes, enable_log=False):
    """通用相关性绘图函数"""
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

def main():
    # 设定你要分析的层索引列表
    # 例如：layer_indices = [0, 1, 2, ..., 31]
    # 请根据你的实际生成数据情况修改这里。
    # 如果你目前只有 layer 0 的数据，相关性计算需要至少两个点才有意义。
    # 这里为了演示，我先只填入 0，你需要添加更多层的索引。
    layer_indices = list(range(32))  # 假设有32层，从0到31
    
    enable_log = True  # 是否对统计指标取 Log 后计算相关性

    # tasks = ["gqa", "textvqa_val", "pope", "mme", "mmbench_en_dev"]
    tasks = ["mmbench_en_dev"]
    for task in tasks:
        print(f"\n{'='*30}\nProcessing Task: {task}\n{'='*30}")
    
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

        # 相关性：raw_attn
        if len(data_list_raw_attn) >= 2:
            df_raw = pd.DataFrame(data_list_raw_attn)
            print(f"Raw Attn Data extracted for {task}:")
            print(df_raw)
            plot_correlations(df_raw, "raw_attn", task, prefixes, metric_suffixes, enable_log=enable_log)

            # --- 新增：Additional Metrics Analysis (Attention Quality & Top Ratios) ---
            additional_metrics = [
                ("attention_quality_mean", "Attention Quality", "purple", "d"),
                ("top_ratio_attention_quality_mean", "Top Ratio Attention Quality", "orange", "o"),
                ("top_ratio_importance_quality_mean", "Top Ratio Importance Quality", "teal", "s")
            ]

            for metric_col, title_prefix, color, marker in additional_metrics:
                if metric_col in df_raw.columns:
                    
                    safe_metric_name = metric_col.replace("_mean", "")

                    # 绘制相关性图
                    plt.figure(figsize=(8, 6))
                    x = df_raw[metric_col].astype(float)
                    y = df_raw["performance"].astype(float)
                    
                    mask = ~np.isnan(x) & ~np.isnan(y)
                    x_clean = x[mask]
                    y_clean = y[mask]
                    
                    if len(x_clean) > 1:
                        corr, p_value = pearsonr(x_clean, y_clean)
                        plt.scatter(x_clean, y_clean, alpha=0.7, color=color)
                        plt.title(f"{title_prefix} vs Performance\nR={corr:.3f}, p={p_value:.3f}")
                        plt.xlabel(f"{title_prefix} (Mean)")
                        plt.ylabel("Performance Score")
                        plt.grid(True)
                        
                        # Annotate using row iteration
                        for _, row in df_raw.iterrows():
                             xv = row[metric_col]
                             yv = row["performance"]
                             if not np.isnan(xv) and not np.isnan(yv):
                                plt.annotate(int(row["layer_idx"]), (xv, yv))

                        out_corr_path = f"/home/user/czx/open_source_proj/lmms-eval/outputs/src_output/raw_attn/correlation_{safe_metric_name}_{task}.png"
                        os.makedirs(os.path.dirname(out_corr_path), exist_ok=True)
                        plt.savefig(out_corr_path)
                        print(f"Correlation plot for {title_prefix} saved to {out_corr_path}")
                    plt.close()

        else:
            print(f"Warning: Not enough raw_attn data for correlation plotting in task {task}.")

       # --- 性能对比折线图 (WO RSS vs Dynamic vs Raw Attn) ---
        plt.figure(figsize=(12, 7))

        # WO RSS
        df_sorted = df.sort_values(by="layer_idx")
        plt.plot(df_sorted["layer_idx"], df_sorted["performance"], marker='o', linestyle='-', color='b', label=f'WO RSS ({task})', alpha=0.8)
        
        # 标注 WO RSS 最高点
        if not df_sorted.empty:
            max_point = df_sorted.loc[df_sorted["performance"].idxmax()]
            plt.annotate(f"L{int(max_point['layer_idx'])}:{max_point['performance']:.2f}", 
                         (max_point['layer_idx'], max_point['performance']), 
                         textcoords="offset points", xytext=(0, 10), ha='center', color='b', fontweight='bold')

        # Dynamic
        if data_list_dynamic:
            df_dynamic = pd.DataFrame(data_list_dynamic)
            print(f"Dynamic Data extracted for {task}:")
            print(df_dynamic)
            df_dynamic_sorted = df_dynamic.sort_values(by="layer_idx")
            plt.plot(df_dynamic_sorted["layer_idx"], df_dynamic_sorted["performance"], marker='s', linestyle='--', color='r', label=f'Dynamic ({task})', alpha=0.8)
            
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
            plt.plot(df_raw_sorted["layer_idx"], df_raw_sorted["performance"], marker='^', linestyle='-.', color='g', label=f'Raw Attn ({task})', alpha=0.8)
            
            # 标注 Raw Attn 最高点
            if not df_raw_sorted.empty:
                max_point_raw = df_raw_sorted.loc[df_raw_sorted["performance"].idxmax()]
                plt.annotate(f"L{int(max_point_raw['layer_idx'])}:{max_point_raw['performance']:.2f}",
                             (max_point_raw['layer_idx'], max_point_raw['performance']),
                             textcoords="offset points", xytext=(0, 10), ha='center', color='g', fontweight='bold')
        else:
            print(f"Warning: No data found for llava_raw_attn_dynamic_{task}, skipping raw_attn in comparison plot.")

        plt.xlabel("Scoring Layer Index")
        plt.ylabel("Performance")
        plt.title(f"Model Performance Comparison: WO RSS vs Dynamic vs Raw Attn ({task})")
        plt.grid(True)
        plt.legend()

        output_perf_path = f"/home/user/czx/open_source_proj/lmms-eval/outputs/src_output/raw_attn/performance_vs_layer_comparison_{task}.png"
        os.makedirs(os.path.dirname(output_perf_path), exist_ok=True)
        plt.savefig(output_perf_path)
        print(f"Performance vs Layer comparison plot saved to {output_perf_path}")
        plt.close()


if __name__ == "__main__":
    main()