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
        elif "mme_cognition_score,none" in res_dict:
            performance = res_dict["mme_cognition_score,none"]
            # 如果没有常用key，尝试获取第一个value
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

def main():
    # 设定你要分析的层索引列表
    # 例如：layer_indices = [0, 1, 2, ..., 31]
    # 请根据你的实际生成数据情况修改这里。
    # 如果你目前只有 layer 0 的数据，相关性计算需要至少两个点才有意义。
    # 这里为了演示，我先只填入 0，你需要添加更多层的索引。
    layer_indices = list(range(32))  # 假设有32层，从0到31
    # task = "gqa"  # 根据你的任务修改
    # task = "textvqa_val"  # 根据你的任务修改
    task = "pope"  # 根据你的任务修改
    # task = "mme"  # 根据你的任务修改
    data_list = []
    for idx in layer_indices:
        data = load_data(idx, task=task, base_path=f"/home/user/czx/open_source_proj/lmms-eval/outputs/logs/llava_dynamic_{task}")
        if data:
            data_list.append(data)
    
    if len(data_list) < 2:
        print("Error: Need at least 2 data points (layers) to calculate correlation.")
        if len(data_list) == 1:
            print("Current data:", data_list[0])
        return

    df = pd.DataFrame(data_list)
    print("Data extracted:")
    print(df)

    metric_suffixes = [
        "var_mean", "cv_mean", "iqr_mean", 
        "gini_mean", "hoyer_mean", "kurtosis_mean", 
        "entropy_mean", "spatial_entropy_mean"
    ]
    
    prefixes = ["importance", "raw_attn"]
    
    for prefix in prefixes:
        # 设置绘图
        rows = 2
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(24, 10))
        axes = axes.flatten()
        
        plot_exists = False
        for i, suffix in enumerate(metric_suffixes):
            if i >= len(axes): break
            
            metric = f"{prefix}_{suffix}"
            if metric not in df.columns:
                continue

            ax = axes[i]
            x = df[metric].astype(float)
            y = df["performance"].astype(float)
            
            # 过滤 NaN 值
            valid_mask = ~np.isnan(x) & ~np.isnan(y)
            x_clean = x[valid_mask]
            y_clean = y[valid_mask]
            
            if len(x_clean) < 1:
                ax.set_title(f"{metric} (No Data)")
                continue

            plot_exists = True

            # 计算相关性
            if len(x_clean) > 1:
                corr, p_value = pearsonr(x_clean, y_clean)
                title = f"{metric} vs Performance\nR={corr:.3f}, p={p_value:.3f}"
            else:
                title = f"{metric} vs Performance"

            ax.scatter(x, y, alpha=0.7)
            ax.set_xlabel(metric)
            ax.set_ylabel("Performance Score")
            ax.set_title(title)
            ax.grid(True)
            
            # 添加层索引标签
            for j, txt in enumerate(df["layer_idx"]):
                if j < len(x) and not np.isnan(x[j]) and not np.isnan(y[j]):
                    ax.annotate(txt, (x[j], y[j]))

        # 隐藏多余的子图
        for j in range(len(metric_suffixes), len(axes)):
            fig.delaxes(axes[j])

        if plot_exists:
            plt.tight_layout()
            output_plot_path = f"/home/user/czx/open_source_proj/lmms-eval/outputs/src_output/correlation_plots_{prefix}_{task}.png"
            plt.savefig(output_plot_path)
            print(f"Correlation plots for {prefix} saved to {output_plot_path}")
        plt.close(fig)

    # 绘制模型性能随 scoring_layer_idx 的变化折线图
    plt.figure(figsize=(10, 6))
    df_sorted = df.sort_values(by="layer_idx")
    plt.plot(df_sorted["layer_idx"], df_sorted["performance"], marker='o', linestyle='-', color='b', label='Exact Match')

    plt.xlabel("Scoring Layer Index")
    plt.ylabel("Performance (Exact Match)")
    plt.title(f"Model Performance vs Scoring Layer Index ({task})")
    plt.grid(True)
    plt.legend()

    # 在点上标注数值
    for x, y in zip(df_sorted["layer_idx"], df_sorted["performance"]):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=9)

    output_perf_path = f"/home/user/czx/open_source_proj/lmms-eval/outputs/src_output/performance_vs_layer_{task}.png"
    plt.savefig(output_perf_path)
    print(f"Performance vs Layer plot saved to {output_perf_path}")

if __name__ == "__main__":
    main()