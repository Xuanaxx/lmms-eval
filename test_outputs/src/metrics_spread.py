import json
import os
import matplotlib.pyplot as plt
import numpy as np
from math import ceil

# 需要可视化的统计指标类型列表
METRICS_TYPES = [
    "var", "cv", "iqr", 
    "gini", "hoyer", "kurtosis", 
    "entropy", "spatial_entropy"
]

# 新增的质量指标
QUALITY_METRICS = [
    "attention_quality",
    "top_ratio_attention_quality",
    "top_ratio_importance_quality"
]

def load_data(layer_idx, base_path):
    """
    加载指定层的统计数据。
    提取 importance 和 raw_attn 下各指标的均值(mean)和方差(var)。
    """
    layer_dir = os.path.join(base_path, f"scoring_layer_{layer_idx}")
    metrics_path = os.path.join(layer_dir, f"metrics_layer_{layer_idx}.json")
    
    if not os.path.exists(metrics_path):
        # 仅在找不到文件时输出提示，避免日志刷屏
        # print(f"Info: Metrics file not found for layer {layer_idx}: {metrics_path}")
        return None

    try:
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
    except Exception as e:
        print(f"Error loading {metrics_path}: {e}")
        return None
    
    stats_key = f"layer_{layer_idx}_stats"
    stats = metrics_data.get(stats_key, {})
    
    # 构建数据结构:
    # {
    #   "layer_idx": idx,
    #   "importance": { "var": {"mean": x, "var": y, "std": z}, ... },
    #   "raw_attn":   { "var": {"mean": x, "var": y, "std": z}, ... },
    #   "quality":    { "attention_quality": {"mean": x, ...}, ... }
    # }
    
    data = {"layer_idx": layer_idx, "importance": {}, "raw_attn": {}, "quality": {}}
    
    for m in METRICS_TYPES:
        for prefix in ["importance", "raw_attn"]:
            mean_key = f"{prefix}_{m}_mean"
            var_key = f"{prefix}_{m}_var"
            
            # 如果json中没有对应key，默认为NaN
            val_mean = stats.get(mean_key, np.nan)
            val_var = stats.get(var_key, np.nan)
            
            # 计算标准差用于绘图 (Std = sqrt(Var))
            if val_var is not None and val_var >= 0:
                val_std = np.sqrt(val_var)
            else:
                val_std = 0
            
            data[prefix][m] = {
                "mean": val_mean,
                "var": val_var,
                "std": val_std
            }

    # 加载 Quality Metrics
    for qm in QUALITY_METRICS:
        mean_key = f"{qm}_mean"
        var_key = f"{qm}_var"
        
        val_mean = stats.get(mean_key, np.nan)
        val_var = stats.get(var_key, np.nan)
        
        if val_var is not None and val_var >= 0:
            val_std = np.sqrt(val_var)
        else:
            val_std = 0
            
        data["quality"][qm] = {
            "mean": val_mean,
            "var": val_var,
            "std": val_std
        }
            
    return data

def plot_metrics_trends(all_data, output_dir, task_name):
    """
    绘制标准指标随层数变化的趋势图（包含均值线和方差范围）。
    仅绘制 METRICS_TYPES。
    """
    # 按层索引排序
    all_data.sort(key=lambda x: x['layer_idx'])
    
    layers = [d['layer_idx'] for d in all_data]
    
    # 仅使用普通指标
    n_metrics = len(METRICS_TYPES)
    cols = 4  # 每行4张图
    rows = ceil(n_metrics / cols)
    
    # 调整figsize以适应所有子图
    fig, axes = plt.subplots(rows, cols, figsize=(24, 6 * rows))
    axes = axes.flatten()
    
    # 1. 绘制 Comparative Metrics (Importance vs Raw Attn)
    for i, metric in enumerate(METRICS_TYPES):
        ax = axes[i]
        
        # 准备绘图数据
        imp_means = []
        imp_stds = []
        raw_means = []
        raw_stds = []
        
        for d in all_data:
            imp_stats = d["importance"][metric]
            raw_stats = d["raw_attn"][metric]
            
            imp_means.append(imp_stats["mean"])
            imp_stds.append(imp_stats["std"])
            raw_means.append(raw_stats["mean"])
            raw_stds.append(raw_stats["std"])
            
        imp_means = np.array(imp_means)
        imp_stds = np.array(imp_stds)
        raw_means = np.array(raw_means)
        raw_stds = np.array(raw_stds)
        
        # 过滤无效数据以便绘图
        valid_idx_imp = ~np.isnan(imp_means)
        valid_idx_raw = ~np.isnan(raw_means)
        
        plot_x = np.array(layers)

        # 绘制 Importance 数据
        if np.any(valid_idx_imp):
            x = plot_x[valid_idx_imp]
            y = imp_means[valid_idx_imp]
            err = imp_stds[valid_idx_imp]
            
            # 绘制均值线
            ax.plot(x, y, label='Importance', color='blue', marker='o', markersize=4, linewidth=2)
            # 绘制方差范围 (Mean +/- Std)
            ax.fill_between(x, y - err, y + err, color='blue', alpha=0.2, label='Imp StdDev')
            
        # 绘制 Raw Attn 数据
        if np.any(valid_idx_raw):
            x = plot_x[valid_idx_raw]
            y = raw_means[valid_idx_raw]
            err = raw_stds[valid_idx_raw]
            
            # 使用虚线区分
            ax.plot(x, y, label='Raw Attn', color='red', marker='s', markersize=4, linestyle='--', linewidth=2)
            ax.fill_between(x, y - err, y + err, color='red', alpha=0.2, label='Raw StdDev')
            
        ax.set_title(f"Metric Trend: {metric}", fontsize=14)
        ax.set_xlabel("Layer Index", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # 只在第一个子图或每个子图中显示图例
        ax.legend(loc='best', fontsize=10)
    
    # 隐藏多余的子图坐标轴
    for k in range(n_metrics, len(axes)):
        fig.delaxes(axes[k])
        
    plt.tight_layout()
    
    # 保存图片
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(output_dir, f"metrics_trends_variance_{task_name}.png")
    plt.savefig(save_path)
    print(f"Comparison plot saved to: {save_path}")
    plt.close()

def plot_quality_comparison(multi_task_data, output_dir):
    """
    绘制 Quality Metrics 跨数据集对比图。
    将所有 Task 的 Quality Metrics 绘制在同一张图上。
    """
    tasks = list(multi_task_data.keys())
    n_metrics = len(QUALITY_METRICS)
    
    # 设置画布
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    colors = plt.cm.tab10(np.linspace(0, 1, len(tasks)))
    
    for i, metric in enumerate(QUALITY_METRICS):
        ax = axes[i]
        
        for j, task in enumerate(tasks):
            task_data = multi_task_data[task]
            # Ensure sorted
            task_data.sort(key=lambda x: x['layer_idx'])
            
            layers = []
            means = []
            
            for d in task_data:
                layers.append(d['layer_idx'])
                stats = d["quality"].get(metric, {})
                means.append(stats.get("mean", np.nan))
            
            layers = np.array(layers)
            means = np.array(means)
            
            valid_idx = ~np.isnan(means)
            if np.any(valid_idx):
                ax.plot(layers[valid_idx], means[valid_idx], label=task, color=colors[j], marker='o', markersize=3, alpha=0.8)
        
        ax.set_title(f"{metric}", fontsize=12)
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Mean Value")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(fontsize=8)
        
    plt.tight_layout()
    save_path = os.path.join(output_dir, "quality_metrics_cross_dataset_comparison.png")
    plt.savefig(save_path)
    print(f"Quality metrics comparison plot saved to: {save_path}")
    plt.close()

def plot_importance_comparison(multi_task_data, output_dir):
    """
    绘制 Importance Metrics 跨数据集对比图。
    将所有 Task 的 Importance Metrics 绘制在同一张图上。
    """
    tasks = list(multi_task_data.keys())
    n_metrics = len(METRICS_TYPES)

    cols = 4
    rows = ceil(n_metrics / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(24, 6 * rows))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, len(tasks)))

    for i, metric in enumerate(METRICS_TYPES):
        ax = axes[i]
        for j, task in enumerate(tasks):
            task_data = multi_task_data[task]
            task_data.sort(key=lambda x: x['layer_idx'])

            layers = []
            means = []
            for d in task_data:
                layers.append(d['layer_idx'])
                stats = d["importance"].get(metric, {})
                means.append(stats.get("mean", np.nan))

            layers = np.array(layers)
            means = np.array(means)

            valid_idx = ~np.isnan(means)
            if np.any(valid_idx):
                ax.plot(layers[valid_idx], means[valid_idx], label=task, color=colors[j], marker='o', markersize=3, alpha=0.8)

        ax.set_title(f"Importance: {metric}", fontsize=12)
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Mean Value")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(fontsize=8)

    for k in range(n_metrics, len(axes)):
        fig.delaxes(axes[k])

    plt.tight_layout()
    save_path = os.path.join(output_dir, "importance_metrics_cross_dataset_comparison.png")
    plt.savefig(save_path)
    print(f"Importance metrics comparison plot saved to: {save_path}")
    plt.close()

def main():
    # 配置参数
    layer_indices = list(range(40))  
    
    tasks = ["gqa", "textvqa_val", "pope", "mme", "mmbench_en_dev"]
    
    output_dir = "/home/user/czx/open_source_proj/lmms-eval/outputs/src_output/qwenvl_metrics_trends_query_attn"
    
    multi_task_data = {}
    
    for task_name in tasks:
        # 路径根据 workspace 信息进行调整
        base_log_path = f"/home/user/czx/open_source_proj/lmms-eval/outputs/logs/qwenvl_query_attn_wo_rss_dynamic_{task_name}"
        print(f"Start collecting metrics for {task_name} from: {base_log_path}")
        
        all_data = []
        found_count = 0
        
        for idx in layer_indices:
            data = load_data(idx, base_log_path)
            if data:
                all_data.append(data)
                found_count += 1
        
        if found_count == 0:
            print(f"Warning: No metrics files found for {task_name} in {base_log_path}.")
            continue
        
        # 存储该任务的数据
        multi_task_data[task_name] = all_data

        print(f"Successfully loaded data for {found_count} layers for {task_name}.")
        
        # 绘图 (标准指标)
        plot_metrics_trends(all_data, output_dir, task_name)

    # 绘制 Quality Metrics 跨数据集对比
    if multi_task_data:
        plot_quality_comparison(multi_task_data, output_dir)
        plot_importance_comparison(multi_task_data, output_dir)
    else:
        print("No data collected for any tasks.")

if __name__ == "__main__":
    main()