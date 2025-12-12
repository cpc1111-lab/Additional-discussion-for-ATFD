import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import gaussian_kde
import os

def extract_and_visualize_scale_priors(ann_file, save_dir=None, top_k_classes=10, dpi=150):
    """
    从 COCO 格式标注中提取并可视化类别级尺度先验。

    参数:
        ann_file (str): COCO 标注文件路径
        save_dir (str): 保存统计量与可视化图的目录
        top_k_classes (int): 可视化实例最多的前 K 个类别（避免绘图过多）
        dpi (int): 图像分辨率
    """
    with open(ann_file, 'r') as f:
        coco = json.load(f)

    cat_id_to_name = {cat['id']: cat['name'] for cat in coco['categories']}
    cat_ids = sorted(cat_id_to_name.keys())

    class_scales = defaultdict(list)
    class_ar = defaultdict(list)

    for ann in coco['annotations']:
        if ann.get('iscrowd', 0) == 1:
            continue
        w, h = ann['bbox'][2], ann['bbox'][3]
        if w <= 0 or h <= 0:
            continue
        scale = np.sqrt(w * h)
        ar = w / h
        class_scales[ann['category_id']].append(scale)
        class_ar[ann['category_id']].append(ar)

    # 按实例数量排序，选 top-k 可视化
    class_counts = {cid: len(scales) for cid, scales in class_scales.items()}
    top_classes = sorted(class_counts.items(), key=lambda x: -x[1])[:top_k_classes]
    top_cat_ids = [cid for cid, _ in top_classes]

    priors = {}
    os.makedirs(save_dir, exist_ok=True)

    # 准备保存的统计量（不含 KDE）
    stats_to_save = {}

    for cat_id in cat_ids:
        scales = np.array(class_scales[cat_id])
        ars = np.array(class_ar[cat_id])

        if len(scales) == 0:
            median_scale = 0.0
            mean_ar = 1.0
            kde = None
        else:
            median_scale = np.median(scales)
            mean_ar = np.mean(ars)
            if len(scales) >= 2:
                try:
                    kde = gaussian_kde(scales, bw_method='scott')
                except np.linalg.LinAlgError:
                    kde = None
            else:
                kde = None

        priors[cat_id] = {
            'scales': scales,
            'ars': ars,
            'median_scale': float(median_scale),
            'mean_ar': float(mean_ar),
            'kde': kde,
            'category_name': cat_id_to_name.get(cat_id, f'cat_{cat_id}'),
            'count': len(scales)
        }

        # 保存统计量（用于后续加载）
        stats_to_save[cat_id] = {
            'category_name': priors[cat_id]['category_name'],
            'num_instances': int(priors[cat_id]['count']),
            'median_scale': priors[cat_id]['median_scale'],
            'mean_ar': priors[cat_id]['mean_ar']
        }

    # 保存 JSON
    with open(os.path.join(save_dir, 'scale_priors.json'), 'w') as f:
        json.dump(stats_to_save, f, indent=2)

    # ===== 可视化 =====
    for cat_id in top_cat_ids:
        info = priors[cat_id]
        if info['count'] == 0:
            continue

        fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=dpi)
        name = info['category_name']
        scales = info['scales']
        ars = info['ars']
        median_s = info['median_scale']
        mean_ar = info['mean_ar']

        # --- 左图：尺度分布 + KDE ---
        axs[0].hist(scales, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='k', linewidth=0.5, label='Empirical')
        if info['kde'] is not None:
            s_grid = np.linspace(scales.min(), scales.max(), 300)
            kde_vals = info['kde'](s_grid)
            axs[0].plot(s_grid, kde_vals, 'r-', lw=2, label='KDE')
        axs[0].axvline(median_s, color='g', linestyle='--', lw=2, label=f'Median = {median_s:.1f}')
        axs[0].set_xlabel('Geometric Scale $s = \\sqrt{w \\cdot h}$')
        axs[0].set_ylabel('Density')
        axs[0].set_title(f'Scale Distribution: {name} (N={info["count"]})')
        axs[0].legend()
        axs[0].grid(True, linestyle='--', alpha=0.5)

        # --- 右图：长宽比分布 ---
        axs[1].hist(ars, bins=50, color='lightcoral', alpha=0.7, edgecolor='k', linewidth=0.5)
        axs[1].axvline(mean_ar, color='b', linestyle='--', lw=2, label=f'Mean AR = {mean_ar:.2f}')
        axs[1].set_xlabel('Aspect Ratio $r = w / h$')
        axs[1].set_ylabel('Count')
        axs[1].set_title(f'Aspect Ratio Distribution: {name}')
        axs[1].legend()
        axs[1].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        vis_path = os.path.join(save_dir, f'scale_prior_vis_cat{cat_id}_{name}.png')
        plt.savefig(vis_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved visualization for category '{name}' (ID={cat_id}) to {vis_path}")

    print(f"\n✅ 共可视化 {len(top_cat_ids)} 个类别，结果保存至: {save_dir}")
    return priors


# -----------------------------
# 示例用法
# -----------------------------
if __name__ == "__main__":
    ann_file = "/home/e222/cpc/D-FINE/dataset/coco/annotations/instances_train2017.json"  # 替换为你的 COCO 标注路径
    save_dir = "scale_priors_output"

    # 提取并可视化
    priors = extract_and_visualize_scale_priors(
        ann_file,
        save_dir=save_dir,
        top_k_classes=80,
        dpi=150
    )

    # 示例：查看飞机类（COCO 中 cat_id=5）
    if 1 in priors and priors[1]['count'] > 0:
        info = priors[1]
        print(f"\n�� 飞机类别统计:")
        print(f"  - 实例数: {info['count']}")
        print(f"  - 中位尺度: {info['median_scale']:.2f}")
        print(f"  - 平均长宽比 (w/h): {info['mean_ar']:.2f}")
        if info['kde'] is not None:
            s_sample = info['kde'].resample(size=1).item()
            w = s_sample * np.sqrt(info['mean_ar'])
            h = s_sample / np.sqrt(info['mean_ar'])
            print(f"  - KDE 采样示例 (w, h): ({w:.1f}, {h:.1f})")