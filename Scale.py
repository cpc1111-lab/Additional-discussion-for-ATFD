import os
import json
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pycocotools.coco import COCO

# -----------------------------
# 配置
# -----------------------------
# COCO 格式标注文件路径（需替换为实际路径）
ANNOTATION_FILE = '/home/e222/cpc/D-FINE/dataset/annotations/instances_train_int_id_fixed.json'
CATEGORIES = ['airplane', 'car', 'person']  # 注意：需与COCO中category name一致
OUTPUT_DIR = 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化COCO api
coco = COCO(ANNOTATION_FILE)

# 获取类别ID
cat_ids = coco.getCatIds(catNms=CATEGORIES)
cats = coco.loadCats(cat_ids)
cat_id_to_name = {cat['id']: cat['name'] for cat in cats}

# 存储每类的尺度
scale_data = {name: [] for name in CATEGORIES}

# 遍历所有标注，提取尺度
for ann in coco.anns.values():
    cat_id = ann['category_id']
    if cat_id not in cat_id_to_name:
        continue
    bbox = ann['bbox']  # [x, y, w, h]
    w, h = bbox[2], bbox[3]
    if w <= 0 or h <= 0:
        continue
    s = np.sqrt(w * h)
    cat_name = cat_id_to_name[cat_id]
    scale_data[cat_name].append(s)

# 转为numpy数组
for k in scale_data:
    scale_data[k] = np.array(scale_data[k])

# -----------------------------
# 绘图：改为 2 行 3 列，同类别的全局 + 局部在同一列
# -----------------------------
plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # 2 行（视图） × 3 列（类别）
fig.suptitle('Scale Distribution in Airport Scenes (AFTD)', fontsize=16)

colors = {'airplane': 'tab:blue', 'car': 'tab:orange', 'person': 'tab:green'}

# 遍历每个类别，分配到对应列
for col_idx, cat in enumerate(CATEGORIES):
    ax_global = axes[0, col_idx]  # 第0行：全局图
    ax_zoom   = axes[1, col_idx]  # 第1行：局部放大图

    data = scale_data[cat]
    if len(data) == 0:
        ax_global.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_global.transAxes)
        ax_zoom.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_zoom.transAxes)
        continue

    # 全局图
    sns.histplot(data, bins=120, stat='density', ax=ax_global,
                 color=colors[cat], alpha=1.0, linewidth=0)
    sns.kdeplot(data, ax=ax_global, color='black', linewidth=2, linestyle='--')
    ax_global.set_title(f'{cat.capitalize()}', fontsize=14)
    ax_global.set_xlabel('')
    ax_global.set_ylabel('Density')

    # 局部放大
    from scipy.stats import gaussian_kde
    from scipy.signal import find_peaks

    kde_full = gaussian_kde(data)
    x_grid = np.linspace(data.min(), data.max(), 1000)
    density = kde_full(x_grid)
    peaks, _ = find_peaks(density, distance=50, prominence=np.max(density) * 0.1)

    if len(peaks) >= 2:
        p1, p2 = x_grid[peaks[0]], x_grid[peaks[1]]
        zoom_min, zoom_max = min(p1, p2) - 5, max(p1, p2) + 5
    else:
        zoom_min, zoom_max = 0, np.percentile(data, 20)

    zoom_data = data[(data >= zoom_min) & (data <= zoom_max)]
    if len(zoom_data) == 0:
        zoom_min, zoom_max = data.min(), np.percentile(data, 10)
        zoom_data = data[(data >= zoom_min) & (data <= zoom_max)]

    sns.histplot(zoom_data, bins=60, stat='density', ax=ax_zoom,
                 color=colors[cat], alpha=1.0, linewidth=0)
    sns.kdeplot(zoom_data, ax=ax_zoom, color='black', linewidth=2, linestyle='--')
    ax_zoom.set_xlabel('Geometric Scale $s = \\sqrt{w \\cdot h}$')
    ax_zoom.set_ylabel('Density')
    # 局部图标题可选，也可以省略以节省空间
    # ax_zoom.set_title(f'Zoomed', fontsize=12)


# 统一设置 x 轴标签（只在底部行显示）
for ax in axes[1, :]:
    ax.set_xlabel('Geometric Scale $s = \\sqrt{w \\cdot h}$')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, 'scale_distribution_multimodal.png'), dpi=300, bbox_inches='tight')
plt.show()