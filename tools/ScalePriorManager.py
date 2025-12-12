# scale_prior_manager.py
import json
import numpy as np
import torch
from sklearn.neighbors import KernelDensity


class ScalePriorManager:
    def __init__(self, use_scale_prior=False):
        """
        Manages nonparametric scale priors for COCO-style datasets.

        Args:
            use_scale_prior (bool): Global switch to enable/disable scale priors.
        """
        self.use_scale_prior = use_scale_prior
        self.priors = {}  # {category_id: { 'kde', 'median_scale', 'mean_ar' }}

    def fit_from_coco_json(self, coco_json_path):
        """
        Automatically fit nonparametric scale priors from COCO-format annotation file.

        Args:
            coco_json_path (str): Path to COCO .json annotation file (e.g., 'train.json').
        """
        if not self.use_scale_prior:
            return

        with open(coco_json_path, 'r') as f:
            coco = json.load(f)

        # Group boxes by category_id
        boxes_by_cat = {}
        for ann in coco['annotations']:
            if ann.get('iscrowd', 0) == 1:
                continue  # skip crowd annotations
            cat_id = ann['category_id']
            bbox = ann['bbox']  # [x, y, w, h] in COCO format
            w, h = bbox[2], bbox[3]
            if w <= 0 or h <= 0:
                continue
            if cat_id not in boxes_by_cat:
                boxes_by_cat[cat_id] = {'ws': [], 'hs': []}
            boxes_by_cat[cat_id]['ws'].append(w)
            boxes_by_cat[cat_id]['hs'].append(h)

        # Fit priors per category
        for cat_id, data in boxes_by_cat.items():
            ws = np.array(data['ws'])
            hs = np.array(data['hs'])
            scales = np.sqrt(ws * hs)  # geometric scale s = √(w·h)
            aspect_ratios = ws / (hs + 1e-8)  # w / h

            # Compute stats
            median_scale = np.median(scales)
            mean_ar = np.mean(aspect_ratios)

            # Fit KDE in log-space for stability
            log_scales = np.log(scales + 1e-8).reshape(-1, 1)
            kde = KernelDensity(bandwidth=0.2, kernel='gaussian').fit(log_scales)

            self.priors[cat_id] = {
                'kde': kde,
                'median_scale': float(median_scale),
                'mean_ar': float(mean_ar),
                '_raw_scales': scales,  # fallback sampler if KDE fails
            }

        print(f"[ScalePriorManager] Fitted scale priors for {len(self.priors)} categories.")

    def sample_scale(self, cat_id):
        """Sample a scale from the learned distribution for category `cat_id`."""
        if not self.use_scale_prior or cat_id not in self.priors:
            return None
        try:
            log_s = self.priors[cat_id]['kde'].sample(1).item()
            return float(np.exp(log_s))
        except Exception:
            # Fallback to random sampling from raw data
            raw = self.priors[cat_id]['_raw_scales']
            return float(np.random.choice(raw))

    def get_median_scale(self, cat_id):
        if not self.use_scale_prior or cat_id not in self.priors:
            return 1.0
        return self.priors[cat_id]['median_scale']

    def get_mean_ar(self, cat_id):
        if not self.use_scale_prior or cat_id not in self.priors:
            return 1.0
        return self.priors[cat_id]['mean_ar']