import json
import argparse
from pathlib import Path

def convert_coco_str_id_to_int(input_json, output_json=None):
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"原始图像数量: {len(data['images'])}")
    print(f"原始标注数量: {len(data['annotations'])}")

    # 构建从原始 image id（可能为字符串）到新整数 id 的映射
    id_map = {}
    new_images = []

    for idx, img in enumerate(data['images']):
        old_id = img['id']
        # 使用索引+1 作为新ID（也可用 hash，但需确保唯一且为正整数）
        new_id = idx + 1
        id_map[old_id] = new_id

        # 复制图像信息，仅替换 id
        new_img = img.copy()
        new_img['id'] = new_id
        new_images.append(new_img)

    # 更新 annotations 中的 image_id
    new_annotations = []
    missing_count = 0
    for ann in data['annotations']:
        old_img_id = ann['image_id']
        if old_img_id not in id_map:
            print(f"警告：标注中 image_id={old_img_id} 在 images 中未找到！")
            missing_count += 1
            continue
        new_ann = ann.copy()
        new_ann['image_id'] = id_map[old_img_id]
        new_annotations.append(new_ann)

    if missing_count > 0:
        print(f"跳过了 {missing_count} 条无效标注。")

    # 构建新数据
    new_data = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': data.get('categories', [])
    }

    # 输出
    out_path = output_json or (Path(input_json).parent / f"{Path(input_json).stem}_int_id.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 转换完成！新文件已保存至: {out_path}")
    print(f"新图像数量: {len(new_images)}")
    print(f"新标注数量: {len(new_annotations)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 COCO 标注文件中的字符串 image id 转换为整数 ID")
    parser.add_argument("input", help="输入的 COCO JSON 文件路径")
    parser.add_argument("-o", "--output", help="输出文件路径（可选）")
    args = parser.parse_args()

    convert_coco_str_id_to_int(args.input, args.output)