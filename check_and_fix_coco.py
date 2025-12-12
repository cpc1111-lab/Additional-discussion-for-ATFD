import json
import argparse
from pathlib import Path

def check_and_fix_coco(ann_file, num_classes=3, fix=True, output=None):
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"�� 正在检查标注文件: {ann_file}")
    print(f"��️ 图像数量: {len(data['images'])}")
    print(f"�� 标注数量: {len(data['annotations'])}")
    print(f"��️ 类别数量（元数据）: {len(data.get('categories', []))}")

    fixed = False

    # =============== Step 1: 检查并修复 image id ===============
    image_id_map = {}
    new_images = []
    for img in data['images']:
        orig_id = img['id']
        try:
            new_id = int(orig_id)
            if new_id != orig_id:
                print(f"�� 图像 ID 类型修正: '{orig_id}' → {new_id}")
                fixed = True
        except (ValueError, TypeError):
            print(f"❌ 无法将图像 ID 转为整数: {orig_id}")
            new_id = hash(str(orig_id)) % (2**31 - 1)
            print(f"�� 自动映射为: {new_id}")
            fixed = True
        image_id_map[orig_id] = new_id
        img['id'] = new_id
        new_images.append(img)
    data['images'] = new_images

    # =============== Step 2: 收集所有实际使用的 category_id ===============
    used_cat_ids = set()
    for ann in data['annotations']:
        cat_id = ann.get('category_id')
        if cat_id is not None:
            used_cat_ids.add(cat_id)
        # 同时修正 image_id 引用
        orig_img_id = ann.get('image_id')
        if orig_img_id in image_id_map:
            new_img_id = image_id_map[orig_img_id]
            if ann['image_id'] != new_img_id:
                ann['image_id'] = new_img_id
                fixed = True
        else:
            print(f"⚠️ 标注引用了不存在的 image_id: {orig_img_id}")

    print(f"�� 实际使用的 category_id: {sorted(used_cat_ids)}")
    actual_num_classes = len(used_cat_ids)

    # =============== Step 3: 检查是否从 0 开始且连续 ===============
    expected_ids = set(range(num_classes))
    is_continuous_from_zero = (used_cat_ids == expected_ids)

    if not is_continuous_from_zero:
        print(f"❌ 类别 ID 不是从 0 开始的连续整数！")
        print(f"   实际类别数: {actual_num_classes}, 期望: {num_classes}")
        if actual_num_classes != num_classes:
            print(f"❗ 实际类别数量 ({actual_num_classes}) 与 --num-classes ({num_classes}) 不符！")
        else:
            print("�� 将尝试重映射为 [0, 1, ..., num_classes-1]")

        # 构建映射：旧 category_id → 新 category_id（从 0 开始）
        sorted_old_ids = sorted(used_cat_ids)
        if len(sorted_old_ids) != num_classes:
            print("�� 无法安全重映射：实际类别数 ≠ num_classes")
            if not fix:
                return None
            else:
                # 即使数量不符，仍尝试映射（但可能丢失或重复）
                print("⚠️ 强制重映射（可能不准确）...")
        cat_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted_old_ids[:num_classes])}
        # 如果实际类别多于 num_classes，只取前 num_classes 个
        # 如果少于，则映射后可能缺类（但用户应确保数据正确）

        # 应用映射到 annotations
        for ann in data['annotations']:
            old_cat = ann['category_id']
            if old_cat in cat_id_map:
                new_cat = cat_id_map[old_cat]
                if new_cat != old_cat:
                    ann['category_id'] = new_cat
                    fixed = True
            else:
                # 未映射的类别（如多余类别）——应删除或报错
                if fix:
                    print(f"��️ 删除使用未映射类别 {old_cat} 的标注")
                    # 我们稍后统一过滤
                    ann['_invalid'] = True
                else:
                    print(f"❓ 标注使用了无法映射的 category_id: {old_cat}")

        # 过滤掉无效标注（标记了 _invalid 的）
        original_ann_count = len(data['annotations'])
        data['annotations'] = [ann for ann in data['annotations'] if not ann.get('_invalid', False)]
        if len(data['annotations']) != original_ann_count:
            fixed = True
            print(f"✅ 移除了 {original_ann_count - len(data['annotations'])} 条无效标注")

        # 更新 categories 元数据
        new_categories = []
        old_id_to_cat = {cat['id']: cat for cat in data.get('categories', [])}

        for new_id in range(num_classes):
            old_id = sorted_old_ids[new_id] if new_id < len(sorted_old_ids) else None
            if old_id in old_id_to_cat:
                cat = old_id_to_cat[old_id].copy()
                cat['id'] = new_id
                new_categories.append(cat)
            else:
                # 创建占位类别
                new_categories.append({"id": new_id, "name": f"class_{new_id}"})
                print(f"�� 添加缺失类别: class_{new_id}")

        data['categories'] = new_categories
        print("✅ 已重映射类别 ID 并更新 categories 字段")
    else:
        print("✅ 类别 ID 已是从 0 开始的连续整数")

    # =============== Step 4: 验证最终状态 ===============
    final_used = {ann['category_id'] for ann in data['annotations']}
    final_expected = set(range(num_classes))
    missing_in_ann = final_expected - final_used
    extra_in_ann = final_used - final_expected

    if missing_in_ann:
        print(f"ℹ️ 注意：类别 {missing_in_ann} 未在标注中使用")
    if extra_in_ann:
        print(f"❌ 错误：标注中存在超出 [0, {num_classes-1}] 的类别: {extra_in_ann}")
        # 这不应发生，除非映射出错
        return None

    # =============== 输出结果 ===============
    print(f"\n✅ 检查完成！")
    print(f"   - 图像 ID 均为整数: ✔️")
    print(f"   - 类别 ID 从 0 开始且数量为 {num_classes}: {'✔️' if is_continuous_from_zero or fixed else '❌'}")

    if fixed and fix:
        out_path = output or (Path(ann_file).parent / f"{Path(ann_file).stem}_fixed.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"�� 已保存修复后的文件: {out_path}")
        return str(out_path)
    else:
        if not is_continuous_from_zero:
            print("❗ 类别 ID 不符合要求（如需自动修复，请添加 --fix 参数）")
            return None
        else:
            print("✅ 无需修复")
            return ann_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查并修复 COCO 标注文件：确保类别从 0 开始连续，数量匹配 --num-classes")
    parser.add_argument("input", help="输入的 COCO JSON 文件路径")
    parser.add_argument("--num-classes", type=int, default=3, help="期望的类别数量（默认: 3）")
    parser.add_argument("--fix", action="store_true", help="自动修复类别 ID 和元数据")
    parser.add_argument("-o", "--output", help="修复后输出路径（可选）")
    args = parser.parse_args()

    result = check_and_fix_coco(
        args.input,
        num_classes=args.num_classes,
        fix=args.fix,
        output=args.output
    )
    if result is None:
        exit(1)