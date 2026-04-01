import os
import re
import pandas as pd

# ========= 你只需要改这里 =========
xlsx_path = "dataset_csv/二分型.xlsx"
slide_dir = "raw_slides"
output_csv = "dataset_csv/gcb_vs_nongcb.csv"

# Excel里这两列的列名，按你的表头改
case_col = "病理号"
label_col = "细胞起源分型"
# ================================


def extract_case_id_from_slide(filename_stem: str):
    """
    从 slide_id 中提取病理号(case_id)
    例子:
      1375314A01#3_1 -> 1375314
      1412190C01     -> 1412190
      X127179A01#3_9 -> X127179
    """
    m = re.match(r"^(X?\d+)", filename_stem)
    if not m:
        return None
    return m.group(1)


def normalize_case_id(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    s = s.replace(".0", "")
    s = s.replace(" ", "")
    return s


def normalize_label(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    s = s.replace(" ", "")
    s = s.replace("－", "-")
    s = s.replace("—", "-")
    s = s.replace("_", "-")
    if s.lower() == "gcb":
        return "GCB"
    if s.lower() in ["non-gcb", "nongcb", "non-gcb."]:
        return "non-GCB"
    return s


def main():
    # 1. 读取 Excel
    df = pd.read_excel(xlsx_path)

    if case_col not in df.columns:
        raise ValueError(f"Excel里找不到列: {case_col}，实际列名: {list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"Excel里找不到列: {label_col}，实际列名: {list(df.columns)}")

    df = df[[case_col, label_col]].copy()
    df["case_id"] = df[case_col].apply(normalize_case_id)
    df["label"] = df[label_col].apply(normalize_label)

    df = df[["case_id", "label"]].dropna()
    df = df.drop_duplicates(subset=["case_id"], keep="first")

    # 2. 扫描 raw_slides
    rows = []
    unmatched_slides = []

    for fname in sorted(os.listdir(slide_dir)):
        if not fname.lower().endswith(".sdpc"):
            continue

        slide_id = os.path.splitext(fname)[0]
        case_id = extract_case_id_from_slide(slide_id)

        if case_id is None:
            unmatched_slides.append((fname, "cannot_parse_case_id"))
            continue

        hit = df[df["case_id"] == case_id]
        if len(hit) == 0:
            unmatched_slides.append((fname, f"case_id_not_found:{case_id}"))
            continue

        label = hit.iloc[0]["label"]
        rows.append({
            "case_id": case_id,
            "slide_id": slide_id,
            "label": label
        })

    out_df = pd.DataFrame(rows)

    if len(out_df) == 0:
        raise RuntimeError("一个匹配到的slide都没有，请检查病理号列和文件名规则。")

    out_df = out_df.sort_values(["case_id", "slide_id"]).reset_index(drop=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"生成成功: {output_csv}")
    print(f"匹配到 {len(out_df)} 张 slides")
    print(f"覆盖 {out_df['case_id'].nunique()} 个 case")

    # 3. 输出未匹配列表，方便你排查
    if unmatched_slides:
        unmatched_df = pd.DataFrame(unmatched_slides, columns=["filename", "reason"])
        unmatched_path = output_csv.replace(".csv", "_unmatched.csv")
        unmatched_df.to_csv(unmatched_path, index=False, encoding="utf-8-sig")
        print(f"未匹配 slide 数量: {len(unmatched_df)}")
        print(f"未匹配明细已保存到: {unmatched_path}")
    else:
        print("所有 .sdpc 都匹配到了标签。")

    # 4. 顺手再导出一个只含 slide_id 的文件，提特征时也能复用
    feat_csv = output_csv.replace(".csv", "_for_features.csv")
    out_df[["slide_id"]].to_csv(feat_csv, index=False, encoding="utf-8-sig")
    print(f"特征提取用CSV已保存到: {feat_csv}")


if __name__ == "__main__":
    main()
