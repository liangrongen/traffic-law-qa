import re
import json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

ARTICLE_RE = re.compile(r"^\s*第([零一二三四五六七八九十百千\d]+)条", re.MULTILINE)
ITEM_RE = re.compile(r"^（([一二三四五六七八九十]+)）")

EXPECTED_MAX = {
    "rtpoints2022": 37,
    "rtaccident2018": 114,
    "rtregulation2017": 115,
    "rtlaw2021": 124,
}

CN_DIGITS = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CN_UNITS = {"十": 10, "百": 100, "千": 1000}


def cn_to_int(cn: str) -> int:
    if cn.isdigit():
        return int(cn)
    result, current = 0, 0
    for ch in cn:
        if ch in CN_DIGITS:
            current = CN_DIGITS[ch]
        elif ch in CN_UNITS:
            if current == 0:
                current = 1
            result += current * CN_UNITS[ch]
            current = 0
    result += current
    return result


def clean_text(text: str) -> str:
    text = text.lstrip("\ufeff")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_by_article(text: str):
    parts = re.split(r"(?=^[ \t]*第[零一二三四五六七八九十百千\d]+条)", text, flags=re.MULTILINE)
    return [p.strip() for p in parts if ARTICLE_RE.match(p)]


def split_article_into_blocks(article_text: str):
    """
    把一条法规切成若干列举块，按行扫描识别。
    返回 [(lead, [(item_no, item_text), ...]), ...]
    无列举时返回 [(article_text, [])]
    """
    lines = [l for l in article_text.split("\n") if l.strip()]
    pending_lead = []
    current_lead = None
    current_items = []
    blocks = []

    def flush():
        nonlocal current_lead, current_items
        if current_items:
            blocks.append((current_lead, list(current_items)))
            current_items = []
            current_lead = None

    for line in lines:
        m = ITEM_RE.match(line)
        if m:
            item_no = m.group(1)
            if current_items:
                last_no = current_items[-1][0]
                if cn_to_int(item_no) <= cn_to_int(last_no):
                    # 序号回退 → 开始新列举块
                    flush()
                    current_lead = "\n".join(pending_lead).strip()
                    pending_lead = []
            else:
                # 空闲状态遇到列举项 → 开始新块
                current_lead = "\n".join(pending_lead).strip()
                pending_lead = []
            current_items.append((item_no, line))
        else:
            if current_items:
                flush()
            pending_lead.append(line)

    flush()

    if not blocks:
        return [(article_text, [])]
    return blocks


def extract_points(lead: str):
    if not lead:
        return None
    m = re.search(r"一次记(\d+)分", lead)
    return int(m.group(1)) if m else None


def process_file(cfg):
    with open(cfg["path"], encoding="utf-8") as f:
        raw = f.read()
    text = clean_text(raw)
    articles = split_by_article(text)
    records = []

    for art in articles:
        m = ARTICLE_RE.match(art)
        if not m:
            continue
        article_cn = m.group(1)
        article_no = cn_to_int(article_cn)
        article_label = f"第{article_cn}条"

        blocks = split_article_into_blocks(art)
        has_items = any(items for _, items in blocks)

        # 父条 chunk：有列举时 embed_text 只放首个 lead 摘要；否则放全文
        if has_items:
            lead_for_embed = (blocks[0][0] or "")[:400]
            article_embed_text = (
                f"{cfg['law']} {cfg['status']} {article_label} {lead_for_embed}"
            )
        else:
            article_embed_text = f"{cfg['law']} {cfg['status']} {art}"

        records.append({
            "id": f"{cfg['law_code']}-art-{article_no:04d}",
            "law": cfg["law"], "law_code": cfg["law_code"],
            "article": article_label, "article_no": article_no,
            "chunk_type": "article", "block_idx": None,
            "item_no": None, "points": None,
            "status": cfg["status"], "topic": cfg["topic"],
            "source": f"{cfg['law']} {article_label}",
            "embed_text": article_embed_text,
            "raw_text": art,
        })

        for block_idx, (lead, items) in enumerate(blocks, 1):
            if not items:
                continue
            points = extract_points(lead)
            for item_no, item_text in items:
                pts_hint = f"一次记{points}分 " if points else ""
                item_embed_text = (
                    f"{cfg['law']} {cfg['status']} {article_label} "
                    f"{pts_hint}第（{item_no}）项 {item_text}"
                )
                if len(blocks) > 1:
                    source = f"{cfg['law']} {article_label}第{block_idx}款第（{item_no}）项"
                else:
                    source = f"{cfg['law']} {article_label}第（{item_no}）项"

                records.append({
                    "id": f"{cfg['law_code']}-art-{article_no:04d}-blk-{block_idx}-it-{item_no}",
                    "law": cfg["law"], "law_code": cfg["law_code"],
                    "article": article_label, "article_no": article_no,
                    "chunk_type": "item", "block_idx": block_idx,
                    "item_no": item_no, "points": points,
                    "status": cfg["status"], "topic": cfg["topic"],
                    "source": source,
                    "embed_text": item_embed_text,
                    "raw_text": item_text,
                })
    return records


def quality_report(all_records):
    print("\n=== 数据质量报告 ===")
    ids = [r["id"] for r in all_records]
    dup = [k for k, v in Counter(ids).items() if v > 1]
    print(f"总记录: {len(all_records)}, 唯一 id: {len(set(ids))}, 重复 id: {len(dup)}")
    if dup:
        print(f"  重复 id 示例: {dup[:10]}")

    empty = [r["id"] for r in all_records if not r["raw_text"].strip()]
    print(f"空文本: {len(empty)}")
    if empty:
        print(f"  示例: {empty[:5]}")

    overlong = [(r["id"], len(r["embed_text"])) for r in all_records if len(r["embed_text"]) > 4500]
    print(f"embed_text 超过 4500: {len(overlong)}")
    if overlong:
        print(f"  示例: {overlong[:5]}")

    points_dist = Counter(r["points"] for r in all_records if r["chunk_type"] == "item")
    print(f"points 分布: {dict(points_dist)}")
    bad_points = [p for p in points_dist if p is not None and p not in {1, 3, 6, 9, 12}]
    if bad_points:
        print(f"  异常分值: {bad_points}")

    print("\n=== 法条连续性 ===")
    for code, mx in EXPECTED_MAX.items():
        nos = sorted({r["article_no"] for r in all_records if r["law_code"] == code})
        missing = sorted(set(range(1, mx + 1)) - set(nos))
        over = sorted(n for n in nos if n > mx)
        n_chunks = sum(1 for r in all_records if r["law_code"] == code)
        n_items = sum(1 for r in all_records if r["law_code"] == code and r["chunk_type"] == "item")
        print(f"  {code}: {n_chunks} chunks ({n_items} items), 缺失 {missing or '无'}, 超出 {over or '无'}")


def main():
    FILES = [
        {"path": DATA_DIR / "记分管理办法.txt", "law": "道路交通安全违法行为记分管理办法",
         "law_code": "rtpoints2022", "status": "现行有效", "topic": "记分"},
        {"path": DATA_DIR / "事故处理程序规定.txt", "law": "道路交通事故处理程序规定",
         "law_code": "rtaccident2018", "status": "现行有效", "topic": "事故处理程序"},
        {"path": DATA_DIR / "实施条例.txt", "law": "中华人民共和国道路交通安全法实施条例",
         "law_code": "rtregulation2017", "status": "现行有效", "topic": "实施条例"},
        {"path": DATA_DIR / "道路交通安全法.txt", "law": "中华人民共和国道路交通安全法",
         "law_code": "rtlaw2021", "status": "现行有效", "topic": "道路交通安全法"},
    ]

    all_records = []
    for cfg in FILES:
        recs = process_file(cfg)
        n_article = sum(1 for r in recs if r["chunk_type"] == "article")
        n_item = sum(1 for r in recs if r["chunk_type"] == "item")
        print(f"{cfg['law']}: {len(recs)} chunks (父条 {n_article}, 列举项 {n_item})")
        all_records.extend(recs)

    ids = [r["id"] for r in all_records]
    assert len(ids) == len(set(ids)), "存在重复 id，切分仍有问题，请检查"

    quality_report(all_records)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "chunks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    print(f"\n总计 {len(all_records)} chunks → {out_path}")


if __name__ == "__main__":
    main()