import json
import lancedb
from pathlib import Path
from typing import Optional
from lancedb.pydantic import LanceModel, Vector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
DB_DIR = PROJECT_ROOT / "traffic_law_db"


class Doc(LanceModel):
    id: str
    law: str
    law_code: str
    article: str
    article_no: int
    chunk_type: str
    block_idx: Optional[int]
    item_no: Optional[str]
    points: Optional[int]
    status: str
    topic: str
    source: str
    embed_text: str
    raw_text: str
    vector: Vector(1024)


def main():
    in_path = OUTPUT_DIR / "chunks_with_vectors.json"
    with open(in_path, encoding="utf-8") as f:
        records = json.load(f)

    dim = len(records[0]["vector"])
    if dim != 1024:
        raise SystemExit(f"向量维度是 {dim}，不是 1024，请先重跑 step2")

    # 唯一性检查（P0）
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids)), "存在重复 id，请重跑 step1"

    # 兼容老数据
    for r in records:
        r.setdefault("block_idx", None)

    db = lancedb.connect(str(DB_DIR))
    table = db.create_table("chunks", schema=Doc, mode="overwrite")
    table.add(records)

    print(f"入库 {len(records)} 条（无索引，精确搜索）")
    print(f"表行数：{table.count_rows()}")


if __name__ == "__main__":
    main()