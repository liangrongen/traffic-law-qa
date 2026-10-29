import lancedb
import requests
import pandas as pd
from pathlib import Path
from query_normalize import build_retrieval_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / "traffic_law_db"


def embed_query(q):
    resp = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "bge-m3", "input": [q]},
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def main():
    db = lancedb.connect(str(DB_DIR))
    table = db.open_table("chunks")

    questions = [
        "超速20%不到50%在高速上扣几分",
        "醉驾营运车几年不能重考",
        "事故后逃逸责任怎么认定",
        "高速上小客车最高能开多少",
        "机动车撞闯红灯行人无责还要赔吗",
        "记1分的情形有哪些",
        "交通事故认定书几天内制作",
        "电动自行车在非机动车道限速多少",
        "饮酒驾驶私家车怎么处罚",
        "高速上违法停车扣几分",
        "普通道路超速20%不到50%扣几分",
    ]

    for q in questions:
        retrieval_text = build_retrieval_text(q)
        vec = embed_query(retrieval_text)
        results = table.search(vec).distance_type("cosine").limit(5).to_pandas()
        print(f"\nQ: {q}")
        if retrieval_text != q:
            print(f"   [检索文本] {retrieval_text}")
        for _, row in results.iterrows():
            score = None
            if "_distance" in row and pd.notna(row["_distance"]):
                score = 1 - float(row["_distance"])
            score_str = f" score={score:.4f}" if score is not None else ""
            pts_str = ""
            if "points" in row and pd.notna(row["points"]):
                pts_str = f" [记{int(row['points'])}分]"
            snippet = row["raw_text"].replace("\n", " ")[:70]
            print(f"  [{row['source']}]{score_str}{pts_str} {snippet}")


if __name__ == "__main__":
    main()