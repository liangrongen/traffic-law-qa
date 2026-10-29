import json
import math
import time
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

MAX_CHARS = 4500
OLLAMA_URL = "http://localhost:11434/api/embed"


def truncate(text: str) -> str:
    return text[:MAX_CHARS] if len(text) > MAX_CHARS else text


def l2_normalize(vec):
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def embed_batch(texts, session, batch_size=8, max_retries=3):
    vectors = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = [truncate(t) for t in texts[i:i + batch_size]]
        payload = {"model": "bge-m3", "input": batch, "truncate": True}

        for attempt in range(1, max_retries + 1):
            try:
                resp = session.post(OLLAMA_URL, json=payload, timeout=600)
            except requests.exceptions.ConnectionError as e:
                print(f"  连接错误 (尝试 {attempt}/{max_retries}): {e}")
                time.sleep(3)
                continue
            except requests.exceptions.Timeout as e:
                print(f"  超时 (尝试 {attempt}/{max_retries}): {e}")
                time.sleep(3)
                continue
            except requests.exceptions.RequestException as e:
                print(f"  请求异常 (尝试 {attempt}/{max_retries}): {e}")
                time.sleep(3)
                continue

            if resp.status_code == 200:
                vectors.extend(resp.json()["embeddings"])
                print(f"  {min(i + batch_size, total)}/{total}")
                break

            print(f"  HTTP {resp.status_code} (尝试 {attempt}/{max_retries})")
            print(f"  响应体: {resp.text[:300]}")
            # 400 大概率是某条超长，降级为逐条
            if resp.status_code == 400 and len(batch) > 1:
                print("  降级为逐条请求...")
                for t in batch:
                    r2 = session.post(
                        OLLAMA_URL,
                        json={"model": "bge-m3", "input": [t], "truncate": True},
                        timeout=600,
                    )
                    if r2.status_code != 200:
                        raise RuntimeError(f"单条也失败: {r2.text}")
                    vectors.append(r2.json()["embeddings"][0])
                    time.sleep(0.1)
                break
            time.sleep(3)
        else:
            raise RuntimeError(f"批次 {i} 重试 {max_retries} 次仍失败")

        time.sleep(0.2)

    return [l2_normalize(v) for v in vectors]


def main():
    in_path = OUTPUT_DIR / "chunks.json"
    with open(in_path, encoding="utf-8") as f:
        records = json.load(f)

    texts = [r["embed_text"] for r in records]
    over = [i for i, t in enumerate(texts) if len(t) > MAX_CHARS]
    print(f"共 {len(texts)} 条，其中 {len(over)} 条超过 {MAX_CHARS} 字符将被截断")
    if over:
        print("超长条目索引:", over[:10], "..." if len(over) > 10 else "")

    print("开始向量化...")
    session = requests.Session()
    vectors = embed_batch(texts, session)

    assert len(vectors) == len(records), f"向量数 {len(vectors)} ≠ 记录数 {len(records)}"
    assert all(len(v) == 1024 for v in vectors), "维度不是 1024，请确认用的是 bge-m3"

    for r, v in zip(records, vectors):
        r["vector"] = v

    out_path = OUTPUT_DIR / "chunks_with_vectors.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    print(f"完成 {len(records)} 条 → {out_path}")


if __name__ == "__main__":
    main()