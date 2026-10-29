import json
import math
import re
import sys
import time
import logging
from pathlib import Path
from functools import lru_cache
from collections import OrderedDict

import requests
import lancedb
import pandas as pd
from flask import Flask, request, Response, render_template, jsonify

PROJECT_ROOT = Path(__file__).resolve().parent
DB_DIR = PROJECT_ROOT / "traffic_law_db"
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from query_normalize import build_retrieval_text

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "bge-m3"
CHAT_MODEL = "qwen2.5:7b"
TOP_K = 5
TOP_K_RETRIEVE = 20
MAX_HISTORY_MSGS = 4
MAX_HISTORY_CHARS = 1500

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("traffic_law")

db = lancedb.connect(str(DB_DIR))
table = db.open_table("chunks")

_ALL_DATA = None


def get_all_data():
    """加载全量数据（不含向量），用于列举类问题过滤"""
    global _ALL_DATA
    if _ALL_DATA is None:
        df = table.to_pandas()
        if "vector" in df.columns:
            df = df.drop(columns=["vector"])
        _ALL_DATA = df
    return _ALL_DATA


@lru_cache(maxsize=256)
def _embed_query_cached(q: str):
    r = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": [q]},
        timeout=30,
    )
    r.raise_for_status()
    vec = r.json()["embeddings"][0]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return tuple(x / norm for x in vec)


def embed_query(q):
    return list(_embed_query_cached(q))


def swap_road(q: str, target: str) -> str:
    """长词优先替换，避免'高速公路'被拆成'普通公路公路'"""
    if target == "普通公路":
        q = q.replace("高速公路", "普通公路")
        q = q.replace("城市快速路", "普通公路")
        q = q.replace("快速路", "普通公路")
        q = q.replace("高速", "普通公路")
    else:
        q = q.replace("普通公路", "高速公路")
        q = q.replace("普通道路", "高速公路")
    return q


def rewrite_query_with_history(question, history):
    """基于规则的指代消解，触发条件收紧"""
    if not history or not question.strip():
        return question

    q = question.strip()
    is_followup = (
        q.startswith(("那", "呢", "这个", "它", "还有", "换成"))
        or q in {"普通道路呢", "高速呢", "那普通呢", "那高速呢",
                 "普通公路呢", "高速公路呢", "那普通公路呢", "那高速公路呢"}
        or (len(q) <= 6 and ("呢" in q or "那" in q))
    )
    if not is_followup:
        return question

    last_user_q = next(
        (h["content"] for h in reversed(history) if h.get("role") == "user"),
        "",
    )
    if not last_user_q:
        return question

    if "普通" in q:
        return swap_road(last_user_q, "普通公路")
    elif "高速" in q:
        return swap_road(last_user_q, "高速公路")
    else:
        return f"{last_user_q} {q}"


def detect_enumeration_query(q: str):
    """'记X分有哪些情形' 检测"""
    m = re.search(r"记\s*(\d+)\s*分", q)
    if not m:
        return None
    if any(w in q for w in ["有哪些", "情形", "包括", "清单", "哪些", "列举"]):
        return int(m.group(1))
    return None


def retrieve_enumeration(points: int, k: int = 20):
    df = get_all_data()
    mask = (df["chunk_type"] == "item") & (df["points"] == points)
    sub = df[mask].copy()
    if sub.empty:
        return sub
    sub = sub.sort_values(["law_code", "article_no", "block_idx", "item_no"])
    return sub.head(k).reset_index(drop=True)


def dedup_by_article(df, k):
    """同一法条只保留一条，item 优先于 article"""
    if df.empty:
        return df

    groups = OrderedDict()
    for _, row in df.iterrows():
        key = (row["law_code"], int(row["article_no"]))
        groups.setdefault(key, []).append(row)

    result_rows = []
    for key, rows in groups.items():
        items = [r for r in rows if r["chunk_type"] == "item"]
        articles = [r for r in rows if r["chunk_type"] == "article"]
        chosen = (items or articles)[:1]
        if chosen:
            result_rows.append(chosen[0])

    if not result_rows:
        return df.head(0)

    if "_distance" in df.columns:
        result_rows.sort(key=lambda r: r["_distance"])
    return pd.DataFrame(result_rows).head(k).reset_index(drop=True)


def retrieve(question, history, k=TOP_K):
    standalone_q = rewrite_query_with_history(question, history)
    retrieval_text = build_retrieval_text(standalone_q)
    logger.info(f"[检索] {question!r} -> {standalone_q!r} -> {retrieval_text!r}")

    points = detect_enumeration_query(standalone_q)
    if points is not None:
        logger.info(f"[列举] points={points}")
        sub = retrieve_enumeration(points)
        if len(sub) > 0:
            return sub, True

    vec = embed_query(retrieval_text)
    results = table.search(vec).distance_type("cosine").limit(TOP_K_RETRIEVE).to_pandas()
    deduped = dedup_by_article(results, k)
    return deduped, False


def build_messages(question, contexts, history, is_enumeration=False):
    ctx_parts = []
    for i, (_, row) in enumerate(contexts.iterrows()):
        pts = row.get("points")
        pts_str = f"【记{int(pts)}分】" if pd.notna(pts) else ""
        ctx_parts.append(f"[{i + 1}] {row['source']} {pts_str}\n{row['raw_text']}")
    ctx = "\n\n".join(ctx_parts)

    enum_hint = ""
    if is_enumeration:
        enum_hint = (
            "\n注意：用户问的是列举类问题，资料已按分值筛选出全部相关项，"
            "请完整列出，不要遗漏。"
        )

    system = (
        "你是严谨的交通法问答助手。请严格根据下面提供的【资料】回答用户问题。\n"
        "【规则】：\n"
        "1. 如果【资料】中有明确答案，请直接给出答案，并在末尾注明依据的资料编号。\n"
        "2. 若资料中的适用条件与问题明显不匹配（如资料仅限高速，而问题问普通道路），"
        "请先说明“该资料仅适用于……”，再给出资料中可适用的一般规则；"
        "若无法确定，回答“根据现有资料无法确定”。不要将不适用的规则强行套用。\n"
        "3. 如果【资料】中完全没有提及用户问的违法行为，直接回答“根据现有资料无法回答”，严禁编造。\n"
        "4. 回答必须简练、准确。罚款只给法定幅度，不猜地方标准。"
        "记分、期限、数值必须与资料一致；区分车型、道路、营运/非营运、是否构成犯罪。\n"
        "5. 资料冲突时说明冲突，并优先特别规定/上位法。\n"
        "6. 禁止输出 markdown 公式符号（如 \\boxed）。"
        f"{enum_hint}"
    )

    messages = [{"role": "system", "content": system}]

    # 历史注入，按字符上限裁剪
    total_chars = 0
    history_msgs = []
    for h in reversed(history[-MAX_HISTORY_MSGS:]):
        if h.get("role") not in ("user", "assistant"):
            continue
        content = (h.get("content") or "").strip()
        if not content:
            continue
        if total_chars + len(content) > MAX_HISTORY_CHARS:
            break
        history_msgs.append({"role": h["role"], "content": content})
        total_chars += len(content)
    for m in reversed(history_msgs):
        messages.append(m)

    messages.append({
        "role": "user",
        "content": f"【资料】\n{ctx}\n\n【问题】\n{question}",
    })
    return messages


def stream_ollama(messages):
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 1024,
                "seed": 42,
            },
        },
        stream=True,
        timeout=300,
    )
    resp.raise_for_status()

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = obj.get("message", {})
        content = msg.get("content", "")
        if content:
            yield ("content", content)
        if obj.get("done"):
            break


def sse(data):
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.route("/health")
def health():
    status = {"db": False, "ollama": False}
    try:
        table.count_rows()
        status["db"] = True
    except Exception as e:
        status["db_error"] = str(e)
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        status["ollama"] = r.status_code == 200
    except Exception as e:
        status["ollama_error"] = str(e)
    return jsonify(status)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    question = (data.get("question") or "").strip()
    history = data.get("history") or []
    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    t0 = time.time()
    try:
        contexts, is_enum = retrieve(question, history)
    except Exception as e:
        logger.exception("检索失败")
        return jsonify({"error": f"检索失败：{e}"}), 500

    sources = []
    if not contexts.empty:
        for _, row in contexts.iterrows():
            item = {"source": row["source"], "raw_text": row["raw_text"]}
            if "points" in row and pd.notna(row["points"]):
                item["points"] = int(row["points"])
            if "_distance" in row and pd.notna(row["_distance"]):
                item["distance"] = float(row["_distance"])
            sources.append(item)

    messages = build_messages(question, contexts, history, is_enumeration=is_enum)
    logger.info(
        f"[请求] 问题={question!r} 命中={len(contexts)} 列举={is_enum} "
        f"检索耗时={time.time() - t0:.2f}s"
    )

    def generate():
        yield sse({"type": "sources", "sources": sources})
        try:
            for t, text in stream_ollama(messages):
                yield sse({"type": t, "text": text})
        except Exception as e:
            logger.exception("LLM 流式失败")
            yield sse({"type": "error", "message": str(e)})
        yield sse({"type": "done"})

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)