\# 交通法问答系统



基于 RAG（检索增强生成）的交通法规问答系统：LanceDB 向量检索 + Ollama 本地模型。



\## 技术栈



\- 检索：`bge-m3`（embedding）+ LanceDB（cosine 精确搜索）

\- 生成：`qwen2.5:7b`（Ollama 本地推理）

\- Web：Flask + SSE 流式输出



\## 目录结构



```text

project/

├── app.py                  # Flask 入口

├── requirements.txt        # Python 依赖

├── templates/index.html    # 前端页面

├── scripts/

│   ├── query\_normalize.py  # 查询归一化 + 指代改写

│   ├── step1\_split.py      # 法规切分

│   ├── step2\_embed.py      # bge-m3 向量化

│   ├── step3\_ingest.py     # 写入 LanceDB

│   └── step4\_test.py       # 检索自测

├── data/                   # 法规原文

└── output/                 # 切分结果

