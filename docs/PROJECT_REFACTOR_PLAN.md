# AgentMedRAG 项目重构设计文档

## 1. 文档目的

本文档用于指导 `GR5293-AgentMedRag` 从当前的 notebook 实验项目，重构为一个更专业、更可复现、更可扩展的医学文献问答 Agent 系统。

当前项目已经具备一个医学 RAG 系统的核心雏形：

- 使用 PubMed 文章摘要作为医学知识来源。
- 使用 `download_pubmed.py` 下载 PubMed 文献元数据和摘要。
- 使用 notebook 生成 QA pair。
- 使用 LangChain 加载 JSON 文档、切分文本、构建向量库。
- 使用 BGE embedding、Chroma、ColBERT、BGE reranker 等组件进行检索和重排序实验。
- 使用 Mistral 7B 或 OpenAI 模型生成答案。
- 使用 RAGAS 评估 faithfulness、answer relevancy、context precision、context recall 等指标。

但是当前项目仍然更接近一个课程项目或研究 notebook，尚未达到专业 Agent 系统的形态。重构目标不是简单地把 notebook 包一层命令行，而是把项目升级为一个结构清晰、模块可替换、输出可信、评估可复现、可以演示和部署的 Medical RAG Agent。

## 2. 当前项目概览

### 2.1 顶层文件

当前仓库主要文件包括：

- `README.md`
  - 描述了项目 pipeline。
  - 包含环境安装命令。
  - 包含常见错误排查表。

- `download_pubmed.py`
  - 使用 BioPython `Entrez` API 从 PubMed 下载文章。
  - 根据开始日期、结束日期、最大文章数量进行检索。
  - 保存文章标题、摘要、发布日期到 JSON。

- `generatequery.ipynb`
  - 从 PubMed 摘要生成 question-answer pairs。
  - 生成后的 QA 数据保存到 `googledrive-finalproject/generated_qa_pairs*.json`。

- `may15_rag_mainscript.ipynb`
  - 当前项目的核心 RAG notebook。
  - 包含文档加载、文本切分、embedding、向量库、rerank、LLM 生成、RAGAS 评估等实验逻辑。

- `Copy_of_intro2new.ipynb`
  - ColBERTv2 indexing/search demo notebook。
  - 更像是 ColBERT 教程或实验参考。

- `googledrive-finalproject/`
  - 存放数据和实验产物。
  - 包括 `pubmed_article.json`、不同规模的 QA pair JSON、ColBERT index plan 等。

### 2.2 当前数据

当前仓库中已有的数据文件：

- `googledrive-finalproject/pubmed_article.json`
  - 当前包含约 2549 篇 PubMed 文章。
  - 每条记录包含：
    - `article_title`
    - `article_abstract`
    - `pub_date.year`
    - `pub_date.month`
    - `pub_date.day`

- `googledrive-finalproject/generated_qa_pairs.json`
  - 小规模 QA pair 示例。

- `googledrive-finalproject/generated_qa_pairs30.json`
- `googledrive-finalproject/generated_qa_pairs80.json`
- `googledrive-finalproject/generated_qa_pairs100.json`
- `googledrive-finalproject/generated_qa_pairs200.json`
  - 不同规模的问答评估数据。

### 2.3 当前 RAG 流程

当前 notebook 中的主流程大致是：

1. 从 Google Drive 或本地 JSON 读取 PubMed 数据。
2. 使用 LangChain `JSONLoader` 加载文章摘要。
3. 用 metadata function 提取文章标题和发布日期。
4. 使用 `TokenTextSplitter` 将摘要切分为 chunk。
5. 使用 `BAAI/bge-large-en-v1.5` 或其他 embedding 模型生成向量。
6. 使用 Chroma 或 FAISS 建立向量库。
7. 使用 retriever 检索 top-k 候选文档。
8. 使用 `BAAI/bge-reranker-base` 对候选文档进行 rerank。
9. 使用 Mistral 7B 或 OpenAI `gpt-4o` 基于上下文生成答案。
10. 使用 RAGAS 计算 faithfulness、answer relevancy、context precision、context recall 等指标。

### 2.4 教程内容与当前项目的关系

用户提供的 PDF 教程标题是：

> 使用 LangChain 和 Mistral 7B 构建医疗问答系统

教程中的主线是一个典型 RAG baseline：

1. 使用 PubMed 作为医学知识来源。
2. JSON 文件包含文章标题、出版日期和文章摘要。
3. 使用 LangChain `JSONLoader` 加载数据。
4. 使用 `TokenTextSplitter` 切分文档。
5. 使用 sentence-transformers / HuggingFace embedding 生成向量。
6. 使用 FAISS 建立向量库。
7. 使用 Mistral 7B 作为本地 LLM。
8. 使用 LangChain `RetrievalQA` 构建问答链。
9. 对比 RAG+LLM 与纯 LLM 的效果。
10. 结论是 RAG 能提升医学问答的准确性、可靠性和可解释性，但会增加检索和生成时间。

当前仓库已经覆盖了教程的大部分内容，并且已有一些增强：

- 不只使用 FAISS，还尝试了 Chroma、Milvus、ColBERT。
- 不只使用普通 embedding，还使用了 BGE embedding。
- 不只检索，还加入了 BGE reranker。
- 不只做单次问答，还生成了 QA pair 用于评估。
- 不只人工比较结果，还引入了 RAGAS 自动评估。

因此，后续重构应该站在教程基础之上，将项目从“RAG demo”提升为“医学 Agent 系统”。

## 3. 当前项目主要问题

### 3.1 notebook 逻辑过重

当前核心逻辑集中在 notebook 中，问题包括：

- 难以复现完整 pipeline。
- 难以测试单个组件。
- 变量状态依赖执行顺序。
- Colab 路径和本地路径混杂。
- 实验代码、正式逻辑、调试代码混在一起。
- 不利于命令行、API 或 Web UI 封装。

专业项目应该将 notebook 降级为 demo 或实验入口，将核心逻辑迁移到 Python package。

### 3.2 缺少明确模块边界

当前代码中 ingestion、chunking、indexing、retrieval、reranking、generation、evaluation 都散在 notebook cell 中。

重构后应形成清晰模块：

- 数据下载
- 文档加载
- 文本切分
- 索引构建
- 检索
- 重排序
- 答案生成
- citation 验证
- 安全 guardrail
- 自动评估
- CLI/API/UI

### 3.3 配置硬编码

当前存在以下硬编码问题：

- Google Drive 路径，如 `/content/drive/MyDrive/finalproject/...`
- embedding 模型名称散落在 notebook 中
- chunk size、overlap、retrieval k、rerank top-k 写死
- OpenAI model 写死为 `gpt-4o`
- HuggingFace model 写死为 `mistralai/Mistral-7B-v0.1`
- Chroma persist directory 写死为 `./chroma_db`

专业项目应使用配置文件统一管理这些参数。

### 3.4 缺少医学安全边界

医学问答系统必须明确边界：

- 不应提供个人诊断。
- 不应替代医生。
- 对治疗、药物、剂量、急症问题要谨慎。
- 证据不足时应拒绝过度推断。
- 应区分“文献提到”与“临床推荐”。

当前 prompt 主要约束“基于 context 回答”，但没有完整医疗安全策略。

### 3.5 citation 不够规范

当前 notebook 会返回 source documents 或标题，但尚未形成标准 citation 输出。

专业医学 RAG 应该：

- 每个关键结论都能追溯到来源文章。
- 输出文章标题、日期、摘要证据片段。
- 支持 source ID 或 PubMed ID。
- 答案中明确哪些结论来自哪些文献。

### 3.6 评估流程未产品化

RAGAS 已经被引入，但仍是 notebook cell 形式。

问题包括：

- 评估输入格式不统一。
- 评估指标配置不统一。
- 评估结果没有标准报告。
- 难以比较不同 embedding / reranker / chunk 参数组合。

重构后应有独立评估模块和可重复运行的 benchmark。

### 3.7 没有 Agent 决策能力

当前系统更像一条固定 RAG chain：

```text
question -> retrieve -> rerank -> answer
```

专业 Agent 应该能：

- 判断问题是否需要澄清。
- 判断是否需要查本地库还是实时 PubMed。
- 拆解复杂医学问题。
- 生成多个检索 query。
- 检查答案是否有足够证据。
- 在证据不足时扩大检索或拒答。
- 输出结构化答案和引用。

## 4. 重构目标

### 4.1 总体目标

将当前项目重构为一个专业的 Medical Literature RAG Agent：

```text
用户医学问题
  -> 问题理解
  -> 查询规划
  -> 本地知识库检索
  -> 可选 PubMed 实时检索
  -> 文档重排序
  -> 证据过滤
  -> 基于证据生成答案
  -> citation 验证
  -> 医学安全检查
  -> 结构化输出
```

### 4.2 工程目标

- 将 notebook 逻辑迁移为 Python package。
- 提供 CLI 入口。
- 提供可选 FastAPI 或 Streamlit/Gradio UI。
- 提供配置文件。
- 提供评估脚本。
- 提供单元测试和 smoke test。
- 保留 notebook 作为演示，而不是核心依赖。

### 4.3 研究目标

- 比较不同检索策略。
- 比较不同 chunk 参数。
- 比较不同 embedding 模型。
- 比较有无 reranker。
- 比较本地 LLM 和 API LLM。
- 用 RAGAS 和人工检查评估答案质量。

### 4.4 产品目标

用户输入一个医学问题后，系统应返回：

- 简洁答案。
- 支持该答案的文献证据。
- 引用来源。
- 置信度。
- 局限性。
- 医学安全提示。
- 可选的进一步阅读文章。

## 5. 推荐项目结构

建议重构后的目录结构如下：

```text
GR5293-AgentMedRag/
  README.md
  pyproject.toml
  .env.example
  configs/
    default.yaml
    local_mistral.yaml
    openai.yaml
    eval.yaml
  data/
    raw/
      pubmed_article.json
    processed/
      documents.jsonl
      chunks.jsonl
    qa/
      generated_qa_pairs.json
      generated_qa_pairs30.json
      generated_qa_pairs80.json
      generated_qa_pairs100.json
      generated_qa_pairs200.json
  indexes/
    chroma/
    faiss/
    colbert/
  docs/
    PROJECT_REFACTOR_PLAN.md
    ARCHITECTURE.md
    EVALUATION.md
    MEDICAL_SAFETY.md
  notebooks/
    legacy/
      generatequery.ipynb
      may15_rag_mainscript.ipynb
      Copy_of_intro2new.ipynb
    demos/
      quickstart_rag_demo.ipynb
      evaluation_demo.ipynb
  scripts/
    ingest_pubmed.py
    build_index.py
    ask.py
    evaluate.py
    generate_qa.py
  src/
    agent_medrag/
      __init__.py
      config.py
      schemas.py
      logging.py
      ingestion/
        __init__.py
        pubmed_client.py
        json_loader.py
        document_normalizer.py
      indexing/
        __init__.py
        chunker.py
        embeddings.py
        vector_store.py
        index_builder.py
      retrieval/
        __init__.py
        retriever.py
        hybrid_retriever.py
        query_rewriter.py
      reranking/
        __init__.py
        bge_reranker.py
        colbert_reranker.py
      generation/
        __init__.py
        llm_provider.py
        prompts.py
        answer_generator.py
      agents/
        __init__.py
        medrag_agent.py
        tools.py
        planner.py
        safety.py
        citation_verifier.py
      evaluation/
        __init__.py
        ragas_runner.py
        benchmark.py
        report.py
      api/
        __init__.py
        app.py
        routes.py
      cli.py
  tests/
    test_chunker.py
    test_loader.py
    test_retriever.py
    test_reranker.py
    test_answer_schema.py
    test_safety.py
```

## 6. 核心数据结构设计

### 6.1 PubMedArticle

用于表示原始 PubMed 文章。

```python
class PubMedArticle(BaseModel):
    article_id: str | None = None
    pmid: str | None = None
    title: str
    abstract: str
    pub_date: date | None = None
    journal: str | None = None
    authors: list[str] = []
    source: str = "pubmed"
```

当前数据没有 PMID，后续建议在 `download_pubmed.py` 中保留 PubMed ID。PMID 对 citation、去重、后续 PubMed 页面跳转都很重要。

### 6.2 MedicalDocument

用于表示规范化后的文档。

```python
class MedicalDocument(BaseModel):
    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any]
```

### 6.3 DocumentChunk

用于表示检索粒度的 chunk。

```python
class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str
    start_char: int | None = None
    end_char: int | None = None
    metadata: dict[str, Any]
```

### 6.4 RetrievedEvidence

用于表示检索或重排序后的证据。

```python
class RetrievedEvidence(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str
    retrieval_score: float | None = None
    rerank_score: float | None = None
    pub_date: str | None = None
    source: str = "local"
    metadata: dict[str, Any] = {}
```

### 6.5 MedicalAnswer

用于表示最终 Agent 输出。

```python
class MedicalAnswer(BaseModel):
    answer: str
    confidence: Literal["low", "medium", "high"]
    citations: list[Citation]
    limitations: str | None = None
    safety_note: str
    retrieval_summary: RetrievalSummary
    evaluation: AnswerEvaluation | None = None
```

### 6.6 Citation

```python
class Citation(BaseModel):
    title: str
    pub_date: str | None
    evidence: str
    doc_id: str
    chunk_id: str
    url: str | None = None
```

## 7. Agent 架构设计

### 7.1 推荐总架构

```text
User Question
  |
  v
MedicalQuestionAnalyzer
  |
  v
QueryPlanner
  |
  v
RetrievalOrchestrator
  |-------------------------|
  v                         v
LocalVectorRetriever     PubMedLiveSearchTool
  |                         |
  |-----------merge---------|
              v
          Reranker
              |
              v
        EvidenceFilter
              |
              v
     MedicalAnswerGenerator
              |
              v
       CitationVerifier
              |
              v
          SafetyGuard
              |
              v
      Structured Final Answer
```

### 7.2 为什么要从 chain 升级为 Agent

普通 RAG chain 适合简单问题，但医学问题常有以下复杂性：

- 问题可能过宽泛。
- 问题可能包含多个实体。
- 问题可能需要比较不同治疗方案。
- 问题可能需要最新文献。
- 问题可能有医疗风险。
- 检索结果可能互相矛盾。
- 文献证据可能不足。

Agent 的价值在于它不是机械执行单条链，而是可以动态选择工具和策略。

### 7.3 Agent 状态设计

建议维护一个 `AgentState`：

```python
class AgentState(BaseModel):
    user_question: str
    normalized_question: str | None = None
    question_type: str | None = None
    entities: list[str] = []
    planned_queries: list[str] = []
    retrieved_evidence: list[RetrievedEvidence] = []
    reranked_evidence: list[RetrievedEvidence] = []
    filtered_evidence: list[RetrievedEvidence] = []
    draft_answer: str | None = None
    citations: list[Citation] = []
    safety_flags: list[str] = []
    final_answer: MedicalAnswer | None = None
```

## 8. Agent 工具设计

### 8.1 AnalyzeMedicalQuestionTool

作用：

- 识别医学问题类型。
- 抽取疾病、药物、治疗、基因、症状、干预方式等实体。
- 判断是否涉及高风险医疗建议。

输入：

```json
{
  "question": "What are recent treatments for Alzheimer's disease?"
}
```

输出：

```json
{
  "question_type": "treatment_review",
  "entities": ["Alzheimer's disease"],
  "risk_level": "medium",
  "needs_current_literature": true
}
```

### 8.2 QueryPlannerTool

作用：

- 将用户问题转为多个检索 query。
- 支持 keyword query、semantic query、PubMed query。
- 对医学术语进行扩展。

示例：

用户问题：

```text
What are recent treatments for Alzheimer's disease?
```

可生成：

```json
[
  "Alzheimer's disease recent treatment",
  "Alzheimer disease therapeutic advances",
  "Alzheimer's disease drug therapy clinical trial",
  "Alzheimer disease monoclonal antibody treatment"
]
```

### 8.3 RetrieveLocalCorpusTool

作用：

- 从本地向量库检索文档。
- 支持 Chroma / FAISS / Qdrant / Milvus backend。

输入：

```json
{
  "queries": ["Alzheimer's disease recent treatment"],
  "top_k": 20
}
```

输出：

```json
{
  "documents": [
    {
      "chunk_id": "...",
      "title": "...",
      "text": "...",
      "retrieval_score": 0.82
    }
  ]
}
```

### 8.4 PubMedLiveSearchTool

作用：

- 当本地知识库不足或用户问“latest/recent/current”时，实时调用 PubMed。
- 支持日期过滤。
- 支持 PMID 返回。

注意：

- 这一步依赖网络和 Entrez API。
- 需要配置 `ENTREZ_EMAIL`。
- 应该设置 rate limit。

### 8.5 RerankEvidenceTool

作用：

- 使用 BGE reranker 或 ColBERT 对初检结果重排序。
- 去除同一标题的重复 chunk。
- 保留最相关的 top-n 证据。

当前 notebook 中已有 BGE reranker 雏形，可以迁移成独立类。

### 8.6 EvidenceFilterTool

作用：

- 过滤低质量证据。
- 去除空摘要、corrigendum、非医学主题、过短文本。
- 根据日期或标题去重。
- 根据 query intent 保留最相关证据。

这一步很重要。当前数据中存在 `CORRIGENDUM` 这类低价值记录，应该在 ingestion 或 retrieval 后过滤。

### 8.7 AnswerWithCitationsTool

作用：

- 使用 LLM 基于证据生成答案。
- 强制答案引用证据。
- 不允许无证据扩展。

Prompt 设计建议：

```text
You are a medical literature assistant.
Answer the user's question using only the provided evidence.
If evidence is insufficient, say so explicitly.
Do not provide personal diagnosis or treatment instructions.
For each key claim, cite the evidence id.

Question:
{question}

Evidence:
{evidence}

Return JSON with:
- answer
- citations
- limitations
- safety_note
- confidence
```

### 8.8 CitationVerifierTool

作用：

- 检查答案中的关键 claim 是否能在 citation 中找到支持。
- 如果某个 claim 没有证据，要求重写或降低 confidence。

可以先做简单规则版本：

- 答案必须至少有一个 citation。
- citation evidence 必须来自 retrieved chunks。
- citation title 不能为空。
- 如果答案包含多个关键结论，至少有多个 evidence 片段支持。

后续可以加入 LLM-based verifier。

### 8.9 MedicalSafetyGuardTool

作用：

- 检查答案是否有医疗风险。
- 对高风险问题加限制性提示。
- 防止模型输出诊断、剂量、紧急处理等不合适内容。

高风险类型包括：

- 个人症状诊断。
- 药物剂量。
- 儿童/孕妇/老年人治疗建议。
- 急症，如胸痛、中风、严重过敏、自杀意图。
- 停药、换药、手术选择。

输出策略：

- 对一般科研问题：正常回答并提示不是医疗建议。
- 对个人健康问题：建议咨询医疗专业人员。
- 对急症：建议立即联系急救服务。
- 对证据不足：明确说明无法从给定文献得出结论。

### 8.10 EvaluateAnswerTool

作用：

- 在开发或批量评估时运行。
- 对生成答案计算 RAGAS 指标。
- 可选地返回 faithfulness、answer relevancy、context precision、context recall。

## 9. 检索系统设计

### 9.1 Baseline 检索

第一版建议使用：

- Embedding model：`BAAI/bge-large-en-v1.5`
- Vector store：Chroma 或 FAISS
- Retriever：top-k semantic search
- Reranker：`BAAI/bge-reranker-base`

理由：

- 当前 notebook 已经使用这些组件。
- 迁移成本低。
- 检索质量通常比无 reranker 更稳定。

### 9.2 Hybrid Retrieval

后续可以加入 hybrid retrieval：

- Dense retrieval：embedding vector search。
- Sparse retrieval：BM25 keyword search。
- Metadata filtering：日期、标题、主题过滤。
- Reranking：BGE / ColBERT。

医学领域中，纯 dense retrieval 有时会漏掉精确术语，例如药名、基因名、缩写。BM25 可以补足精确词匹配。

### 9.3 Query Rewriting

建议实现 query rewriting：

- 原始用户问题。
- 医学术语扩展版本。
- 简短 keyword query。
- PubMed Boolean query。

例如：

```text
user question:
How does sonodynamic therapy differ from conventional antibiotics?

planned queries:
1. sonodynamic therapy conventional antibiotics multidrug resistant bacterial infections
2. SDT reactive oxygen species bacterial infection antibiotic resistance
3. antibacterial sonodynamic therapy mechanism drug resistance
```

### 9.4 Chunk 策略

当前教程使用：

- `chunk_size=128`
- `chunk_overlap=50`

当前项目 notebook 使用过：

- `chunk_size=200`
- `chunk_overlap=64`

建议将 chunk 参数配置化，并使用评估集比较：

| chunk_size | chunk_overlap | 预期特点 |
|---|---:|---|
| 128 | 50 | 检索粒度细，适合短答案，但上下文可能不完整 |
| 200 | 64 | 当前项目接近使用值，平衡粒度和上下文 |
| 256 | 64 | 更适合摘要级段落 |
| 512 | 128 | 上下文完整，但可能引入噪声 |

### 9.5 文档过滤策略

建议 ingestion 阶段过滤：

- 空摘要。
- 摘要长度过短。
- 标题为 `CORRIGENDUM` 或 `Correction` 的记录。
- 非医学主题，如果项目范围要限制为医学。
- 重复标题。

但过滤策略应可配置，因为有些 correction/corrigendum 在特定研究中可能仍有价值。

## 10. LLM 生成系统设计

### 10.1 LLM Provider 抽象

建议不要让系统绑定某一个模型。定义统一接口：

```python
class LLMProvider(Protocol):
    def generate(self, prompt: str, **kwargs) -> str:
        ...
```

支持：

- OpenAI provider
- HuggingFace local provider
- LangChain provider
- Mock provider for tests

### 10.2 推荐模型组合

开发阶段：

- Generator：OpenAI `gpt-4o` 或 `gpt-4.1-mini`
- Embedding：BGE local 或 OpenAI embeddings
- Reranker：BGE reranker

本地演示阶段：

- Generator：Mistral 7B / BioMistral / Llama 3
- Embedding：BGE
- Vector store：FAISS 或 Chroma

课程展示阶段：

- 展示 Mistral 7B baseline。
- 展示 RAG+Mistral 优于纯 Mistral。
- 展示 reranker 对结果的提升。
- 展示 RAGAS 指标变化。

### 10.3 Prompt 模板

推荐拆分多个 prompt：

- query planning prompt
- answer generation prompt
- citation verification prompt
- safety checking prompt
- summarization prompt

答案生成 prompt 应强调：

- 只能使用给定 context。
- 证据不足就说明不足。
- 不提供个人医疗建议。
- 每条关键结论必须有 citation。
- 输出 JSON，方便下游解析。

### 10.4 输出格式

推荐最终输出：

```json
{
  "answer": "Based on the retrieved PubMed abstracts, ...",
  "confidence": "medium",
  "citations": [
    {
      "title": "Antibacterial sonodynamic nanomedicine: mechanism, category, and applications.",
      "pub_date": "2025-03-25",
      "evidence": "Sonodynamic therapy leverages reactive oxygen species...",
      "doc_id": "pubmed_0008",
      "chunk_id": "pubmed_0008_chunk_0"
    }
  ],
  "limitations": "The answer is based on abstracts only, not full-text clinical guidelines.",
  "safety_note": "This is for literature review only and is not medical advice."
}
```

## 11. 医学安全设计

### 11.1 系统定位

Agent 应被定义为：

> A medical literature assistant for research and educational use.

不应定义为：

> A doctor, clinician, diagnostic assistant, or treatment decision system.

### 11.2 必须避免的输出

系统不应：

- 诊断用户疾病。
- 推荐个人治疗方案。
- 给出药物剂量。
- 建议停止或更换药物。
- 对急症给延迟处理建议。
- 将单篇论文结论包装成临床共识。

### 11.3 风险分级

建议将问题分为：

- `low`
  - 一般医学机制问题。
  - 文献综述问题。
  - “What does the paper say about...”。

- `medium`
  - 治疗选择比较。
  - 药物副作用。
  - 疾病管理。

- `high`
  - 个人症状诊断。
  - 急症。
  - 药物剂量。
  - 儿童/孕妇治疗。
  - 自杀或自残。

### 11.4 安全响应模板

低风险：

```text
This answer summarizes the retrieved medical literature and is not medical advice.
```

中风险：

```text
Treatment decisions depend on individual clinical context. This answer summarizes literature only and should be discussed with a qualified clinician.
```

高风险：

```text
I cannot provide diagnosis or personalized treatment instructions. If this may be urgent, contact emergency medical services or a qualified healthcare professional immediately.
```

### 11.5 证据不足处理

如果检索证据不足：

```text
The retrieved abstracts do not provide enough evidence to answer this question reliably.
```

如果证据互相矛盾：

```text
The retrieved evidence appears mixed. Some sources suggest ..., while others indicate .... More targeted review is needed.
```

## 12. 评估体系设计

### 12.1 自动评估指标

建议保留并标准化 RAGAS：

- `faithfulness`
  - 答案是否忠实于检索上下文。

- `answer_relevancy`
  - 答案是否回应问题。

- `context_precision`
  - 检索到的上下文是否相关。

- `context_recall`
  - 检索是否覆盖参考答案需要的信息。

也可加入：

- latency
- token usage
- retrieval hit rate
- citation coverage
- citation precision

### 12.2 自定义医学指标

建议加入项目自己的指标：

#### Citation Coverage

答案中的关键 claim 有多少被 citation 覆盖。

#### Unsupported Claim Count

答案中有多少句子无法被 evidence 支持。

#### Refusal Accuracy

在证据不足或高风险问题上，系统是否正确拒答或加限制。

#### Source Diversity

答案是否依赖多个不同文献，而不是重复 chunk。

#### Recency Awareness

对“recent/latest/current”问题，是否优先检索较新文献。

### 12.3 评估输入格式

建议统一 QA 数据格式：

```json
{
  "id": "qa_001",
  "question": "...",
  "reference_answer": "...",
  "expected_titles": ["..."],
  "category": "treatment",
  "risk_level": "low"
}
```

当前 `generated_qa_pairs*.json` 可以迁移到这个格式。

### 12.4 评估报告格式

建议每次评估输出：

```text
reports/
  eval_2026-05-13_2300/
    config.yaml
    metrics.json
    per_question_results.jsonl
    summary.md
```

`summary.md` 应包含：

- 使用的 embedding model。
- 使用的 LLM。
- chunk 参数。
- retrieval k。
- rerank top-k。
- 平均 faithfulness。
- 平均 answer relevancy。
- 平均 context precision。
- 平均 context recall。
- 平均 latency。
- 失败案例分析。

### 12.5 消融实验

建议至少做以下对比：

1. 纯 LLM vs RAG+LLM。
2. 无 reranker vs BGE reranker。
3. Chroma vs FAISS。
4. chunk size 128 vs 200 vs 512。
5. Mistral 7B vs OpenAI model。
6. top-k 5 vs 15 vs 30。

## 13. CLI 设计

建议提供统一命令：

```bash
medrag ingest --input data/raw/pubmed_article.json
medrag build-index --config configs/default.yaml
medrag ask "What are recent treatments for Alzheimer's disease?"
medrag evaluate --qa data/qa/generated_qa_pairs100.json
medrag serve --host 127.0.0.1 --port 8000
```

### 13.1 `medrag ingest`

作用：

- 加载原始 PubMed JSON。
- 标准化字段。
- 过滤无效文档。
- 输出 processed documents。

### 13.2 `medrag build-index`

作用：

- 加载 processed documents。
- chunk 文本。
- 生成 embedding。
- 构建向量库。
- 保存 index。

### 13.3 `medrag ask`

作用：

- 对单个问题运行完整 Agent。
- 输出结构化答案。

示例：

```bash
medrag ask "How does sonodynamic therapy differ from conventional antibiotics?"
```

### 13.4 `medrag evaluate`

作用：

- 批量运行 QA。
- 生成评估报告。

### 13.5 `medrag serve`

作用：

- 启动 FastAPI 服务。
- 提供 `/ask`、`/health`、`/evaluate` 等接口。

## 14. API 设计

### 14.1 POST `/ask`

请求：

```json
{
  "question": "How does sonodynamic therapy differ from conventional antibiotics?",
  "options": {
    "retrieval_k": 15,
    "rerank_top_k": 4,
    "include_debug": true
  }
}
```

响应：

```json
{
  "answer": "...",
  "confidence": "high",
  "citations": [...],
  "limitations": "...",
  "safety_note": "...",
  "debug": {
    "planned_queries": [...],
    "retrieved_count": 15,
    "reranked_count": 4,
    "latency_ms": 6421
  }
}
```

### 14.2 GET `/health`

返回：

```json
{
  "status": "ok",
  "index_loaded": true,
  "llm_provider": "openai",
  "vector_store": "chroma"
}
```

### 14.3 POST `/evaluate`

用于触发小规模评估。

生产环境中不建议开放给任意用户，只用于开发或内部 demo。

## 15. UI 设计

如果要做展示，建议用 Streamlit 或 Gradio。

### 15.1 页面布局

左侧：

- 模型选择。
- retrieval k。
- rerank top-k。
- 是否启用 PubMed live search。
- 是否显示 debug。

主区域：

- 问题输入框。
- 答案区域。
- citation cards。
- retrieved contexts。
- 评估分数。

### 15.2 Citation Card

每个 citation 展示：

- 文章标题。
- 出版日期。
- 证据片段。
- 检索分数。
- rerank 分数。
- PubMed 链接，如果有 PMID。

### 15.3 Debug View

展示：

- Query planner 生成的 queries。
- 初始检索结果。
- rerank 后结果。
- safety flags。
- 生成耗时。

## 16. 配置系统设计

建议 `configs/default.yaml`：

```yaml
data:
  raw_pubmed_path: data/raw/pubmed_article.json
  processed_docs_path: data/processed/documents.jsonl
  chunks_path: data/processed/chunks.jsonl

chunking:
  strategy: token
  chunk_size: 200
  chunk_overlap: 64

embedding:
  provider: huggingface
  model_name: BAAI/bge-large-en-v1.5
  device: auto
  normalize_embeddings: true

vector_store:
  provider: chroma
  persist_directory: indexes/chroma

retrieval:
  top_k: 15
  score_threshold: null

reranking:
  enabled: true
  provider: bge
  model_name: BAAI/bge-reranker-base
  top_k: 4

llm:
  provider: openai
  model: gpt-4o
  temperature: 0.1
  max_tokens: 512

agent:
  enable_query_planning: true
  enable_pubmed_live_search: false
  enable_citation_verification: true
  enable_safety_guard: true

evaluation:
  metrics:
    - faithfulness
    - answer_relevancy
    - context_precision
    - context_recall
```

`.env.example`：

```text
OPENAI_API_KEY=
HF_TOKEN=
ENTREZ_EMAIL=
```

## 17. 迁移计划

### 阶段 1：整理项目结构

目标：

- 不改变核心行为。
- 只把 notebook 中稳定逻辑迁移到 Python module。

任务：

1. 新建 `src/agent_medrag/`。
2. 新建 `configs/default.yaml`。
3. 新建 `data/`、`indexes/`、`notebooks/legacy/`。
4. 将当前 notebook 移入 `notebooks/legacy/`，或保留原位置并在 README 标注 legacy。
5. 将 `download_pubmed.py` 逻辑迁入 `ingestion/pubmed_client.py`。

产出：

- 项目结构清晰。
- 原始 notebook 仍可作为参考。

### 阶段 2：实现 ingestion 和 indexing

目标：

- 能从 JSON 构建本地向量库。

任务：

1. 实现 `json_loader.py`。
2. 实现 `document_normalizer.py`。
3. 实现 `chunker.py`。
4. 实现 `embeddings.py`。
5. 实现 `vector_store.py`。
6. 实现 `scripts/build_index.py` 或 CLI `medrag build-index`。

验收：

```bash
medrag ingest --input data/raw/pubmed_article.json
medrag build-index
```

能够成功构建 index。

### 阶段 3：实现基础 RAG

目标：

- 用模块化代码复现当前 notebook RAG 功能。

任务：

1. 实现 `retriever.py`。
2. 实现 `bge_reranker.py`。
3. 实现 `llm_provider.py`。
4. 实现 `answer_generator.py`。
5. 实现 `prompts.py`。
6. 实现 `medrag ask`。

验收：

```bash
medrag ask "How does sonodynamic therapy differ from conventional antibiotics?"
```

输出答案和引用来源。

### 阶段 4：实现 Agent 工具化

目标：

- 从固定 chain 升级为工具化 Agent。

任务：

1. 实现 `AnalyzeMedicalQuestionTool`。
2. 实现 `QueryPlannerTool`。
3. 实现 `RetrieveLocalCorpusTool`。
4. 实现 `RerankEvidenceTool`。
5. 实现 `AnswerWithCitationsTool`。
6. 实现 `CitationVerifierTool`。
7. 实现 `MedicalSafetyGuardTool`。
8. 实现 `MedRAGAgent` orchestration。

验收：

- 对普通问题能检索并回答。
- 对证据不足问题能说明不足。
- 对个人医疗建议问题能加安全边界。

### 阶段 5：实现评估系统

目标：

- 将 RAGAS 从 notebook 迁移为可重复运行的评估脚本。

任务：

1. 标准化 QA 数据格式。
2. 实现 `ragas_runner.py`。
3. 实现 `benchmark.py`。
4. 实现 `report.py`。
5. 实现 `medrag evaluate`。

验收：

```bash
medrag evaluate --qa data/qa/generated_qa_pairs100.json
```

生成：

- metrics JSON
- per-question JSONL
- summary markdown

### 阶段 6：实现 API/UI

目标：

- 项目可展示、可交互。

任务：

1. FastAPI `/ask` endpoint。
2. FastAPI `/health` endpoint。
3. Streamlit 或 Gradio UI。
4. 展示 citations、contexts、scores。

验收：

```bash
medrag serve
```

或：

```bash
streamlit run app.py
```

用户可在浏览器中提问。

## 18. 测试策略

### 18.1 单元测试

建议测试：

- JSON loader 是否正确加载文章。
- metadata 是否正确提取。
- chunker 是否产生非空 chunk。
- retriever 是否返回 expected schema。
- reranker 是否保留 top-k。
- answer schema 是否可解析。
- safety guard 是否识别高风险问题。

### 18.2 Smoke Test

一个最小端到端测试：

1. 加载 10 篇文章。
2. 构建小 index。
3. 提问一个已知答案问题。
4. 检查是否返回：
   - answer
   - citations
   - safety_note

### 18.3 Regression Test

使用固定 QA 文件：

- `generated_qa_pairs30.json`
- `generated_qa_pairs100.json`

每次重构后运行：

- 平均 faithfulness 不应明显下降。
- 平均 answer relevancy 不应明显下降。
- citation coverage 不应下降。

## 19. README 重写建议

重构后 README 应包含：

1. 项目一句话介绍。
2. 系统架构图。
3. Quickstart。
4. 数据准备。
5. 构建索引。
6. 命令行问答。
7. API/UI 启动。
8. 评估方法。
9. 医学安全声明。
10. 项目结构。
11. 与教程 baseline 的区别。

README 开头建议：

```markdown
# AgentMedRAG

AgentMedRAG is a medical literature question-answering agent built on PubMed abstracts, retrieval-augmented generation, reranking, citation verification, and RAG evaluation.

It is intended for medical literature exploration and educational use, not for diagnosis or personalized medical advice.
```

## 20. 与教程 baseline 的升级点

教程 baseline：

```text
PubMed JSON -> JSONLoader -> chunk -> FAISS -> Mistral 7B -> RetrievalQA
```

重构后的 AgentMedRAG：

```text
PubMed JSON / live PubMed
  -> document normalization
  -> configurable chunking
  -> embedding index
  -> hybrid retrieval
  -> BGE/ColBERT reranking
  -> evidence filtering
  -> query planning
  -> medical answer generation
  -> citation verification
  -> safety guard
  -> structured answer
  -> RAGAS evaluation
```

关键升级：

- 从 notebook 到 package。
- 从单条 RetrievalQA chain 到 Agent。
- 从无结构答案到结构化 citation answer。
- 从手工测试到自动评估。
- 从固定路径到配置化。
- 从 demo 到可部署服务。
- 从普通 RAG 到带医学安全边界的 RAG。

## 21. 推荐优先级

如果时间有限，建议按以下优先级做：

1. 模块化基础 RAG。
2. 标准 citation 输出。
3. BGE reranker 封装。
4. RAGAS 评估脚本。
5. 医学安全 guard。
6. Query planner。
7. FastAPI/Streamlit。
8. PubMed live search。
9. Hybrid retrieval。
10. ColBERT 深度集成。

最小可交付版本应该包含：

- `medrag ingest`
- `medrag build-index`
- `medrag ask`
- `medrag evaluate`
- 结构化 citation answer
- 医学安全声明

## 22. 建议的 MVP 定义

MVP 不需要一开始就实现完整 Agent 图，但必须具备专业系统的基本形态。

### MVP 功能

- 从 `pubmed_article.json` 加载数据。
- 构建 Chroma 或 FAISS index。
- 用 BGE embedding 检索。
- 用 BGE reranker 重排。
- 用 OpenAI 或 Mistral 生成答案。
- 输出 citation。
- 对答案加 safety note。
- 用 QA 文件运行 RAGAS 评估。

### MVP 不必包含

- 多轮对话 memory。
- 实时 PubMed 搜索。
- LangGraph 复杂状态机。
- 完整 Web UI。
- ColBERT 生产集成。
- 多向量库切换。

这些可以放到第二阶段。

## 23. 后续可扩展方向

### 23.1 Full-text 支持

当前项目主要使用摘要。后续可以加入：

- PubMed Central full text。
- PDF parsing。
- section-aware chunking。
- abstract vs methods vs results vs conclusion 权重。

### 23.2 临床指南支持

可以加入：

- CDC
- WHO
- NIH
- NICE
- professional guidelines

并在输出中区分：

- research article
- review
- clinical guideline
- case report

### 23.3 Evidence Level

给文献证据分级：

- clinical guideline
- systematic review
- randomized controlled trial
- cohort study
- case report
- narrative review
- animal/in vitro study

### 23.4 Multi-agent 设计

后续可以拆成多个角色：

- Retriever Agent
- Medical Evidence Reviewer
- Answer Writer
- Citation Auditor
- Safety Reviewer

但第一版不建议过早复杂化。

### 23.5 Knowledge Graph

可以构建实体关系：

- disease -> treatment
- drug -> side effect
- gene -> disease
- intervention -> outcome

再结合 RAG 提升复杂问题回答能力。

## 24. 风险与注意事项

### 24.1 医学准确性风险

RAG 只能提高 groundedness，不能保证医学正确性。

缓解：

- citation verification。
- safety guard。
- 明确非医疗建议。
- 使用高质量来源。
- 对证据不足拒答。

### 24.2 数据质量风险

PubMed 搜索结果可能包含非医学、纠错、过短摘要或不相关文献。

缓解：

- ingestion filter。
- metadata 清理。
- dedup。
- source quality scoring。

### 24.3 模型幻觉风险

LLM 可能编造不存在的细节。

缓解：

- 只允许基于 context。
- JSON schema 输出。
- citation verifier。
- faithfulness evaluation。

### 24.4 复现风险

Colab、HuggingFace、OpenAI、GPU 环境不同会导致结果不一致。

缓解：

- 固定依赖版本。
- 记录 config。
- 保存 evaluation report。
- 提供 CPU fallback。

### 24.5 成本风险

OpenAI 和 reranker 批量评估可能产生成本。

缓解：

- 小规模 smoke eval。
- 缓存 LLM 输出。
- 支持 local model。
- 分离 dev config 和 full eval config。

## 25. 结论

当前项目已经具备 Medical RAG 的核心能力，但主要问题是工程化不足、Agent 决策能力不足、输出可信度机制不足、评估流程不够标准化。

推荐重构方向是：

1. 先把 notebook 逻辑模块化。
2. 再将固定 RAG chain 升级为工具化 Agent。
3. 强制结构化 citation 输出。
4. 加入医学安全 guard。
5. 将 RAGAS 评估产品化。
6. 最后提供 CLI/API/UI，形成完整项目展示。

最终项目应当定位为：

> AgentMedRAG: a citation-grounded medical literature assistant built with PubMed, retrieval-augmented generation, reranking, medical safety guardrails, and reproducible RAG evaluation.

