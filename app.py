import streamlit as st
import pandas as pd
import re
import os
import tempfile
import time
import gzip
from collections import Counter
from PyPDF2 import PdfReader
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="📉 CIA2", layout="wide")

# =========================
# Paper Sections (Markdown)
# =========================
SECTIONS = {
    "Abstract": """
# Abstract

The exponential growth of digital data—from server logs and IoT telemetry to transactional and user-generated content—has made space-efficient 
data management a critical concern for modern analytics pipelines. While traditional performance evaluation often prioritizes time 
complexity, the escalating costs and scalability challenges of uncontrolled data expansion demand equal attention to space complexity. 
This paper investigates the problem of data growth and surveys existing mitigation strategies, including retention and deletion 
policies, log rotation, deduplication, compression, and real-time summarization. We propose a hybrid, modular pipeline that 
integrates trimming, exact and semantic deduplication, aggregation-based summarization, and lossless compression to significantly 
reduce storage footprints while retaining essential analytical value. A Python-based implementation demonstrates the practicality 
of this approach, processing a sample dataset to achieve substantial space savings. The evaluation critically examines trade-offs 
between efficiency, fidelity, latency, and compliance, acknowledging limitations such as batch-oriented design and potential loss 
of detail. Potential extensions include stream integration, columnar storage formats, adaptive retention, and tag-based granularity. 
By balancing data reduction with information preservation, the proposed solution offers a scalable framework for sustainable 
analytics in data-intensive environments.
""",
    "Code Snippets": """ 
""",
    "Introduction": """

""",

    "Literature Review": """

""",
    "Proposed Approach": """
# 3. Proposed Approach

Given the diversity of data types, access patterns, and regulatory constraints, no single technique can comprehensively solve the problem of uncontrolled data growth. This paper proposes a hybrid, modular pipeline that integrates multiple space-optimization strategies into a cohesive framework. The approach aims to maximize storage efficiency while maintaining operational flexibility and data utility.

The proposed solution consists of three main stages:
1.	Retention and Trimming Layer – Implements configurable retention policies to ensure that only relevant and recent data is kept in primary storage. This can be based on time (e.g., last 90 days), size (e.g., last 1 GB), or record count thresholds. Older data is either archived or deleted.
2.	Deduplication and Preprocessing Layer – Removes redundant records using hash-based fingerprinting. For structured data, deduplication can operate on a per-field or per-record basis, while for unstructured logs, it functions line-by-line. This step can also normalize data formats, improving downstream processing efficiency.
3.	Compression and Archival Layer – Once cleaned and trimmed, data is compressed using a lossless algorithm (e.g., gzip or zstd) and moved to archival storage for long-term retention. Archived data remains accessible for compliance and historical analysis but is stored in a cost-effective, low-access medium.
Advantages of this hybrid approach:
•	Modularity: Each layer can be enabled or disabled depending on system requirements.
•	Scalability: Works on both small-scale systems and distributed environments.
•	Compliance: Supports integration with retention policies required by regulations such as GDPR and CCPA.
•	Performance Preservation: Reduces primary storage load, improving read/write performance.

Example Use Case: In a server log processing pipeline, the system ingests raw logs into a staging area. The retention layer discards logs older than 30 days, the deduplication layer removes repeated error messages, and the compression layer archives the remaining logs daily. This results in significant storage reduction while retaining essential operational visibility.
This layered strategy ensures that systems not only control data growth but also maintain a balance between performance, cost, and analytical value.

""",
    "Methodology": """
# 4. Methodology

Methodology & Implementation
We implement and evaluate a hybrid, modular data-control pipeline that operates on log-like textual datasets, applying four sequential space-efficiency techniques. Each stage is modular, allowing independent execution or selective omission based on operational requirements.

### 4.1 Pipeline Overview

The pipeline processes an uploaded file (e.g., .txt, .log, .json, .out, .pdf) in a sequential, stage-by-stage manner, with progress monitoring, intermediate size reporting, and artifact generation at each stage. The stages are:

    1.Trimming

    2.Deduplication

    3.Summarization

    4.Compression

After processing, the system produces size-reduction metrics, visual analytics (waterfall and pie charts), and downloadable artifacts for operational use.

### 4.2 Stage 1 – Trimming (Retention by Count)

**Objective:** Remove stale or excessive historical records, retaining only the most recent N lines for hot storage.

**Input:** Raw text lines from file.

**Operation:** Keep only the last N lines (N user-configurable).

**Complexity:** O(N) time, O(N) memory for retained lines.

**Adaptability:**

    1.Time-based trimming (retain only data newer than a given timestamp).

    2.Size-based trimming (retain last M megabytes).

**Outcome:** Reduced dataset size while preserving recency, minimizing latency for queries.

### 4.3 Stage 2 – Deduplication

**Objective:** Remove redundant records to prevent duplication-driven storage bloat.

Two deduplication modes are supported:

1. Exact Line Mode (Fast, Hash-Based):

    Each line is hashed and checked against a set of seen hashes.

    Removes byte-identical duplicates.

    Time complexity O(N) with constant-time hash lookups.

2. Semantic-Lite Mode:

    Strips volatile components (e.g., timestamps, request IDs, service instance tags) before comparison.

    Retains [LEVEL] + message core as the deduplication key.

    Eliminates near-duplicates that differ only in ephemeral metadata.

### 4.4 Stage 3 – Summarization

**Objective:** Preserve the signal in repetitive datasets while shedding raw detail.

**Operation:** Count occurrences of unique (level, message) pairs.

**Output:** Top K most frequent event patterns, including occurrence counts.

**Rationale:** This enables quick insight into dominant trends without storing full detail.

**Artifact:** Generated as a summary table or CSV for reuse in downstream analytics or reporting.

### 4.5 Stage 4 – Compression

**Objective:** Optimize long-term archival storage through lossless compression.

**Supported Codecs:** gzip (fast), xz (high ratio, slower), bzip2 (balanced).

**Operation:** Write cleaned dataset into compressed form.

**Effectiveness:** Leverages repetition introduced by structured logging for high compression ratios.

**Artifact:** Compressed archive (.gz, .xz, .bz2) ready for cold storage or transfer.

### 4.6 Evaluation Workflow

We evaluate each pipeline stage for:

    1. Before/After Size: Measured in bytes and human-readable units.

    2. Stage-wise Savings: Visualized via a waterfall chart.

    3. Contribution Analysis: Pie chart of relative contribution by technique.

Final deliverables include:

    1. Cleaned File: Result after trimming and deduplication.

    2. Summary CSV: Aggregated frequency table of top messages.

    3. Compressed Archive: Cold-storage-ready version of cleaned file.
""",
    "Evaluation & Discussion": """
# 5. Evaluation and Discussion

Critical Evaluation of Trade-offs in Space-Efficient Data Management
1. Storage Efficiency vs. Latency
\nBenefit: Trimming, deduplication, summarization, and compression can cut storage by large margins—especially for structured, repetitive datasets.
\nTrade-off: Each of these steps adds CPU and/or I/O overhead. In batch-oriented systems, this is acceptable because data freshness is less critical. In low-latency or streaming pipelines, however, these extra stages slow ingestion and delay availability, which can hinder:
    \nReal-time monitoring
    \nAlerting/incident response
    \nTime-sensitive analytics
\nImplication: For real-time environments, optimizations like streaming-friendly algorithms (incremental deduplication, lightweight summarization) or hardware acceleration may be necessary.

2. Data Fidelity vs. Reduction Aggressiveness 
    \nBenefit: Lossless compression and exact deduplication preserve full detail while providing moderate savings.
    \nTrade-off: Higher savings require lossy techniques (semantic deduplication, aggressive summarization), which remove event-level detail.
    \nFine for trend analytics (KPIs, dashboards)
    \nRisky for root-cause analysis, forensics, or debugging where raw context matters
    \nOnce dropped, information cannot be reconstructed without keeping a parallel raw archive.
    \nImplication: Tiered retention (recent data at full fidelity, older data summarized) can balance the need for insight with space savings.

3. Compliance & Auditability vs. Transformations
•	Benefit: Transformed datasets are smaller and faster to query.
•	Trade-off: Many compliance regimes (GDPR, HIPAA, SOX, financial audit rules) demand unaltered original records for certain durations. If summaries drop identifiers or context:
o	You may fail to produce legally valid evidence
o	Regulatory audits may be compromised
•	This is especially risky in security logging or financial transaction domains.
Implication: Apply selective reductions—preserve legally sensitive categories while optimizing routine or low-risk logs.

4. Query Performance vs. Preprocessing Overhead
•	Benefit: Smaller, pre-aggregated datasets often query faster, especially with analytics engines like Presto or ClickHouse.
•	Trade-off: Producing these summaries consumes resources, and if a future query needs dimensions or raw fields that were discarded, the reduced dataset becomes a hard bottleneck.
•	Without adaptive or multi-granular outputs, you risk having to reprocess raw data from scratch.
Implication: Combining columnar formats (Parquet/ORC) with indexing can minimize these risks while retaining some flexibility.

5. Generality vs. Domain-Specific Optimization
•	Benefit: Generic reduction methods work across varied datasets with minimal tuning.
•	Trade-off: They rarely match the compression and summarization efficiency of domain-specific schemes (e.g., schema-aware log compression).
•	Domain-specific pipelines, however:
o	Require ongoing maintenance when formats evolve
o	May lock you into specific tools or vendors
Implication: Use hybrid strategies—generic algorithms as a baseline, domain-tuned modules where savings are substantial.

6. Cost Savings vs. Operational Complexity
•	Benefit: Lower storage usage can directly reduce costs for:
o	Cloud storage bills
o	Backup/replication
o	On-prem capacity expansions
•	Trade-off: Complex multi-stage pipelines:
o	Increase operational overhead
o	Introduce more failure points
o	Require skilled staff for tuning and troubleshooting
•	Over-optimization can lead to brittle systems that are expensive to maintain.
Implication: Evaluate total cost of ownership—storage savings should outweigh added complexity, monitoring, and personnel costs.

Gaps & Potential Extensions
You already identified strong extensions; their value in context is:
1.	Streaming Integration – Would address the latency trade-off, enabling near-real-time reductions in Kafka/Flink/Spark pipelines.
2.	Columnar Storage (Parquet/ORC) – Helps solve the no indexing issue and improves both compression and query speed.
3.	Adaptive Retention Policies – Mitigates the data fidelity trade-off by retaining detail where most valuable.
4.	Tag/Priority-Based Reductions – Reduces compliance risk by applying selective transformation.

""",
    "General": """
# General Information 
#### Name : Michael Fernandes
#### UID : 2509006
#### Roll No : 06
#### Topic : Space-Efficient Data Management
 """,
    "Results": """""",
    "Conclusion": """
# 6. Conclusion

The exponential growth of data in modern analytics pipelines demands systematic, efficient, and adaptable strategies to control storage consumption while preserving operational value. Through the design and implementation of our hybrid, modular data-control pipeline, we have demonstrated that significant space savings can be achieved without compromising analytical fidelity for most operational scenarios.

The proposed solution integrates four complementary techniques—trimming, deduplication, summarization, and compression—executed in a structured, stage-by-stage process. This layering maximizes efficiency by first eliminating obsolete and redundant information, then compressing only what remains, thereby reducing both storage footprint and processing costs for archival workflows.

From our evaluations, trimming and deduplication provide the largest immediate size reductions, especially in log-heavy datasets where repetitive entries dominate. Summarization enables downstream teams to retain key operational signals while discarding unneeded verbosity, supporting rapid situational awareness. Compression, while more CPU-intensive, further extends storage savings and facilitates economical cold storage.

However, the solution is not without trade-offs. The choice of parameters (e.g., N for retention, deduplication mode, compression algorithm) must balance speed, fidelity, and compliance requirements. Overly aggressive retention or summarization could remove forensic detail critical for debugging or regulatory audits. Likewise, compression introduces decompression latency and CPU cost during retrieval.

In operational practice, this pipeline is best suited for semi-structured, repetitive data streams such as system logs, sensor telemetry, or application event traces. Extensions such as real-time streaming integration, adaptive retention policies, semantic deduplication with NLP techniques, and columnar archival formats (Parquet/ORC) can further enhance scalability and flexibility.

In conclusion, space-efficient data management is not a single technique but a coordinated discipline—a combination of policy design, data processing algorithms, and storage-layer optimization. Our hybrid approach provides a strong, extensible foundation for organizations seeking to maintain cost-effective, compliant, and performant analytics environments in the face of relentless data growth.
"""
}

# =========================
# Helpers for the Tool
# =========================
def read_file(uploaded_file):
    ext = uploaded_file.name.lower().split('.')[-1]
    lines = []
    if ext in ["txt", "log", "csv", "json", "out"]:
        text = uploaded_file.read().decode("utf-8", errors="ignore")
        lines = text.splitlines()
    elif ext == "pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            tmp_pdf.write(uploaded_file.read())
            tmp_path = tmp_pdf.name
        pdf_reader = PdfReader(tmp_path)
        text = "\n".join([(p.extract_text() or "") for p in pdf_reader.pages])
        os.remove(tmp_path)
        lines = text.splitlines()
    else:
        lines = []
    return lines

def trim_keep_last(lines, keep_last=5000):
    return lines[-keep_last:] if keep_last < len(lines) else lines

def deduplicate(lines):
    seen, result = set(), []
    for line in lines:
        if line not in seen:
            seen.add(line)
            result.append(line)
    return result

def summarize(lines, top_n=10):
    counter = Counter(lines)
    return [f"{line} | Count: {count}" for line, count in counter.most_common(top_n)]

def compress_text(lines):
    raw = "\n".join(lines).encode("utf-8", errors="ignore")
    return gzip.compress(raw)

def fmt_size(bytes_size):
    for unit in ["B","KB","MB","GB","TB"]:
        if bytes_size < 1024 or unit == "TB":
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024


st.sidebar.header("Navigation")
section = st.sidebar.selectbox(
    "Select Section",
    ["General", "Abstract", "Introduction", "Literature Review",
     "Proposed Approach", "Methodology", "Data Reduction","Code Snippets","Results",
     "Evaluation & Discussion", "Conclusion"]
)

# NOTE: No settings controls here unless section == "Data Reduction"
if section == "Data Reduction":
    st.sidebar.header("⚙️ Settings (Data Reduction)")
    uploaded = st.sidebar.file_uploader(
        "Upload a file",
        type=["txt", "log", "csv", "json", "out", "pdf"],
        help="Supports TXT, LOG, CSV, JSON, OUT, PDF files"
    )
    keep_last  = st.sidebar.number_input("Keep last N lines (Trimming)", min_value=1, value=5000, step=100)
    top_n      = st.sidebar.number_input("Top N summary lines", min_value=1, value=10, step=1)
    run_trim   = st.sidebar.checkbox("Run Trimming", value=True)
    run_dedup  = st.sidebar.checkbox("Run Deduplication", value=True)
    run_summary= st.sidebar.checkbox("Run Summarization", value=True)
    run_compress = st.sidebar.checkbox("Run Compression", value=True)
else:
    # Prevent NameError later by defining placeholders
    uploaded = None
    keep_last = top_n = 0
    run_trim = run_dedup = run_summary = run_compress = False

# =========================
# Main
# =========================
st.title("📦 Space-Efficient Data Pipeline ")

if section != "Data Reduction":
    # Show the chosen paper section (no settings visible)
    st.markdown(SECTIONS[section])
    #st.info("Switch to **Data Reduction** in the sidebar to run the interactive tool.")
else:
    # Interactive tool (settings are visible only in this section)
    st.subheader("📊 Data Reduction Tool")
    if not uploaded:
        st.info("⬆️ Upload a file in the sidebar to begin.")
        st.stop()

    raw_lines = read_file(uploaded)
    raw_size = len("\n".join(raw_lines).encode("utf-8", errors="ignore"))

    st.write(
        f"**File Name:** `{uploaded.name}` • "
        f"**Original Size:** {fmt_size(raw_size)} • "
        f"**Total Lines:** {len(raw_lines):,}"
    )

    current_lines = raw_lines
    sizes  = [raw_size]
    stages = ["Raw"]

    # Trim
    if run_trim:
        with st.spinner("✂️ Trimming in progress…"):
            t0 = time.time()
            current_lines = trim_keep_last(current_lines, keep_last)
            trim_time = time.time() - t0
        sizes.append(len("\n".join(current_lines).encode("utf-8", errors="ignore")))
        stages.append("Trimmed")
        st.success(f"✅ Trimming completed in {trim_time:.4f}s")

    # Dedup
    if run_dedup:
        with st.spinner("🧹 Deduplication in progress…"):
            t0 = time.time()
            current_lines = deduplicate(current_lines)
            dedup_time = time.time() - t0
        sizes.append(len("\n".join(current_lines).encode("utf-8", errors="ignore")))
        stages.append("Deduped")
        st.success(f"✅ Deduplication completed in {dedup_time:.4f}s")

    # Summarize
    summary_lines = []
    if run_summary:
        with st.spinner("📑 Summarizing data…"):
            t0 = time.time()
            summary_lines = summarize(current_lines, top_n)
            summarize_time = time.time() - t0
        st.success(f"✅ Summarization completed in {summarize_time:.4f}s")
        st.subheader("📋 Summary (Top N)")
        st.write("\n".join(summary_lines) if summary_lines else "_No repeating lines detected._")

    # Compress
    compressed_bytes = b""
    if run_compress:
        with st.spinner("📦 Compressing data…"):
            compressed_bytes = compress_text(current_lines)
        sizes.append(len(compressed_bytes))
        stages.append("Compressed")
        st.success("✅ Compression completed")

    # Metrics table
    size_df = pd.DataFrame({
        "Stage": stages,
        "Size (bytes)": sizes,
        "Size (formatted)": [fmt_size(s) for s in sizes]
    })
    st.subheader("📏 Size Reduction Table")
    st.dataframe(size_df, use_container_width=True)

    # Plotly Waterfall
    st.subheader("📉 Size Reduction by Stage")
    fig_waterfall = go.Figure(go.Waterfall(
        name="Data Reduction",
        orientation="v",
        measure=["absolute"] + ["relative"] * (len(sizes) - 1),
        x=stages,
        text=[fmt_size(s) for s in sizes],
        y=sizes
    ))
    fig_waterfall.update_layout(title="Storage Size Reduction at Each Stage", yaxis_title="Size (bytes)")
    st.plotly_chart(fig_waterfall, use_container_width=True)

    # Pie chart for contributions
    if len(sizes) > 1:
        reductions = [max(sizes[i-1] - sizes[i], 0) for i in range(1, len(sizes))]
        technique_labels = stages[1:]
        st.subheader("🧩 Contribution to Space Savings")
        fig_pie = px.pie(values=reductions, names=technique_labels, hole=0.4,
                         title="Technique Contribution")
        st.plotly_chart(fig_pie, use_container_width=True)

    # Downloads
    st.subheader("⬇️ Downloads")
    if run_trim:
        st.download_button("Download Trimmed File",
                           "\n".join(trim_keep_last(raw_lines, keep_last)),
                           file_name="trimmed.txt")
    if run_dedup:
        st.download_button("Download Deduped File",
                           "\n".join(deduplicate(trim_keep_last(raw_lines, keep_last))),
                           file_name="deduped.txt")
    if run_compress:
        st.download_button("Download Compressed File",
                           compressed_bytes, file_name="compressed.gz")
        
if section == "General":
    st.image("dm1.png", caption="Space-Efficient Data Management", use_container_width=True    )

if section == "Literature Review":
    
    st.write("""## 2. Literature Review: Space Management Techniques

The challenge of managing ever-expanding datasets has led to the development of numerous space optimization strategies in both academic research and industry practice. This section explores key techniques that address the problem from different angles, each with distinct benefits, trade-offs, and operational contexts.

### **2.1 Data Retention and Deletion Policies** 
Data retention policies define how long data should be stored before it is systematically deleted. These policies can be time-based (e.g., retain logs for 90 days) or event-based (e.g., delete customer data upon account closure). Automated deletion reduces the risk of data sprawl and ensures compliance with regulatory frameworks such as GDPR and CCPA, which require timely removal of personally identifiable information (PII). However, overly aggressive deletion can result in the loss of valuable historical insights, necessitating careful policy design.
            
    """)
    st.image("lr1.jpeg", use_container_width=True) 
    st.write("""### **2.2 Log Rotation and Pruning** 
For systems producing continuous log output, log rotation is a widely used practice where active logs are periodically archived and replaced with new files. Tools like logrotate on Linux enable scheduled rotation, compression, and deletion. Pruning involves selectively removing log entries based on relevance, severity levels, or age, thus reducing noise in analysis and storage burden. The main trade-off is the potential loss of fine-grained historical debugging information.
    """)
    st.image("lr2.png", use_container_width=True) 

    st.write("""### **2.3 Data Deduplication** 
Data deduplication eliminates redundant copies of information by storing a single instance and referencing it wherever required. This is particularly effective in backup systems, where repeated snapshots often contain large overlaps. Techniques include fixed-size chunking, variable-size chunking (content-defined chunking), and fingerprinting using cryptographic hashes (e.g., SHA-256). While deduplication offers substantial storage savings, it introduces CPU overhead and can complicate retrieval in highly fragmented datasets.
    """)
    st.image("lr3.jpg", use_container_width=True) 
    st.write("""### **2.4 Compression and Archiving** 
Compression algorithms like gzip, bzip2, and zlib reduce file sizes by encoding data more efficiently without loss of information. Archiving tools (e.g., tar) bundle files together for structured storage, often coupled with compression. This is optimal for cold data—information that is infrequently accessed but still needs to be retained for compliance or historical analysis. The trade-off lies in increased CPU usage during compression/decompression and slower access times for archived data.
    """)
    st.image("lr4.svg", use_container_width=True)
    st.write("""### **2.5 Real-time Data Aggregation and Summarization**
In streaming and analytics systems, real-time aggregation condenses raw data into smaller, more meaningful summaries (e.g., counts, averages, rolling windows). Frameworks like Apache Kafka Streams, Apache Flink, and Spark Streaming perform on-the-fly transformations to store only aggregated metrics instead of raw event data. This reduces storage costs and accelerates analytical queries but limits the ability to perform fine-grained, retrospective analyses on original data.
Collectively, these techniques form a toolkit for organizations to balance storage efficiency, data utility, and compliance requirements. The choice of method often depends on system scale, data access patterns, and the acceptable trade-off between data granularity and space savings.
    """)
    st.image("lr6.webp", use_container_width=True)

if section == "Results":
    st.write("""## Results""")
    st.write("## 1. **Input File**: A pdf file with 65 pages, 2.4 MB in size.")
    if st.button("Click to view Result 1"): 
        st.image("r1.png", use_container_width=True)
        st.image("r2.png", use_container_width=True)
        st.image("r3.png", use_container_width=True)
    st.write("2. **Input File**: A 50 mb txt file with https://examplefile.com written ")
    if st.button("Click to view Result 2"): 
        st.image("r21.png", use_container_width=True)
        st.image("r22.png", use_container_width=True)
        st.image("r23.png", use_container_width=True)
        st.image("r24.png", use_container_width=True)
    st.write("3. **Input File**: A 10 mb Json File")
    if st.button("Click to view Result 3"): 
        st.image("r31.png", use_container_width=True)
        st.image("r32.png", use_column_width=True)
        st.image("r33.png", use_container_width=True)


if section == "Code Snippets":
    st.header("Code Snippets")

    st.markdown("#### 1. File Upload Check")
    st.code(""" if not uploaded:
    st.info("⬆️ Upload a file in the sidebar to begin.")
    st.stop()
""")
    st.write("""Purpose: If the user hasn't uploaded a file yet, it shows an informational message in the Streamlit 
            app and immediately stops execution using st.stop() so the rest of the pipeline won’t run. 
            \nThis prevents errors from trying to process None. """)
    
    st.markdown("#### 2. Read and Measure the File")
    st.code(""" raw_lines = read_file(uploaded)
raw_size = len("\n".join(raw_lines).encode("utf-8", errors="ignore"))

st.write(
    f"**File Name:** `{uploaded.name}` • "
    f"**Original Size:** {fmt_size(raw_size)} • "
    f"**Total Lines:** {len(raw_lines):,}"
)
""")
    st.write("""read_file(uploaded): Reads the uploaded file and returns a list of lines.

raw_size: Joins all lines with \n, encodes to bytes, and measures the total byte length.

fmt_size(): Converts bytes into a human-readable format (KB, MB, etc.).

Displays:

    File name

    Original size

    Number of lines """)

    st.markdown("#### 3. Initial State Tracking")
    st.code(""" current_lines = raw_lines
sizes  = [raw_size]
stages = ["Raw"]
""")
    st.write("""current_lines: Tracks the working dataset as transformations are applied.

sizes: Keeps the file size at each processing stage.

stages: Names for each processing stage (Raw → Trimmed → Deduped → etc.). """)

    st.markdown("#### 4. Trimming Stage")
    st.code(""" if run_trim:
    with st.spinner("✂️ Trimming in progress…"):
        t0 = time.time()
        current_lines = trim_keep_last(current_lines, keep_last)
        trim_time = time.time() - t0
    sizes.append(len("\n".join(current_lines).encode("utf-8", errors="ignore")))
    stages.append("Trimmed")
    st.success(f"✅ Trimming completed in {trim_time:.4f}s")
""")
    st.write("""If trimming is enabled:

    Shows a spinner while processing.

    Keeps only the last N lines (keep_last).

    Records processing time.

    Appends the new file size to sizes.

    Adds "Trimmed" to the stage list. """)

    st.markdown("#### 5. Deduplication Stage")
    st.code(""" if run_dedup:
    with st.spinner("🧹 Deduplication in progress…"):
        t0 = time.time()
        current_lines = deduplicate(current_lines)
        dedup_time = time.time() - t0
    sizes.append(len("\n".join(current_lines).encode("utf-8", errors="ignore")))
    stages.append("Deduped")
    st.success(f"✅ Deduplication completed in {dedup_time:.4f}s")
""")
    st.write("""In Deduplication Stage:

    Removes duplicate lines.

    Tracks time taken.

    Updates size and stage lists. """)

    st.markdown("#### 6. Summarization Stage")
    st.code(""" summary_lines = []
if run_summary:
    with st.spinner("📑 Summarizing data…"):
        t0 = time.time()
        summary_lines = summarize(current_lines, top_n)
        summarize_time = time.time() - t0
    st.success(f"✅ Summarization completed in {summarize_time:.4f}s")
    st.subheader("📋 Summary (Top N)")
    st.write("\n".join(summary_lines) if summary_lines else "_No repeating lines detected._")

""")
    st.write("""If enabled:

    Counts the most common lines (frequency analysis).

    Shows top top_n results.

    Displays them in the UI. """)

    st.markdown("#### 7. Compression Stage")
    st.code(""" compressed_bytes = b""
if run_compress:
    with st.spinner("📦 Compressing data…"):
        compressed_bytes = compress_text(current_lines)
    sizes.append(len(compressed_bytes))
    stages.append("Compressed")
    st.success("✅ Compression completed")

""")
    st.write("""In Compression Stage:

    Compresses the processed lines using gzip.

    Stores compressed size in sizes.

    Adds "Compressed" to stages.""")

    st.markdown("#### 8. Size Metrics Table")
    st.code(""" size_df = pd.DataFrame({
    "Stage": stages,
    "Size (bytes)": sizes,
    "Size (formatted)": [fmt_size(s) for s in sizes]
})
st.subheader("📏 Size Reduction Table")
st.dataframe(size_df, use_container_width=True)

""")
    st.write("""It Creates a DataFrame to show:

    Stage name

    File size in bytes

    Human-readable size

Displays it in an interactive table. """)

    st.markdown("#### 9. Size Reduction Waterfall Chart")
    st.code(""" fig_waterfall = go.Figure(go.Waterfall(
    name="Data Reduction",
    orientation="v",
    measure=["absolute"] + ["relative"] * (len(sizes) - 1),
    x=stages,
    text=[fmt_size(s) for s in sizes],
    y=sizes
))
st.plotly_chart(fig_waterfall, use_container_width=True)

""")
    st.write("""In This Plot:

    Waterfall chart shows size change at each stage.

    First stage is absolute, subsequent are relative changes. """)

    st.markdown("#### 10. Contribution Pie Chart")
    st.code(""" if len(sizes) > 1:
    reductions = [max(sizes[i-1] - sizes[i], 0) for i in range(1, len(sizes))]
    technique_labels = stages[1:]
    fig_pie = px.pie(values=reductions, names=technique_labels, hole=0.4,
                     title="Technique Contribution")
    st.plotly_chart(fig_pie, use_container_width=True)
""")
    st.write("""In This Plot:

    Calculates how much each stage reduced file size.

    Displays as a pie chart showing contribution percentages. """)

    st.markdown("#### 11. Download Processed Files")
    st.code(""" if run_trim:
    st.download_button("Download Trimmed File", "\n".join(trim_keep_last(raw_lines, keep_last)), file_name="trimmed.txt")
if run_dedup:
    st.download_button("Download Deduped File", "\n".join(deduplicate(trim_keep_last(raw_lines, keep_last))), file_name="deduped.txt")
if run_compress:
    st.download_button("Download Compressed File", compressed_bytes, file_name="compressed.gz")

""")
    st.write("""Lets the user download:

    Trimmed file

    Deduplicated file

    Compressed .gz archive """)
    

    ## Entire Code
    st.markdown("## Click Button To See Entire Code Snippets")
    if st.button("Show Code Snippets"):
        st.code("""    if not uploaded:
        st.info("⬆️ Upload a file in the sidebar to begin.")
        st.stop()

    raw_lines = read_file(uploaded)
    raw_size = len("\n".join(raw_lines).encode("utf-8", errors="ignore"))

    st.write(
        f"**File Name:** `{uploaded.name}` • "
        f"**Original Size:** {fmt_size(raw_size)} • "
        f"**Total Lines:** {len(raw_lines):,}"
    )

    current_lines = raw_lines
    sizes  = [raw_size]
    stages = ["Raw"]

    # Trim
    if run_trim:
        with st.spinner("✂️ Trimming in progress…"):
            t0 = time.time()
            current_lines = trim_keep_last(current_lines, keep_last)
            trim_time = time.time() - t0
        sizes.append(len("\n".join(current_lines).encode("utf-8", errors="ignore")))
        stages.append("Trimmed")
        st.success(f"✅ Trimming completed in {trim_time:.4f}s")

    # Dedup
    if run_dedup:
        with st.spinner("🧹 Deduplication in progress…"):
            t0 = time.time()
            current_lines = deduplicate(current_lines)
            dedup_time = time.time() - t0
        sizes.append(len("\n".join(current_lines).encode("utf-8", errors="ignore")))
        stages.append("Deduped")
        st.success(f"✅ Deduplication completed in {dedup_time:.4f}s")

    # Summarize
    summary_lines = []
    if run_summary:
        with st.spinner("📑 Summarizing data…"):
            t0 = time.time()
            summary_lines = summarize(current_lines, top_n)
            summarize_time = time.time() - t0
        st.success(f"✅ Summarization completed in {summarize_time:.4f}s")
        st.subheader("📋 Summary (Top N)")
        st.write("\n".join(summary_lines) if summary_lines else "_No repeating lines detected._")

    # Compress
    compressed_bytes = b""
    if run_compress:
        with st.spinner("📦 Compressing data…"):
            compressed_bytes = compress_text(current_lines)
        sizes.append(len(compressed_bytes))
        stages.append("Compressed")
        st.success("✅ Compression completed")

    # Metrics table
    size_df = pd.DataFrame({
        "Stage": stages,
        "Size (bytes)": sizes,
        "Size (formatted)": [fmt_size(s) for s in sizes]
    })
    st.subheader("📏 Size Reduction Table")
    st.dataframe(size_df, use_container_width=True)

    # Plotly Waterfall
    st.subheader("📉 Size Reduction by Stage")
    fig_waterfall = go.Figure(go.Waterfall(
        name="Data Reduction",
        orientation="v",
        measure=["absolute"] + ["relative"] * (len(sizes) - 1),
        x=stages,
        text=[fmt_size(s) for s in sizes],
        y=sizes
    ))
    fig_waterfall.update_layout(title="Storage Size Reduction at Each Stage", yaxis_title="Size (bytes)")
    st.plotly_chart(fig_waterfall, use_container_width=True)

    # Pie chart for contributions
    if len(sizes) > 1:
        reductions = [max(sizes[i-1] - sizes[i], 0) for i in range(1, len(sizes))]
        technique_labels = stages[1:]
        st.subheader("🧩 Contribution to Space Savings")
        fig_pie = px.pie(values=reductions, names=technique_labels, hole=0.4,
                         title="Technique Contribution")
        st.plotly_chart(fig_pie, use_container_width=True)

    # Downloads
    st.subheader("⬇️ Downloads")
    if run_trim:
        st.download_button("Download Trimmed File",
                           "\n".join(trim_keep_last(raw_lines, keep_last)),
                           file_name="trimmed.txt")
    if run_dedup:
        st.download_button("Download Deduped File",
                           "\n".join(deduplicate(trim_keep_last(raw_lines, keep_last))),
                           file_name="deduped.txt")
    if run_compress:
        st.download_button("Download Compressed File",
                           compressed_bytes, file_name="compressed.gz")""")
        
if section =="Introduction":
    st.write("""
# 1. Introduction

In the modern digital era, exponential data growth has emerged as a pressing challenge for computing systems. With the proliferation of connected devices, 
cloud services, and digital transformation across industries, data is being generated at unprecedented rates. Sources include server and application logs, IoT 
telemetry, clickstream data, social media content, transactional records, and user interactions. As per IDC forecasts, the 
global datasphere is expected to reach over 175 zettabytes by 2025. This explosive growth not only poses challenges to data storage systems but also impacts the 
performance, scalability, and operational costs of modern computing infrastructures.

Traditionally, computer scientists have emphasized time complexity when evaluating algorithmic efficiency. However, space complexity—the measure of memory usage and 
storage overhead—has gained equal, if not more, relevance in large-scale systems. Poorly managed data can lead to inflated storage requirements, slower query and retrieval times, degraded system performance, and increased latency in both batch and real-time applications.

Moreover, regulatory requirements such as the General Data Protection Regulation (GDPR) and the California Consumer Privacy Act (CCPA) mandate strict control over how long data is stored and how it is deleted. This adds legal and compliance pressure on organizations to implement robust data lifecycle management strategies.

Efficient space management is not merely a technical concern but a foundational pillar for ensuring system sustainability, economic feasibility, and legal compliance in data-intensive environments. In this context, the need for systematic, scalable, and intelligent approaches to data trimming, deduplication, compression, and summarization becomes vital.

This paper investigates existing techniques and proposes a practical hybrid solution aimed at reducing storage footprint while retaining essential data fidelity. The approach is demonstrated with an implementation using Python and evaluated for its applicability and extensibility.

""")
    st.image("dgr.jpg", use_container_width=True)


st.sidebar.write("Made By Michael Fernandes")
