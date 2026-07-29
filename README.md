# AI4Invest RAG Pipeline Optimization

An enterprise-grade, preprocessed Retrieval-Augmented Generation (RAG) pipeline designed to extract hidden market signals from financial news articles and generate institutional directional calls for Nifty/Bank Nifty.

## 📌 Architecture Highlights
- **Sentence-Level Triage:** Solves "Signal Dilution" using regex chunking paired with custom fine-tuned `sapt3009/AI4Invest-Indian-FinBERT`.
- **Set-Based Deduplication:** Achieves over **99% token footprint reduction** by purging duplicate media coverage into unique bullish/bearish signal sets.
- **Routed Model Stack:** Leverages cheap SLMs for high-volume data bouncer duties and flagship LLMs (`llama-3.3-70b-versatile` via Groq) strictly for clean context synthesis.

## 📁 Repository Structure
- `01_load_csv_data.py`: Ingests and merges daily financial CSV news dumps.
- `02_finbert_sentence_chunker.py`: Slices text into sentences and applies FinBERT sentiment confidence thresholds.
- `03_deduplicate_and_assemble.py`: Deduplicates thousands of sentences into a compact JSON context payload using Python sets.
- `04_generate_real_nifty_call.py`: Queries flagship Groq LLM with strict temperature controls to output markdown investment theses.
- `run_pipeline.py`: Master CLI orchestrator running the end-to-end pipeline.
- `REAL_NIFTY_MARKET_CALL.md`: Output directional trading thesis.

## 🚀 Quickstart
1. Set your API key:
   ```bash
   $env:GROQ_API_KEY="your_groq_api_key"