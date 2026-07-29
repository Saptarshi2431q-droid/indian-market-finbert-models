import json
import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Fix Windows OpenMP and UTF-8 terminal encoding
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.stdout.reconfigure(encoding='utf-8')

print('=' * 65, flush=True)
print('🚀 AI4INVEST VECTOR RAG PIPELINE (NIFTY BANK RETRIEVAL)', flush=True)
print('=' * 65, flush=True)

# 1. Check input file
dump_file = 'real_articles_dump.json'
if not os.path.exists(dump_file):
  print(
      f'❌ Error: {dump_file} not found! Please run 01_load_csv_data.py first.',
      flush=True,
  )
  sys.exit(1)

with open(dump_file, 'r', encoding='utf-8') as f:
  articles = json.load(f)

# Filter out very short texts
valid_articles = [text.strip() for text in articles if len(text.strip()) > 20]
print(f'\n Loaded {len(valid_articles)} articles from {dump_file}.', flush=True)

# 2. Load PyTorch Transformer Embedding Engine
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f' Using compute device: {device.upper()}', flush=True)

model_name = 'sentence-transformers/all-MiniLM-L6-v2'
print(f' Loading embedding model: {model_name}...', flush=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)


# Helper function for batch vector embedding generation
def get_embeddings(texts, batch_size=32):
  all_embeddings = []
  for i in range(0, len(texts), batch_size):
    batch = texts[i : i + batch_size]
    encoded_input = tokenizer(
        batch, padding=True, truncation=True, max_length=512, return_tensors='pt'
    ).to(device)

    with torch.no_grad():
      model_output = model(**encoded_input)

    # Mean Pooling
    token_embeddings = model_output[0]
    attention_mask = encoded_input['attention_mask'].unsqueeze(-1).expand(
        token_embeddings.size()
    )
    sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
    sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
    pooled = sum_embeddings / sum_mask

    # Normalize vectors for Cosine Similarity calculations
    normalized = F.normalize(pooled, p=2, dim=1)
    all_embeddings.append(normalized.cpu())

    pct = int((min(i + batch_size, len(texts)) / len(texts)) * 100)
    print(
        f'   Embedding Progress: {min(i + batch_size, len(texts))}/{len(texts)} articles ({pct}%)',
        end='\r',
        flush=True,
    )

  print('\n Embeddings generated successfully!', flush=True)
  return torch.cat(all_embeddings, dim=0)


# Generate embeddings for all 1,176 articles
print(
    '\n Vectorizing dataset into 384-dimensional semantic space...', flush=True
)
article_vectors = get_embeddings(valid_articles)

# 3. Vector Search Query for NIFTY BANK
query = (
    'NIFTY Bank performance, RBI Monetary Policy, HDFC Bank, Axis Bank, ICICI'
    ' Bank, SBI, private sector banks, credit growth, NPA bad loans'
)
print(f"\n Querying Vector Index for NIFTY BANK:\n   '{query}'", flush=True)

# Encode query into vector space
query_vector = get_embeddings([query])

# Calculate Cosine Similarity across all articles mathematically
similarities = torch.mm(query_vector, article_vectors.T).squeeze(0)

# Retrieve Top 10 highest-scoring documents
top_k = 10
top_indices = torch.topk(similarities, k=top_k).indices.tolist()

retrieved_docs = []
print('\n' + '=' * 65, flush=True)
print('🎯 TOP 10 RETRIEVED BANKING ARTICLES (VECTOR SEARCH)', flush=True)
print('=' * 65, flush=True)

for rank, idx in enumerate(top_indices, 1):
  score = round(float(similarities[idx]), 4)
  doc = valid_articles[idx]
  retrieved_docs.append({'rank': rank, 'score': score, 'text': doc})
  print(f'\n[{rank}] (Similarity Score: {score})', flush=True)
  print(f'    {doc[:250]}...', flush=True)

# 4. Save retrieved context for downstream LLM generation
output_payload = {
    'query': query,
    'top_k': top_k,
    'retrieved_chunks': [item['text'] for item in retrieved_docs],
    'detailed_results': retrieved_docs,
}

with open('nifty_bank_vector_context.json', 'w', encoding='utf-8') as f:
  json.dump(output_payload, f, indent=4)

print('\n' + '=' * 65, flush=True)
print(
    ' SUCCESS: Retrieved Nifty Bank vector context saved to'
    ' nifty_bank_vector_context.json!',
    flush=True,
)
print('=' * 65 + '\n', flush=True)