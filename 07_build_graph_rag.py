import json
import os
import sys
import networkx as nx
from groq import Groq

sys.stdout.reconfigure(encoding='utf-8')

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

print("=" * 65)
print("🕸️ AI4INVEST GRAPH RAG: KNOWLEDGE GRAPH BUILDER")
print("=" * 65)

# Load existing vector-retrieved context or raw dump
with open("real_articles_dump.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

# Sample high-signal articles for structured entity-relation extraction
sample_articles = articles[:30]

print(f"\n1. Extracting Entity-Relation Triplets from {len(sample_articles)} market articles...")

extraction_prompt = f"""
You are an expert Financial Knowledge Graph Engineer.
Extract structured Knowledge Graph Triplets from the following market news headlines.

FORMAT INSTRUCTION:
Return ONLY a valid JSON array of objects, where each object has:
"subject": Entity A (e.g. "HDFC Bank", "Crude Oil", "RBI")
"relation": Relationship verb (e.g. "CUTS_JOBS", "RAISES_COSTS", "INVESTS_IN", "ACQUIRED")
"object": Entity B (e.g. "Workforce", "Borrowing Costs", "Ather Energy")

NEWS HEADLINES:
{json.dumps(sample_articles, indent=2)}
"""

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": extraction_prompt}],
    temperature=0.0,
    response_format={"type": "json_object"}
)

raw_json = response.choices[0].message.content
triplets_data = json.loads(raw_json)

# Extract triplet list from response
triplets = triplets_data.get("triplets", triplets_data.get("data", []))
if isinstance(triplets_data, list):
    triplets = triplets_data

print(f" Extracted {len(triplets)} Knowledge Triplets successfully!")

# 2. Build NetworkX Directed Graph G = (V, E)
G = nx.DiGraph()

for t in triplets:
    subj = t.get("subject", "").strip()
    rel = t.get("relation", "").strip()
    obj = t.get("object", "").strip()
    if subj and rel and obj:
        G.add_edge(subj, obj, relation=rel)

print(f"\n2. Knowledge Graph Summary:")
print(f"   Total Nodes (Entities): {G.number_of_nodes()}")
print(f"   Total Edges (Relationships): {G.number_of_edges()}")

# 3. Save Graph Structure to JSON
graph_data = {
    "nodes": list(G.nodes()),
    "edges": [
        {"source": u, "target": v, "relation": d["relation"]}
        for u, v, d in G.edges(data=True)
    ]
}

with open("knowledge_graph.json", "w", encoding="utf-8") as f:
    json.dump(graph_data, f, indent=4)

print("\n Saved Knowledge Graph to knowledge_graph.json!")