import json
import os
import sys
import networkx as nx
from groq import Groq

sys.stdout.reconfigure(encoding='utf-8')

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

print("=" * 65)
print("🕸️ AI4INVEST GRAPH RAG: MULTI-HOP GRAPH SYNTHESIS")
print("=" * 65)

# Load Knowledge Graph from JSON
with open("knowledge_graph.json", "r", encoding="utf-8") as f:
    data = json.load(f)

G = nx.DiGraph()
for edge in data["edges"]:
    G.add_edge(edge["source"], edge["target"], relation=edge["relation"])

print(f"\n1. Loaded Knowledge Graph with {G.number_of_nodes()} Nodes and {G.number_of_edges()} Edges.")

# Perform Graph Traversal to extract multi-hop paths
print("\n2. Extracting Multi-Hop Relationship Chains (Paths)...")
graph_paths = []
for u, v, d in G.edges(data=True):
    path_str = f"[{u}] ──({d['relation']})──> [{v}]"
    graph_paths.append(path_str)

# Query Groq 70B with Knowledge Graph Structure
prompt = f"""
You are a Chief Knowledge Officer & Financial Graph Analyst.
Analyze the following multi-hop Knowledge Graph paths extracted from July 14-16, 2026 news articles:

KNOWLEDGE GRAPH PATHS (ENTITIES & RELATIONSHIPS):
{json.dumps(graph_paths, indent=2)}

Generate a **GRAPH RAG MARKET INTELLIGENCE REPORT** containing:
1. Executive Entity Summary (Key connected players, regulators, and assets)
2. Causal Relationship Chains (Multi-step systemic impacts, e.g. how Event A drives Event B which impacts Sector C)
3. Graph Centrality Analysis (Which single entity node has the most influence in the network?)
4. Strategic Market Recommendations based on Graph Connectivity
"""

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0
)

report = response.choices[0].message.content

with open("GRAPH_RAG_MARKET_REPORT.md", "w", encoding="utf-8") as f:
    f.write(report)

print("\n Graph RAG Report Generated Successfully!")
print(" Saved output to GRAPH_RAG_MARKET_REPORT.md\n")
print(report)