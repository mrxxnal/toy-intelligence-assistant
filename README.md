# Toy Intelligence Assistant

An AI-powered multi-modal recommendation system that suggests toys using images, text queries, and conversational AI.

---

## Features

- Image-based toy similarity search (CLIP + FAISS)
- Smart gift recommendation engine
- LLM-powered toy explanations (RAG)
- Conversational Q&A about toys
- Multi-mode interface (Finder / Gift Advisor / Parent Assistant)

---

## Architecture

User Input → Streamlit UI  
→ Embedding Model (CLIP)  
→ FAISS Vector Search  
→ Retrieval System  
→ LLM Explanation Layer  
→ Final Output

---

## Tech Stack

- Python
- Streamlit
- HuggingFace Transformers (CLIP)
- FAISS
- OpenAI GPT (RAG layer)
- Pandas

---

## Evaluation

To evaluate the performance of the Toy Intelligence Assistant, I tested the system across a variety of real-world scenarios that simulate how a user would actually interact with it — both through image uploads and natural language queries. The goal was not just to check whether the system “works,” but to understand how well it retrieves meaningful, relevant, and context-aware toy recommendations.

Overall, the system performs reliably and produces results that are semantically aligned with user intent, especially in structured toy categories such as crafts, puzzles, and party supplies. The combination of CLIP-based embeddings and FAISS retrieval creates a strong baseline for visual similarity, while the ranking logic in the gift advisor mode adds an additional layer of practical usefulness.

⸻

Sample Queries

Query 1 (Image-based retrieval):

* Input: Image of a craft kit (arts and DIY activity set)
* Output: The system consistently returned similar creative kits such as craft sets, puzzle-based toys, and DIY activity products that share both visual patterns and functional intent.
* Observation: Even when product images varied slightly in color or packaging style, the embedding space effectively grouped them based on underlying category similarity (creative, hands-on activity toys).

⸻

Query 2 (Text-based gift recommendation):

* Input: “gift for 7-year-old girl under $20 who likes crafts”
* Output: The system recommended low-cost creative kits, art supplies, and DIY craft-based toys that align with both the age constraint and budget limitation.
* Observation: The lightweight ranking approach successfully prioritized affordability and age suitability while still maintaining thematic relevance to “craft-related” intent.

⸻

Observations

Across multiple test runs, several consistent patterns emerged:

* Strong semantic grouping via FAISS:
    The CLIP embeddings do a good job of clustering visually and conceptually similar toys, especially in categories like crafts, puzzles, and educational kits.
* Category signals improve precision:
    Products with clearly defined categories (e.g., “Arts & Crafts”, “Puzzles”) tend to produce much more accurate and stable recommendations compared to loosely described items.
* Age alignment improves usability:
    Even simple age parsing significantly improves the practical relevance of recommendations, especially for gift-oriented queries.
* Budget-aware filtering enhances realism:
    Incorporating price constraints makes the system feel much closer to a real shopping assistant rather than just a similarity engine.


Qualitative Insight

From a user perspective, the system feels most reliable when working within structured toy domains such as:

* Arts & Crafts kits
* Jigsaw puzzles and brain games
* Party supplies and themed sets

In these categories, both visual embeddings and textual metadata reinforce each other, leading to strong recommendation consistency.

However, performance becomes less stable in cases where:

* Product descriptions are very short or generic (e.g., “toy set” or “fun activity”)
* Images lack clear visual cues or are heavily stylized packaging shots
* Items span multiple ambiguous categories without strong semantic anchors

In these scenarios, the model still retrieves reasonable neighbors, but the ranking confidence is noticeably weaker, which is expected given the reliance on embedding similarity rather than deep product understanding.


Final Reflection

From a practical standpoint, the system demonstrates how combining vision embeddings, vector search, and lightweight reasoning can already produce a surprisingly capable recommendation engine without requiring a fully trained domain-specific model. The most interesting takeaway is that even relatively simple ranking heuristics layered on top of FAISS significantly improve real-world usability — especially for tasks like gift discovery, where intent is often vague but expectations are high.

---

## Setup Instructions

```bash
pip install -r requirements.txt
streamlit run app/main.py