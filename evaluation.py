from __future__ import annotations
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
import json
from uuid import uuid4

try:
    from chatbot import make_llm, embeddings_model, best_answer, session_mgr
except ImportError as e:
    raise ImportError("❌ Could not import chatbot.py. Ensure chatbot.py is in the same directory.") from e

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─────────── VALIDATION ───────────
def validate_test_data(test_data: List[Dict], required_keys: List[str]) -> None:
    """Validate test data format."""
    for i, entry in enumerate(test_data):
        for key in required_keys:
            if key not in entry:
                raise ValueError(f"❌ Missing key '{key}' in test_data entry {i}")

# ─────────── RETRIEVAL EVALUATION ───────────
def evaluate_retrieval(test_data: List[Dict], top_k: int = 3) -> Dict[str, float]:
    """
    Evaluate RAG retrieval quality with precision, recall, MRR, and semantic similarity.
    test_data format: [{"query": str, "expected_context": str}]
    """
    validate_test_data(test_data, ["query", "expected_context"])
    precision_scores, recall_scores, mrr_scores, semantic_sims = [], [], [], []

    all_ids = session_mgr.all_ids()
    if not all_ids:
        raise RuntimeError("❌ No active session found in session_mgr.")
    cur_sid = all_ids[0]

    for entry in test_data:
        query, expected_context = entry["query"], entry["expected_context"]

        # Retrieve answer + context
        try:
            answer, retrieved_docs = best_answer(
                query,
                session_mgr.get(cur_sid),
                cur_sid,
                "en",
                return_docs=True
            )
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {e}")
            retrieved_docs = []
            answer = ""

        logger.info(f"\nQ: {query}\nExpected Context: {expected_context}\nRetrieved Docs: {retrieved_docs}\nAnswer: {answer}\n")

        # Precision, Recall, MRR
        hits = [1 for doc in retrieved_docs[:top_k] if expected_context.lower() in doc.lower()]
        precision = sum(hits) / len(retrieved_docs[:top_k]) * 100 if retrieved_docs else 0
        recall = 100 if any(hits) else 0
        mrr = 1 / (next((i + 1 for i, hit in enumerate(hits) if hit), len(retrieved_docs) + 1))

        precision_scores.append(precision)
        recall_scores.append(recall)
        mrr_scores.append(mrr * 100)

        # Semantic Similarity
        try:
            ref_emb = embeddings_model.embed_query(expected_context)
            top_doc_emb = embeddings_model.embed_query(retrieved_docs[0]) if retrieved_docs else ref_emb
            sim = cosine_similarity([ref_emb], [top_doc_emb])[0][0]
            semantic_sims.append(sim * 100)
        except Exception as e:
            logger.error(f"Embedding similarity error for query '{query}': {e}")
            semantic_sims.append(0.0)

    # F1 Score
    f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision_scores, recall_scores)]

    return {
        "Precision@K": np.mean(precision_scores),
        "Recall@K": np.mean(recall_scores),
        "F1@K": np.mean(f1_scores),
        "MRR@K": np.mean(mrr_scores),
        "Semantic Similarity": np.mean(semantic_sims)
    }

# ─────────── GENERATION EVALUATION ───────────
def evaluate_generation(test_data: List[Dict]) -> Dict[str, float]:
    """
    Evaluate generated answers quality (relevance, completeness, similarity).
    test_data format: [{"question": str, "expected_answer": str}]
    """
    validate_test_data(test_data, ["question", "expected_answer"])
    relevance_scores, completeness_scores, similarity_scores = [], [], []

    all_ids = session_mgr.all_ids()
    if not all_ids:
        raise RuntimeError("❌ No active session found in session_mgr.")
    cur_sid = all_ids[0]

    llm = make_llm(temp=0.0)

    for entry in test_data:
        q, expected = entry["question"], entry["expected_answer"]
        try:
            answer = best_answer(q, session_mgr.get(cur_sid), cur_sid, "en")
        except Exception as e:
            logger.error(f"Generation failed for question '{q}': {e}")
            answer = ""

        logger.info(f"Q: {q}\nExpected: {expected}\nAnswer: {answer}\n")

        # Semantic Similarity
        try:
            ref_emb = embeddings_model.embed_query(expected)
            ans_emb = embeddings_model.embed_query(answer) if answer else ref_emb
            sim = cosine_similarity([ref_emb], [ans_emb])[0][0]
            similarity_scores.append(sim * 100)
        except Exception as e:
            logger.error(f"Semantic similarity error for question '{q}': {e}")
            similarity_scores.append(0.0)

        # LLM Judge for Relevance
        try:
            relevance_prompt = (
                f"Rate the relevance (1–5) of this answer to the question (5=highly relevant, 1=irrelevant).\n"
                f"Q: {q}\nAnswer: {answer}\nOnly return a number."
            )
            relevance_score = llm.invoke(relevance_prompt).content.strip()
            relevance_score = int(relevance_score) if relevance_score.isdigit() and 1 <= int(relevance_score) <= 5 else 3
            relevance_scores.append(relevance_score * 20)
        except Exception as e:
            logger.error(f"LLM relevance scoring error for question '{q}': {e}")
            relevance_scores.append(50)

        # LLM Judge for Completeness
        try:
            completeness_prompt = (
                f"Rate the completeness (1–5) of this answer compared to the expected answer (5=fully complete, 1=incomplete).\n"
                f"Q: {q}\nExpected: {expected}\nAnswer: {answer}\nOnly return a number."
            )
            completeness_score = llm.invoke(completeness_prompt).content.strip()
            completeness_score = int(completeness_score) if completeness_score.isdigit() and 1 <= int(completeness_score) <= 5 else 3
            completeness_scores.append(completeness_score * 20)
        except Exception as e:
            logger.error(f"LLM completeness scoring error for question '{q}': {e}")
            completeness_scores.append(50)

    return {
        "Answer Relevance": np.mean(relevance_scores),
        "Answer Completeness": np.mean(completeness_scores),
        "Answer Similarity": np.mean(similarity_scores)
    }

# ─────────── GRAPH VISUALIZATION ───────────
def plot_metrics(retrieval_metrics: Dict[str, float], generation_metrics: Dict[str, float], output_file: str = "metrics.png"):
    """
    Plot retrieval and generation metrics in a grouped bar chart, excluding 0% scores, with enhanced UI.
    """
    # Filter out metrics with 0% score
    filtered_retrieval = {k: v for k, v in retrieval_metrics.items() if v > 0}
    filtered_generation = {k: v for k, v in generation_metrics.items() if v > 0}

    if not filtered_retrieval and not filtered_generation:
        logger.warning("No metrics above 0% to plot.")
        return

    # Dynamic figure size based on number of metrics
    num_metrics = len(filtered_retrieval) + len(filtered_generation)
    fig, ax = plt.subplots(figsize=(max(10, num_metrics * 2), 6))

    # Data for plotting
    all_metrics = list(filtered_retrieval.keys()) + list(filtered_generation.keys())
    all_values = list(filtered_retrieval.values()) + list(filtered_generation.values())
    colors = ['#1E90FF'] * len(filtered_retrieval) + ['#32CD32'] * len(filtered_generation)  # DodgerBlue and LimeGreen

    # Create bars
    bars = ax.bar(all_metrics, all_values, color=colors)
    ax.set_ylim(0, 100)
    ax.set_title("RAG Evaluation Metrics", fontsize=14, pad=15, weight='bold')
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha="center", fontsize=9, weight='bold')

    # Add legend
    ax.legend(['Retrieval Metrics', 'Generation Metrics'], loc='upper right', fontsize=10, frameon=True)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

# ─────────── SAVE RESULTS ───────────
def save_results(metrics: Dict[str, float], filename: str = "evaluation_results.json"):
    """Save evaluation metrics to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Results saved to {filename}")

# ─────────── MAIN ───────────
if __name__ == "__main__":
    # Retrieval test
    retrieval_test = [
        {"query": "What is Azure?", "expected_context": "Azure is a cloud by Microsoft"},
        {"query": "What is LangChain?", "expected_context": "LangChain is a framework for building LLM-powered apps"},
    ]

    retrieval_metrics = evaluate_retrieval(retrieval_test, top_k=3)
    logger.info(f"Retrieval Metrics: {retrieval_metrics}")
    save_results(retrieval_metrics, "retrieval_results.json")

    # Generation test
    generation_test = [
        {"question": "What is Azure?", "expected_answer": "Azure is Microsoft’s cloud platform."},
        {"question": "What is LangChain?", "expected_answer": "LangChain is a framework for building AI applications with LLMs."},
    ]

    generation_metrics = evaluate_generation(generation_test)
    logger.info(f"Generation Metrics: {generation_metrics}")
    save_results(generation_metrics, "generation_results.json")

    # Plot metrics
    plot_metrics(retrieval_metrics, generation_metrics)