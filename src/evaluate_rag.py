import argparse
import csv
import json
import re
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


def ask_rag(question: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Appelle le pipeline RAG défini dans src/rag.py et retourne un dict normalisé.
    Utilise un cache local pour éviter de recharger index + modèle à chaque question.
    """
    import rag  # src/rag.py (si tu lances: python src/evaluate_rag.py depuis la racine du projet)
    from sentence_transformers import SentenceTransformer

    # Cache local attaché à la fonction
    if not hasattr(ask_rag, "_cache"):
        index, meta, texts = rag.load_index()
        emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        ask_rag._cache = {
            "index": index,
            "meta": meta,
            "texts": texts,
            "emb_model": emb_model,
        }

    cache = ask_rag._cache
    index = cache["index"]
    meta = cache["meta"]
    texts = cache["texts"]
    emb_model = cache["emb_model"]

    q_emb = emb_model.encode([question], normalize_embeddings=True).astype("float32")

    scores, ids = index.search(q_emb, top_k)

    contexts = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue

        m = meta[idx]
        snippet = texts[idx]

        contexts.append({
            "text": snippet,
            "source": m["source"],
            "page": m["page"],
            "score": float(score),
        })

    answer = rag.generate_answer(question, contexts)

    return {
        "answer_text": answer,
        "sources": [
            {"source": c["source"], "page": c["page"], "score": c["score"]}
            for c in contexts
        ],
    }

CITATION_PATTERNS = [
    r"\[\d+\]",                  # ex: [1], [2]
    r"p(?:age)?\s*\.?\s*\d+",    # ex: p. 3 / page 3
    r"source\s*[:\-]",           # ex: Source: ...
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tests/validation légers pour pipeline RAG.")
    parser.add_argument(
        "--questions",
        type=str,
        default="eval/evaluation_questions.json",
        help="Chemin vers le fichier JSON des questions.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="eval/results.csv",
        help="Chemin du CSV de sortie.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Nombre de passages récupérés (top-k).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Nombre d'exécutions par question (pour moyenner le temps).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Affiche les réponses pendant l'évaluation.",
    )
    return parser.parse_args()


def load_questions(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Le fichier questions doit contenir une liste JSON.")

    required_keys = {"id", "question", "expected_type"}
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Question #{i} n'est pas un objet JSON.")
        missing = required_keys - set(item.keys())
        if missing:
            raise ValueError(f"Question #{i} manque les clés: {missing}")

    return data


def detect_citation_stats(answer_text: str) -> Dict[str, Any]:
    """
    Heuristique simple de détection de citations.
    Ce n'est pas parfait, mais suffisant pour tests/validation légers.
    """
    if not answer_text:
        return {
            "has_citation": 0,
            "n_citation_matches": 0,
            "citation_format_hint": 0,
        }

    matches = []
    for pattern in CITATION_PATTERNS:
        found = re.findall(pattern, answer_text, flags=re.IGNORECASE)
        matches.extend(found)

    n_matches = len(matches)
    return {
        "has_citation": 1 if n_matches > 0 else 0,
        "n_citation_matches": n_matches,
        "citation_format_hint": 1 if n_matches > 0 else 0,
    }


def one_query_with_timing(question: str, top_k: int, runs: int = 1) -> Dict[str, Any]:
    """
    Exécute plusieurs fois la même question (optionnel) pour moyenner le temps.
    Retourne le dernier résultat + stats temps.
    """
    times = []
    last_out: Dict[str, Any] = {"answer_text": "", "sources": []}

    for _ in range(runs):
        t0 = time.perf_counter()
        out = ask_rag(question=question, top_k=top_k)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        last_out = out if isinstance(out, dict) else {"answer_text": str(out), "sources": []}

    answer_text = str(last_out.get("answer_text", "") or "")
    sources = last_out.get("sources", [])
    if sources is None:
        sources = []

    return {
        "answer_text": answer_text,
        "sources": sources,
        "elapsed_sec_mean": mean(times) if times else None,
        "elapsed_sec_last": times[-1] if times else None,
    }


def make_result_row(
    q_item: Dict[str, Any],
    rag_out: Dict[str, Any],
    top_k: int,
) -> Dict[str, Any]:
    answer_text = rag_out.get("answer_text", "")
    sources = rag_out.get("sources", [])
    if not isinstance(sources, list):
        sources = [sources]

    citation_stats = detect_citation_stats(answer_text)

    # Heuristique très simple pour "retrieval non vide" :
    # si on a des sources OU une réponse non vide
    retrieval_non_empty = 1 if (sources or (answer_text and answer_text.strip())) else 0

    # Pour annotation manuelle ensuite (dans le CSV)
    # ok / partial / ko
    return {
        "id": q_item.get("id", ""),
        "question": q_item.get("question", ""),
        "expected_type": q_item.get("expected_type", ""),
        "top_k": top_k,
        "elapsed_sec_mean": rag_out.get("elapsed_sec_mean"),
        "elapsed_sec_last": rag_out.get("elapsed_sec_last"),
        "retrieval_non_empty": retrieval_non_empty,
        "answer_len_chars": len(answer_text),
        "has_citation": citation_stats["has_citation"],
        "n_citation_matches": citation_stats["n_citation_matches"],
        "citation_format_hint": citation_stats["citation_format_hint"],
        "n_sources_returned": len(sources),
        "answer_preview": answer_text[:300].replace("\n", " "),
        "manual_label": "",    # à remplir après: ok / partial / ko
        "manual_notes": "",    # notes manuelles
    }


def save_results_csv(rows: List[Dict[str, Any]], out_csv: str) -> None:
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        # Crée un CSV vide avec colonnes minimales si besoin
        fieldnames = ["id", "question", "expected_type", "top_k", "manual_label", "manual_notes"]
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        return

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("Aucun résultat.")
        return

    n = len(rows)
    avg_time = mean([r["elapsed_sec_mean"] for r in rows if r.get("elapsed_sec_mean") is not None])
    pct_citations = 100.0 * sum(int(r.get("has_citation", 0)) for r in rows) / n
    pct_non_empty = 100.0 * sum(int(r.get("retrieval_non_empty", 0)) for r in rows) / n

    print("\nRésumé tests/validation RAG")
    print(f"Questions évaluées       : {n}")
    print(f"Temps moyen (s)          : {avg_time:.3f}")
    print(f"Réponses non vides (%)   : {pct_non_empty:.1f}")
    print(f"Réponses avec citation (%) : {pct_citations:.1f}")


def main() -> None:
    args = parse_args()
    questions = load_questions(args.questions)

    rows: List[Dict[str, Any]] = []
    for q in questions:
        question_text = q["question"]
        rag_out = one_query_with_timing(
            question=question_text,
            top_k=args.top_k,
            runs=max(1, args.runs),
        )
        row = make_result_row(q_item=q, rag_out=rag_out, top_k=args.top_k)
        rows.append(row)

        if args.verbose:
            print(f"\n[{q['id']}] {question_text}")
            print(f"Temps moyen: {row['elapsed_sec_mean']:.3f}s | Citations: {row['has_citation']}")
            print(f"Réponse: {rag_out.get('answer_text', '')[:500]}")

    save_results_csv(rows, args.out_csv)
    print_summary(rows)
    print(f"\nCSV sauvegardé : {args.out_csv}")
    print("Tu peux maintenant remplir 'manual_label' (ok/partial/ko) dans le CSV.")


if __name__ == "__main__":
    main()