"""
Métriques d'évaluation pour le pipeline efficient-llm-pipeline.

Fonctions principales :
    extract_answer(text)       — extrait la réponse numérique après '####'
    is_correct(pred, expected) — compare deux réponses GSM8K
    evaluate_batch(responses, expected) — accuracy sur un batch
    benchmark_summary(baseline, turboquant) — tableau comparatif
"""

import re


def extract_answer(text: str) -> str | None:
    """
    Extrait la réponse numérique finale d'une solution GSM8K.

    Le format GSM8K termine toujours par '#### <réponse>'.
    On normalise les séparateurs de milliers et les espaces.

    Args:
        text: réponse complète du modèle (avec CoT)
    Returns:
        réponse normalisée (str) ou None si pas trouvée

    Examples:
        >>> extract_answer("...calculs... #### 42")
        '42'
        >>> extract_answer("#### 1,234")
        '1234'
        >>> extract_answer("pas de réponse ici")
        None
    """
    if "####" not in text:
        return None

    after = text.split("####")[-1].strip()

    # Supprimer les séparateurs de milliers (1,234 → 1234)
    after = after.replace(",", "")

    # Garder uniquement le premier token numérique (ignorer unités, texte après)
    match = re.search(r"-?\d+\.?\d*", after)
    return match.group(0) if match else None


def is_correct(pred: str, expected: str) -> bool:
    """
    Compare deux réponses GSM8K après normalisation.

    Args:
        pred    : réponse prédite (brute ou déjà extraite)
        expected: réponse attendue (brute ou déjà extraite)
    Returns:
        True si les réponses sont identiques après normalisation
    """
    # Extraire si contient '####', sinon utiliser tel quel
    p = extract_answer(pred) if "####" in pred else pred.strip().replace(",", "")
    e = extract_answer(expected) if "####" in expected else expected.strip().replace(",", "")

    if p is None or e is None:
        return False

    # Comparaison numérique (tolère 1e-6 pour les flottants)
    try:
        return abs(float(p) - float(e)) < 1e-6
    except ValueError:
        return p == e


def evaluate_batch(responses: list[str], expected: list[str]) -> dict:
    """
    Calcule les métriques d'accuracy sur un batch de réponses.

    Args:
        responses: liste de réponses du modèle
        expected : liste de réponses attendues (même ordre)
    Returns:
        dict avec :
            accuracy     (float) — taux de bonnes réponses
            correct      (int)   — nombre de bonnes réponses
            total        (int)   — nombre total de questions
            no_answer    (int)   — réponses sans '####'
            results      (list)  — bool par exemple

    Examples:
        >>> evaluate_batch(["#### 42", "#### 10"], ["42", "9"])
        {'accuracy': 0.5, 'correct': 1, 'total': 2, 'no_answer': 0, 'results': [True, False]}
    """
    assert len(responses) == len(expected), "responses et expected doivent avoir la même taille"

    results = [is_correct(r, e) for r, e in zip(responses, expected, strict=True)]
    no_answer = sum(1 for r in responses if extract_answer(r) is None)

    return {
        "accuracy": sum(results) / len(results),
        "correct": sum(results),
        "total": len(results),
        "no_answer": no_answer,
        "results": results,
    }


def benchmark_summary(baseline: dict, turboquant: dict) -> str:
    """
    Génère un tableau comparatif lisible baseline vs TurboQuant.

    Args:
        baseline   : dict avec keys accuracy, avg_time, avg_vram
        turboquant : dict avec keys accuracy, avg_time, avg_vram, compression_ratio
    Returns:
        str formaté pour affichage notebook/terminal
    """
    acc_delta = turboquant["accuracy"] - baseline["accuracy"]
    time_delta = turboquant["avg_time"] - baseline["avg_time"]  # noqa: F841
    vram_delta = turboquant["avg_vram"] - baseline["avg_vram"]  # noqa: F841

    lines = [
        "=" * 55,
        f"{'BENCHMARK -- Baseline vs TurboQuant':^55}",
        "=" * 55,
        f"{'':25} {'Baseline':>12} {'TurboQuant':>12}",
        "-" * 55,
        f"{'Accuracy GSM8K':25} {baseline['accuracy'] * 100:>11.1f}% {turboquant['accuracy'] * 100:>11.1f}%",
        f"{'Delta accuracy':25} {'':>12} {acc_delta * 100:>+11.1f}%",
        f"{'Temps moyen (s)':25} {baseline['avg_time']:>12.1f} {turboquant['avg_time']:>12.1f}",
        f"{'VRAM peak (GB)':25} {baseline['avg_vram']:>12.2f} {turboquant['avg_vram']:>12.2f}",
        f"{'Compression KV cache':25} {'1x':>12} {turboquant['compression_ratio']:>11.1f}x",
        "=" * 55,
    ]
    return "\n".join(lines)
