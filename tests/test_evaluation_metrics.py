from evaluation.metrics import (
    exact_match,
    numeric_match,
    retrieval_metrics,
    source_matches,
    token_f1,
)


def test_exact_and_token_match_accept_normalized_text() -> None:
    assert exact_match("Answer  value", "answer value") == 1.0
    assert token_f1("alpha beta", "beta alpha") == 1.0


def test_numeric_match_requires_expected_numbers() -> None:
    correct = numeric_match("Pressure is 0.25 MPa", "0.25 MPa")
    incorrect = numeric_match("Pressure is 0.2 MPa", "0.25 MPa")
    assert correct["both_correct"]
    assert not incorrect["both_correct"]


def test_retrieval_metrics_calculate_ranked_scores() -> None:
    metrics = retrieval_metrics([[True, False], [False, True]], 2)
    assert metrics["hit_rate"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["mrr"] == 0.75


def test_source_matching_prefers_document_id_then_path() -> None:
    result = {"document_id": "doc-1", "document_path": "different.docx"}
    expected = [{"document_id": "doc-1", "document_path": "expected.docx"}]
    assert source_matches(result, expected)
