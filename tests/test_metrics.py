from src.evaluate.metrics import evaluate_batch, extract_answer, is_correct


class TestExtractAnswer:
    def test_standard(self):
        assert extract_answer("some reasoning #### 42") == "42"

    def test_thousands_separator(self):
        assert extract_answer("#### 1,234") == "1234"

    def test_negative(self):
        assert extract_answer("#### -5") == "-5"

    def test_float(self):
        assert extract_answer("#### 3.14") == "3.14"

    def test_no_marker(self):
        assert extract_answer("no answer here") is None

    def test_empty_after_marker(self):
        assert extract_answer("#### ") is None

    def test_text_after_number(self):
        assert extract_answer("#### 42 dollars") == "42"


class TestIsCorrect:
    def test_exact_match(self):
        assert is_correct("#### 42", "42") is True

    def test_mismatch(self):
        assert is_correct("#### 42", "43") is False

    def test_float_tolerance(self):
        assert is_correct("#### 3.0", "3") is True

    def test_both_with_markers(self):
        assert is_correct("#### 10", "#### 10") is True

    def test_no_pred(self):
        assert is_correct("no answer", "42") is False


class TestEvaluateBatch:
    def test_mixed_batch(self):
        responses = ["#### 42", "#### 10", "no answer"]
        expected = ["42", "10", "5"]
        result = evaluate_batch(responses, expected)
        assert result["accuracy"] == 2 / 3
        assert result["correct"] == 2
        assert result["total"] == 3
        assert result["no_answer"] == 1
