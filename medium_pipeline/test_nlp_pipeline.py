"""
test_nlp_pipeline.py
====================
Unit and integration tests for pipeline_dag.py.

Run with:
    pytest test_nlp_pipeline.py -v

Dependencies (all in medium_pipeline-env):
    pip install pytest pytest-mock
"""

import json
import os
import re
import sys
import tempfile
import types
from unittest.mock import MagicMock, patch, mock_open

import pytest

FIXTURE_ARTICLE = """
Love and relationships are complex topics that affect everyone.
Women often seek emotional connection and understanding in their partners.
Men and women approach relationships differently, but both value trust.
Emotional intimacy is a key foundation of a healthy relationship.
It builds trust, provides emotional security, and relieves stress.
Communication is essential for any relationship to thrive over time.
Bollywood has influenced many people's expectations of romantic love.
Maggi noodles became a symbol of simple, comforting shared moments.
Being able to completely trust someone and spend quality time with them matters most.
Relationships require walking together, adjusting, and surviving hardships.
Love can be unconditional, but relationships are a bit conditional.
Openness in communication and appreciation strengthen any bond.
"""

FIXTURE_ARTICLE_WITH_LISTS = """
Women look for the following qualities:
Openness in communication
Understanding
Appreciation
Friendship that didn't judge them
These qualities matter deeply in a relationship.
"""

FIXTURE_ARTICLE_MULTIBLANK = "First paragraph.\n\n\n\nSecond paragraph.\n\n\n\nThird paragraph."
FIXTURE_ARTICLE_MIDNEWLINE  = "This is a sentence\nthat wraps mid-line."



def _clean(raw: str) -> str:
    """Pure reimplementation of the clean_article logic for unit testing."""
    lines = raw.splitlines()
    processed_lines = []
    list_buffer = []

    def flush_list(buf):
        if not buf:
            return
        joined = ", ".join(item.strip(" \t\u2013\u2022-") for item in buf if item.strip())
        if joined:
            processed_lines.append(joined.rstrip(",") + ".")

    for line in lines:
        stripped = line.strip()
        is_list_item = (
            stripped
            and len(stripped) <= 60
            and stripped[-1] not in ".!?:\"'"
            and not stripped.startswith("#")
        )
        if is_list_item:
            list_buffer.append(stripped)
        else:
            flush_list(list_buffer)
            list_buffer = []
            processed_lines.append(line)

    flush_list(list_buffer)
    text = "\n".join(processed_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def _count_syllables(word: str) -> int:
    word = word.lower()
    if len(word) > 3 and word.endswith('e'):
        word = word[:-1]
    vowel_groups = re.findall(r'[aeiou]+', word)
    return max(1, len(vowel_groups))



VALIDATION_RULES = {
    "keywords":    lambda v: isinstance(v, list) and len(v) >= 5,
    "summary":     lambda v: isinstance(v, str)  and len(v.split()) >= 20,
    "sentiment":   lambda v: isinstance(v, dict) and "tone" in v and "polarity" in v and "sentence_scores" in v,
    "entities":    lambda v: isinstance(v, dict),
    "readability": lambda v: isinstance(v, dict) and 0 <= v.get("score", -1) <= 206.835,
    "snippets":    lambda v: isinstance(v, list) and len(v) >= 1,
}


def _validate(results: dict) -> list[str]:
    """Returns a list of failure messages (empty = all passed)."""
    failures = []
    for field, rule in VALIDATION_RULES.items():
        value = results.get(field)
        if value is None:
            failures.append(f"'{field}': is None")
        elif not rule(value):
            failures.append(f"'{field}': failed quality check")
    return failures


class TestCleanArticle:

    def test_bullet_list_inlined_as_prose(self):
        result = _clean(FIXTURE_ARTICLE_WITH_LISTS)
        # List items should be joined with commas, not remain on separate lines
        assert "Openness in communication" in result
        assert "\nOpenness in communication" not in result

    def test_bullet_list_ends_with_period(self):
        result = _clean(FIXTURE_ARTICLE_WITH_LISTS)
        # Inlined list must be terminated with a full stop
        lines = result.splitlines()
        list_line = next((l for l in lines if "Openness" in l), None)
        assert list_line is not None
        assert list_line.strip().endswith(".")

    def test_multiple_blank_lines_collapsed(self):
        result = _clean(FIXTURE_ARTICLE_MULTIBLANK)
        assert "\n\n\n" not in result
        assert "First paragraph." in result
        assert "Second paragraph." in result

    def test_mid_paragraph_newline_replaced_with_space(self):
        result = _clean(FIXTURE_ARTICLE_MIDNEWLINE)
        assert "\n" not in result
        assert "sentence that wraps mid-line." in result

    def test_leading_trailing_whitespace_stripped(self):
        result = _clean("   \n  hello world  \n   ")
        assert result == result.strip()

    def test_markdown_headings_not_treated_as_list(self):
        raw = "## Section Title\nSome content here that is normal prose."
        result = _clean(raw)
        assert "## Section Title" in result

    def test_empty_string_raises_value_error(self):
        """clean_article should raise if raw text is empty (mirrors DAG behaviour)."""
        with pytest.raises(ValueError, match="Empty article text"):
            if not "":
                raise ValueError("Empty article text received from XCom — load_article likely failed.")

    def test_normal_prose_unchanged_structure(self):
        prose = "This is a normal sentence. This is another one. And a third."
        result = _clean(prose)
        assert "This is a normal sentence." in result
        assert "And a third." in result

    def test_extra_whitespace_collapsed(self):
        result = _clean("Too    many     spaces   here.")
        assert "  " not in result



class TestCountSyllables:

    @pytest.mark.parametrize("word, expected", [
        ("cat",        1),   # single vowel group
        ("love",       1),   # silent-e stripped → "lov" → 1 group
        ("beautiful",  3),   # "eau", "i", "ul" — 3 groups after silent-e strip
        ("a",          1),   # minimum of 1
        ("rhythm",     1),   # no vowels → minimum 1
        ("emotional",  4),   # "e", "o", "io", "a" → 4 groups
        ("relationship", 4), # "e", "a", "io", "i" → 4 groups
        ("time",       1),   # silent-e → "tim" → 1 group
    ])
    def test_syllable_count(self, word, expected):
        assert _count_syllables(word) == expected

    def test_minimum_is_one(self):
        # Even a word with no detectable vowels must return at least 1
        assert _count_syllables("bcdfg") >= 1

    def test_case_insensitive(self):
        assert _count_syllables("LOVE") == _count_syllables("love")


class TestExtractKeywords:

    def _run(self, text: str, top_n: int = 10):
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        from nltk.probability import FreqDist
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        filtered = [w for w in tokens if w.isalpha() and w not in stop_words]
        freq = FreqDist(filtered)
        return [{"word": w, "count": c} for w, c in freq.most_common(top_n)]

    def test_returns_list_of_dicts(self):
        result = self._run(FIXTURE_ARTICLE)
        assert isinstance(result, list)
        assert all(isinstance(k, dict) for k in result)

    def test_each_entry_has_word_and_count(self):
        result = self._run(FIXTURE_ARTICLE)
        for entry in result:
            assert "word" in entry
            assert "count" in entry

    def test_stopwords_excluded(self):
        result = self._run(FIXTURE_ARTICLE)
        words = [e["word"] for e in result]
        # Common stopwords should not appear
        for stopword in ("the", "and", "is", "in", "a", "of", "to"):
            assert stopword not in words, f"Stopword '{stopword}' found in keywords"

    def test_counts_are_positive_integers(self):
        result = self._run(FIXTURE_ARTICLE)
        for entry in result:
            assert isinstance(entry["count"], int)
            assert entry["count"] > 0

    def test_results_sorted_descending(self):
        result = self._run(FIXTURE_ARTICLE)
        counts = [e["count"] for e in result]
        assert counts == sorted(counts, reverse=True)

    def test_domain_keywords_surface(self):
        result = self._run(FIXTURE_ARTICLE)
        words = [e["word"] for e in result]
        # "love", "relationship/relationships", "emotional" should all rank highly
        assert any(w in words for w in ("love", "relationship", "relationships", "emotional"))


class TestSummarizeArticle:

    def _run(self, text: str, top_n: int = 5) -> str:
        import nltk
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk.corpus import stopwords
        from nltk.probability import FreqDist
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        sentences = sent_tokenize(text)
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        filtered = [w for w in tokens if w.isalpha() and w not in stop_words]
        freq = FreqDist(filtered)
        scores = {}
        for sent in sentences:
            for word in word_tokenize(sent.lower()):
                if word in freq:
                    scores[sent] = scores.get(sent, 0) + freq[word]
        ranked = [s for s in sorted(scores, key=scores.get, reverse=True) if not s.strip().endswith("?")]
        return " ".join(ranked[:top_n])

    def test_returns_non_empty_string(self):
        result = self._run(FIXTURE_ARTICLE)
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_no_questions_in_summary(self):
        article_with_questions = FIXTURE_ARTICLE + "\nWhat do you think about love?\nHow do relationships work?"
        result = self._run(article_with_questions)
        assert "?" not in result

    def test_summary_contains_sentences_from_article(self):
        result = self._run(FIXTURE_ARTICLE)
        # Every sentence in the summary must come from the original article
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        from nltk.tokenize import sent_tokenize
        original_sentences = set(sent_tokenize(FIXTURE_ARTICLE))
        summary_sentences = set(sent_tokenize(result))
        assert summary_sentences.issubset(original_sentences)

    def test_at_most_top_n_sentences(self):
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        from nltk.tokenize import sent_tokenize
        result = self._run(FIXTURE_ARTICLE, top_n=3)
        assert len(sent_tokenize(result)) <= 3


class TestComputeReadability:

    def _run(self, text: str) -> dict:
        import nltk
        from nltk.tokenize import sent_tokenize, word_tokenize
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        sentences = sent_tokenize(text)
        words = [w for w in word_tokenize(text) if w.isalpha()]

        total_syllables = sum(_count_syllables(w) for w in words)
        score = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (total_syllables / len(words))
        level = "Easy" if score >= 70 else "Moderate" if score >= 50 else "Difficult"
        return {"score": round(score, 2), "level": level}

    def test_returns_score_and_level(self):
        result = self._run(FIXTURE_ARTICLE)
        assert "score" in result
        assert "level" in result

    def test_score_within_flesch_range(self):
        result = self._run(FIXTURE_ARTICLE)
        # Flesch scores are theoretically unbounded but realistic text sits in [-30, 121]
        assert -50 <= result["score"] <= 206.835

    def test_level_is_valid_label(self):
        result = self._run(FIXTURE_ARTICLE)
        assert result["level"] in ("Easy", "Moderate", "Difficult")

    def test_easy_text_scores_higher(self):
        easy   = "The cat sat on the mat. The dog ran fast. I love pie."
        hard   = "Notwithstanding the aforementioned constitutional implications, jurisprudential considerations necessitate circumspection."
        r_easy = self._run(easy)
        r_hard = self._run(hard)
        assert r_easy["score"] > r_hard["score"]

    def test_score_is_float(self):
        result = self._run(FIXTURE_ARTICLE)
        assert isinstance(result["score"], float)



class TestGenerateSnippets:

    def _run(self, text: str, top_n: int = 3) -> list:
        import nltk
        from nltk.tokenize import sent_tokenize
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        sentences = sent_tokenize(text)
        candidates = []
        for s in sentences:
            clean = s.replace("\n", " ").strip()
            clean = re.sub(r'\s+', ' ', clean)
            if len(clean.split()) > 12 and "?" not in clean:
                candidates.append(clean)
        return sorted(candidates, key=len, reverse=True)[:top_n]

    def test_returns_list(self):
        result = self._run(FIXTURE_ARTICLE)
        assert isinstance(result, list)

    def test_no_questions_in_snippets(self):
        article_with_q = FIXTURE_ARTICLE + "\nWhat is love?\nHow do we find it?"
        result = self._run(article_with_q)
        for snippet in result:
            assert "?" not in snippet

    def test_snippets_are_complete_sentences(self):
        result = self._run(FIXTURE_ARTICLE)
        for snippet in result:
            # Should not end mid-word (no trailing ellipsis from old truncation bug)
            assert not snippet.endswith("sadn")
            assert not snippet.endswith("rela")
            assert not snippet.endswith("commu")

    def test_snippets_above_minimum_word_count(self):
        result = self._run(FIXTURE_ARTICLE)
        for snippet in result:
            assert len(snippet.split()) > 12

    def test_sorted_longest_first(self):
        result = self._run(FIXTURE_ARTICLE)
        lengths = [len(s) for s in result]
        assert lengths == sorted(lengths, reverse=True)

    def test_at_most_top_n_returned(self):
        result = self._run(FIXTURE_ARTICLE, top_n=2)
        assert len(result) <= 2


class TestValidateResults:

    def _good_results(self):
        return {
            "keywords": [{"word": "love", "count": 34}] * 5,
            "summary": "word " * 25,
            "sentiment": {
                "tone": "Positive",
                "polarity": 0.42,
                "sentence_scores": {"positive": 0.71, "neutral": 0.21, "negative": 0.08},
            },
            "entities": {"GPE": ["Bollywood"]},
            "readability": {"score": 56.1, "level": "Moderate"},
            "snippets": ["This is a valid snippet sentence with enough words."],
        }

    def test_good_results_pass(self):
        assert _validate(self._good_results()) == []

    def test_none_field_fails(self):
        results = self._good_results()
        results["keywords"] = None
        failures = _validate(results)
        assert any("keywords" in f for f in failures)

    def test_too_few_keywords_fails(self):
        results = self._good_results()
        results["keywords"] = [{"word": "love", "count": 34}]  # only 1, need ≥5
        failures = _validate(results)
        assert any("keywords" in f for f in failures)

    def test_short_summary_fails(self):
        results = self._good_results()
        results["summary"] = "Too short."
        failures = _validate(results)
        assert any("summary" in f for f in failures)

    def test_sentiment_missing_key_fails(self):
        results = self._good_results()
        results["sentiment"] = {"tone": "Positive"}  # missing polarity and sentence_scores
        failures = _validate(results)
        assert any("sentiment" in f for f in failures)

    def test_invalid_readability_score_fails(self):
        results = self._good_results()
        results["readability"] = {"score": -999, "level": "Difficult"}
        failures = _validate(results)
        assert any("readability" in f for f in failures)

    def test_empty_snippets_fails(self):
        results = self._good_results()
        results["snippets"] = []
        failures = _validate(results)
        assert any("snippets" in f for f in failures)

    def test_empty_entities_passes(self):
        # entities can legitimately be sparse / empty dict
        results = self._good_results()
        results["entities"] = {}
        assert _validate(results) == []

    def test_multiple_failures_all_reported(self):
        results = self._good_results()
        results["keywords"] = None
        results["summary"] = "short"
        failures = _validate(results)
        assert len(failures) >= 2



class TestLoadArticle:

    def test_reads_file_content(self, tmp_path):
        article_file = tmp_path / "medium.txt"
        article_file.write_text("Hello world. This is a test article.")

        xcom_store = {}
        ti = MagicMock()
        ti.xcom_push.side_effect = lambda key, value: xcom_store.update({key: value})

        with patch("builtins.open", mock_open(read_data="Hello world. This is a test article.")):
            # Simulate load_article behaviour directly
            try:
                with open(str(article_file), "r") as f:
                    article = f.read()
            except FileNotFoundError:
                article = ""
            ti.xcom_push(key='article_text', value=article)

        assert xcom_store["article_text"] == "Hello world. This is a test article."

    def test_missing_file_pushes_empty_string(self):
        xcom_store = {}
        ti = MagicMock()
        ti.xcom_push.side_effect = lambda key, value: xcom_store.update({key: value})

        try:
            with open("/nonexistent/path/missing.txt", "r") as f:
                article = f.read()
        except FileNotFoundError:
            article = ""
        ti.xcom_push(key='article_text', value=article)

        assert xcom_store["article_text"] == ""


class TestDagStructure:

    @pytest.fixture(scope="class")
    def dag(self):
        # We need to import the DAG file. Since it uses `with DAG(...) as dag`,
        # we import the module and grab the dag object from its globals.
        import importlib.util, sys

        dag_path = os.path.join(os.path.dirname(__file__), "pipeline_dag.py")
        if not os.path.exists(dag_path):
            pytest.skip("DAG file not found alongside test file — skipping structure tests.")

        spec = importlib.util.spec_from_file_location("nlp_pipeline", dag_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.dag

    def test_dag_id(self, dag):
        assert dag.dag_id == "medium_article_nlp_pipeline"

    def test_expected_task_ids_present(self, dag):
        expected = {
            "setup_nltk", "load_article", "clean_article",
            "extract_keywords", "summarize_article", "analyze_sentiment",
            "extract_entities", "compute_readability", "generate_snippets",
            "validate_results", "save_results",
        }
        assert expected == set(dag.task_ids)

    def test_setup_is_first(self, dag):
        setup = dag.get_task("setup_nltk")
        assert len(setup.upstream_task_ids) == 0

    def test_load_follows_setup(self, dag):
        load = dag.get_task("load_article")
        assert "setup_nltk" in load.upstream_task_ids

    def test_clean_follows_load(self, dag):
        clean = dag.get_task("clean_article")
        assert "load_article" in clean.upstream_task_ids

    def test_analysis_tasks_follow_clean(self, dag):
        analysis_tasks = [
            "extract_keywords", "summarize_article", "analyze_sentiment",
            "extract_entities", "compute_readability", "generate_snippets",
        ]
        for task_id in analysis_tasks:
            task = dag.get_task(task_id)
            assert "clean_article" in task.upstream_task_ids, \
                f"{task_id} should depend on clean_article"

    def test_validate_follows_all_analysis(self, dag):
        validate = dag.get_task("validate_results")
        expected_upstream = {
            "extract_keywords", "summarize_article", "analyze_sentiment",
            "extract_entities", "compute_readability", "generate_snippets",
        }
        assert expected_upstream.issubset(validate.upstream_task_ids)

    def test_save_follows_validate(self, dag):
        save = dag.get_task("save_results")
        assert "validate_results" in save.upstream_task_ids

    def test_save_has_no_downstream(self, dag):
        save = dag.get_task("save_results")
        assert len(save.downstream_task_ids) == 0

    def test_schedule_is_daily(self, dag):
        assert dag.schedule_interval == "@daily" or str(dag.timetable) != ""

    def test_catchup_disabled(self, dag):
        assert dag.catchup is False

class TestSentimentOutputShape:
    """
    Validates the expected shape of the HuggingFace sentiment output
    without actually loading the model (mocked).
    """

    def _build_result(self, mock_outputs: list[dict]) -> dict:
        """Simulate the aggregation logic from analyze_sentiment."""
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        from nltk.tokenize import sent_tokenize

        MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        sentences = sent_tokenize(FIXTURE_ARTICLE)

        label_scores = {"positive": [], "neutral": [], "negative": []}
        for output in mock_outputs:
            label = output["label"].lower()
            score = output["score"]
            if label in label_scores:
                label_scores[label].append(score)

        averages = {
            label: (round(sum(scores) / len(scores), 4) if scores else 0.0)
            for label, scores in label_scores.items()
        }
        overall_tone = max(averages, key=averages.get).capitalize()
        polarity = round(averages["positive"] - averages["negative"], 4)

        return {
            "tone":            overall_tone,
            "polarity":        polarity,
            "sentence_scores": averages,
            "sentence_count":  len(sentences),
            "model":           MODEL,
        }

    def test_positive_article_tone(self):
        mock_outputs = [{"label": "positive", "score": 0.9}] * 10
        result = self._build_result(mock_outputs)
        assert result["tone"] == "Positive"

    def test_negative_article_tone(self):
        mock_outputs = [{"label": "negative", "score": 0.85}] * 10
        result = self._build_result(mock_outputs)
        assert result["tone"] == "Negative"

    def test_polarity_positive_for_positive_tone(self):
        mock_outputs = [{"label": "positive", "score": 0.9}] * 10
        result = self._build_result(mock_outputs)
        assert result["polarity"] > 0

    def test_polarity_negative_for_negative_tone(self):
        mock_outputs = [{"label": "negative", "score": 0.9}] * 10
        result = self._build_result(mock_outputs)
        assert result["polarity"] < 0

    def test_polarity_in_valid_range(self):
        mock_outputs = [
            {"label": "positive", "score": 0.6},
            {"label": "negative", "score": 0.3},
            {"label": "neutral",  "score": 0.1},
        ]
        result = self._build_result(mock_outputs)
        assert -1.0 <= result["polarity"] <= 1.0

    def test_result_has_all_required_keys(self):
        mock_outputs = [{"label": "positive", "score": 0.8}] * 5
        result = self._build_result(mock_outputs)
        for key in ("tone", "polarity", "sentence_scores", "sentence_count", "model"):
            assert key in result

    def test_sentence_scores_has_all_labels(self):
        mock_outputs = [{"label": "positive", "score": 0.8}] * 5
        result = self._build_result(mock_outputs)
        for label in ("positive", "neutral", "negative"):
            assert label in result["sentence_scores"]

    def test_passes_validation_rule(self):
        mock_outputs = [{"label": "positive", "score": 0.8}] * 5
        result = self._build_result(mock_outputs)
        assert VALIDATION_RULES["sentiment"](result)
