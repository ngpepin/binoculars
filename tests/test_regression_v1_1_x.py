import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "binoculars.py"
FIXTURE_ATHENS = REPO_ROOT / "tests" / "fixtures" / "Athens.md"


def load_binoculars_module():
    """Load binoculars.py as a module with a llama_cpp stub when unavailable."""
    if "llama_cpp" not in sys.modules:
        stub = types.ModuleType("llama_cpp")

        class DummyLlama:
            pass

        stub.Llama = DummyLlama
        sys.modules["llama_cpp"] = stub

    spec = importlib.util.spec_from_file_location("binoculars_module", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestRegressionV11X(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binoculars = load_binoculars_module()
        cls.athens_text = FIXTURE_ATHENS.read_text(encoding="utf-8")

    def test_fixture_athens_is_present_for_regression(self):
        self.assertTrue(FIXTURE_ATHENS.is_file(), "Expected tests/fixtures/Athens.md to exist")
        self.assertIn("Night pressed down on Athens", self.athens_text)

    def test_heatmap_notes_table_order_and_columns_v1_1_x(self):
        spans = self.binoculars.split_markdown_paragraph_spans(self.athens_text)
        self.assertGreaterEqual(len(spans), 6, "Fixture must provide at least 6 paragraph spans")

        # Deterministic synthetic profile so this regression test is stable and fast.
        logppls = [3.2, 0.9, 1.8, 2.2, 1.0, 3.0]
        deltas_if_removed = [-0.40, +0.20, -0.10, -0.20, +0.30, -0.05]
        observer_doc_logppl = 2.0

        rows = []
        for i, (span, logppl, delta_if_removed) in enumerate(
            zip(spans[:6], logppls, deltas_if_removed),
            start=1,
        ):
            s, e = span
            rows.append(
                {
                    "paragraph_id": i,
                    "char_start": s,
                    "char_end": e,
                    "token_start": i * 100,
                    "token_end": i * 100 + 25,
                    "transitions": 10 + i,
                    "logPPL": float(logppl),
                    "delta_vs_doc_logPPL": float(logppl - observer_doc_logppl),
                    "delta_doc_logPPL_if_removed": float(delta_if_removed),
                }
            )

        profile = {
            "unit": "paragraph",
            "total_paragraphs": len(spans),
            "paragraphs_with_transitions": len(rows),
            "doc_logPPL": observer_doc_logppl,
            "rows": rows,
        }

        md = self.binoculars.build_heatmap_markdown(
            text=self.athens_text,
            source_label="tests/fixtures/Athens.md",
            paragraph_profile=profile,
            top_k=2,
            observer_logppl=observer_doc_logppl,
            observer_ppl=7.389,
            performer_logppl=2.2,
            performer_ppl=9.025,
            logxppl=2.5,
            xppl=12.182,
            binoculars_score=0.8,
        )

        self.assertIn("## Notes Table", md)
        self.assertIn(
            "| Index | Label | % contribution | Paragraph | logPPL | delta_vs_doc | delta_if_removed | Transitions | Chars | Tokens |",
            md,
        )

        # Selected set with top_k=2: HIGH => paragraph 1,6 ; LOW => paragraph 2,5
        # Notes are numbered in document order: 1,2,5,6.
        self.assertIn("[[#Notes Table|[1]]]", md)
        self.assertIn("[[#Notes Table|[2]]]", md)
        self.assertIn("[[#Notes Table|[3]]]", md)
        self.assertIn("[[#Notes Table|[4]]]", md)

        self.assertIn("| 1 | HIGH | -20.00% | 1 |", md)
        self.assertIn("| 2 | LOW | +10.00% | 2 |", md)
        self.assertIn("| 3 | LOW | +15.00% | 5 |", md)
        self.assertIn("| 4 | HIGH | -2.50% | 6 |", md)

        self.assertLess(md.find("| 1 | HIGH |"), md.find("| 2 | LOW |"))
        self.assertLess(md.find("| 2 | LOW |"), md.find("| 3 | LOW |"))
        self.assertLess(md.find("| 3 | LOW |"), md.find("| 4 | HIGH |"))

    def test_master_profile_resolution_default_and_override_v1_1_x(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fast_cfg = tmp / "fast.json"
            long_cfg = tmp / "long.json"
            master_cfg = tmp / "config.binoculars.json"

            fast_cfg.write_text("{}", encoding="utf-8")
            long_cfg.write_text("{}", encoding="utf-8")
            master_cfg.write_text(
                json.dumps(
                    {
                        "default": "fast",
                        "profiles": {
                            "fast": str(fast_cfg),
                            "long": str(long_cfg),
                        },
                    }
                ),
                encoding="utf-8",
            )

            label, path = self.binoculars.resolve_profile_config_path(str(master_cfg), None)
            self.assertEqual(label, "fast")
            self.assertEqual(path, str(fast_cfg))

            label, path = self.binoculars.resolve_profile_config_path(str(master_cfg), "long")
            self.assertEqual(label, "long")
            self.assertEqual(path, str(long_cfg))

            with self.assertRaises(ValueError):
                self.binoculars.resolve_profile_config_path(str(master_cfg), "unknown")

    def test_master_profile_object_entry_with_max_tokens_override_v1_1_x(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fast_cfg = tmp / "fast.json"
            long_cfg = tmp / "long.json"
            master_cfg = tmp / "config.binoculars.json"

            fast_cfg.write_text("{}", encoding="utf-8")
            long_cfg.write_text("{}", encoding="utf-8")
            master_cfg.write_text(
                json.dumps(
                    {
                        "default": "fast",
                        "profiles": {
                            "fast": {
                                "path": str(fast_cfg),
                                "max_tokens": 1234,
                            },
                            "long": str(long_cfg),
                        },
                    }
                ),
                encoding="utf-8",
            )

            label, path, max_tokens = self.binoculars.resolve_profile_config(str(master_cfg), "fast")
            self.assertEqual(label, "fast")
            self.assertEqual(path, str(fast_cfg))
            self.assertEqual(max_tokens, 1234)

            label, path, max_tokens = self.binoculars.resolve_profile_config(str(master_cfg), "long")
            self.assertEqual(label, "long")
            self.assertEqual(path, str(long_cfg))
            self.assertIsNone(max_tokens)

    def test_markdown_heatmap_strips_hardbreak_backslashes_v1_1_x(self):
        text = "Para one.\\\nQuoted line.\n\nPara two.\\ \"Dialog\"\n\nPara three."
        spans = self.binoculars.split_markdown_paragraph_spans(text)
        self.assertEqual(len(spans), 3)

        rows = []
        logppls = [3.0, 1.0, 2.5]
        deltas = [-0.2, 0.3, -0.1]
        for i, (span, lp, d) in enumerate(zip(spans, logppls, deltas), start=1):
            s, e = span
            rows.append(
                {
                    "paragraph_id": i,
                    "char_start": s,
                    "char_end": e,
                    "token_start": i * 10,
                    "token_end": i * 10 + 5,
                    "transitions": 5,
                    "logPPL": float(lp),
                    "delta_vs_doc_logPPL": float(lp - 2.0),
                    "delta_doc_logPPL_if_removed": float(d),
                }
            )

        profile = {
            "unit": "paragraph",
            "total_paragraphs": 3,
            "paragraphs_with_transitions": 3,
            "doc_logPPL": 2.0,
            "rows": rows,
        }

        md = self.binoculars.build_heatmap_markdown(
            text=text,
            source_label="fixture.md",
            paragraph_profile=profile,
            top_k=1,
            observer_logppl=2.0,
            observer_ppl=7.389,
            performer_logppl=2.2,
            performer_ppl=9.025,
            logxppl=2.5,
            xppl=12.182,
            binoculars_score=0.8,
        )

        self.assertNotIn("\\\n", md)
        self.assertNotIn("\\ \"", md)
        self.assertIn('"Dialog"', md)

    def test_console_heatmap_rendering_format_v1_1_x(self):
        text = "Para one.\\\nQuoted line.\n\nPara two.\\ \"Dialog\"\n\nPara three."
        spans = self.binoculars.split_markdown_paragraph_spans(text)
        self.assertEqual(len(spans), 3)

        rows = []
        logppls = [3.0, 1.0, 2.5]
        deltas = [-0.2, 0.3, -0.1]
        for i, (span, lp, d) in enumerate(zip(spans, logppls, deltas), start=1):
            s, e = span
            rows.append(
                {
                    "paragraph_id": i,
                    "char_start": s,
                    "char_end": e,
                    "token_start": i * 10,
                    "token_end": i * 10 + 5,
                    "transitions": 5,
                    "logPPL": float(lp),
                    "delta_vs_doc_logPPL": float(lp - 2.0),
                    "delta_doc_logPPL_if_removed": float(d),
                }
            )

        profile = {
            "unit": "paragraph",
            "total_paragraphs": 3,
            "paragraphs_with_transitions": 3,
            "doc_logPPL": 2.0,
            "rows": rows,
        }

        old_columns = os.environ.get("COLUMNS")
        os.environ["COLUMNS"] = "100"
        try:
            out = self.binoculars.build_heatmap_output_console(
                text=text,
                source_label="fixture.md",
                paragraph_profile=profile,
                top_k=1,
                observer_logppl=2.0,
                observer_ppl=7.389,
                performer_logppl=2.2,
                performer_ppl=9.025,
                logxppl=2.5,
                xppl=12.182,
                binoculars_score=0.8,
                force_color=False,
            )
        finally:
            if old_columns is None:
                os.environ.pop("COLUMNS", None)
            else:
                os.environ["COLUMNS"] = old_columns

        self.assertIn("Notes Table", out)
        self.assertIn("[1]", out)
        self.assertIn("[2]", out)
        self.assertIn("┌", out)
        self.assertIn("┬", out)
        self.assertIn("└", out)
        self.assertNotIn("[[#Notes Table|", out)
        self.assertNotIn("\\\n", out)
        self.assertNotIn("\\ \"", out)
        self.assertNotIn("\n\n\n", out)
        self.assertNotIn("\x1b[", out)
        for line in out.splitlines():
            self.assertLessEqual(len(line), 85)

    def test_changed_span_helper_v1_1_x(self):
        self.assertEqual(
            self.binoculars.changed_span_in_new_text("abcdef", "abcXdef"),
            (3, 4),
        )
        self.assertEqual(
            self.binoculars.changed_span_in_new_text("abcdef", "abdef"),
            (2, 2),
        )
        self.assertEqual(
            self.binoculars.changed_span_in_new_text("abcdef", "abXYdef"),
            (2, 4),
        )
        self.assertEqual(
            self.binoculars.changed_span_in_new_text("same", "same"),
            (0, 0),
        )

    def test_spellcheck_helpers_v1_1_x(self):
        dictionary = {
            "hello",
            "world",
            "and",
            "don't",
            "co",
            "operate",
            "state",
            "of",
            "the",
            "art",
            "alpha",
            "beta",
        }

        self.assertTrue(self.binoculars.is_word_spelled_correctly("hello", dictionary))
        self.assertTrue(self.binoculars.is_word_spelled_correctly("DON'T", dictionary))
        self.assertTrue(self.binoculars.is_word_spelled_correctly("state-of-the-art", dictionary))
        self.assertTrue(self.binoculars.is_word_spelled_correctly("alpha/beta", dictionary))
        self.assertTrue(self.binoculars.is_word_spelled_correctly("logPPL", dictionary))

        self.assertFalse(self.binoculars.is_word_spelled_correctly("helo", dictionary))
        self.assertFalse(self.binoculars.is_word_spelled_correctly("wrld", dictionary))

        spans = self.binoculars.find_misspelled_spans("hello wrld and helo", dictionary)
        words = ["hello wrld and helo"[s:e] for s, e in spans]
        self.assertEqual(words, ["wrld", "helo"])


if __name__ == "__main__":
    unittest.main()
