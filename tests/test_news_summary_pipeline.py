"""Offline regression checks: no OpenRouter calls or project data writes."""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import config

spec = importlib.util.spec_from_file_location("news_summary", ROOT / "scripts/04b-summarize_news.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.news_path = Path(self.temp.name) / "news.csv"
        self.out = Path(self.temp.name) / "summaries.csv"
        self.personas = pd.DataFrame([
            {"persona_id": "AL_1", "state": "AL", "persona_text": "An Alabama persona."},
            {"persona_id": "AK_1", "state": "AK", "persona_text": "An Alaska persona."},
        ])
        pd.DataFrame([
            {"url": f"https://example.com/{group}/{i}", "title": f"Story {i}",
             "outlet": "Example", "diet": group, "published_at_utc": "2026-10-26T12:00:00Z",
             "prompt_paragraph": f"A supported policy development {group} {i}.",
             "llm_status": "keep", "national": True, "election_relevant": True}
            for group in module.GROUPS for i in range(5)
        ]).to_csv(self.news_path, index=False)
        self.news = module.load_news(self.news_path)

    def plan(self):
        return module.make_plan(self.personas, self.news, "2026-10-27", 2026)

    def test_nine_distinct_items_three_per_group_and_reproducible(self):
        plan = self.plan()
        repeated = self.plan()
        self.assertEqual(
            [{k: v for k, v in row.items() if k != "generated_at_utc"} for row in plan],
            [{k: v for k, v in row.items() if k != "generated_at_utc"} for row in repeated],
        )
        for row in plan:
            articles = json.loads(row["articles_json"])
            self.assertEqual(len({a["url"] for a in articles}), 9)
            self.assertEqual(pd.Series(a["diet"] for a in articles).value_counts().to_dict(),
                             {g: 3 for g in module.GROUPS})
        self.assertNotEqual(plan[0]["articles_json"], plan[1]["articles_json"])

    def test_shortage_fails_instead_of_repeating_articles(self):
        raw = pd.read_csv(self.news_path)
        raw = raw.drop(raw[raw.diet.eq("left_leaning")].index[:3])
        raw.to_csv(self.news_path, index=False)
        with self.assertRaisesRegex(ValueError, "at least 3"):
            module.load_news(self.news_path)

    def test_preserves_all_nine_paragraphs_without_compression(self):
        plan = self.plan()
        for row in plan:
            articles = json.loads(row["articles_json"])
            expected = "\n\n".join(f"[{i}] {a['prompt_paragraph']}" for i, a in enumerate(articles, 1))
            self.assertEqual(row["summary"], expected)
            self.assertEqual(row["summary_method"], "concatenate_v1")
        articles = json.loads(plan[0]["articles_json"])
        articles[0]["prompt_paragraph"] = "long existing summary " * 400
        self.assertIn(articles[0]["prompt_paragraph"], module.concatenate_summaries(articles))

    def test_main_needs_no_key_and_dry_run_does_not_write(self):
        argv = ["04b", "--out", str(self.out), "--run", "2026-10-27"]
        with patch.object(config, "NEWS_FILE", self.news_path), \
             patch.object(config, "read_table", return_value=self.personas), \
             patch.object(config, "get_api_key", side_effect=AssertionError("No API key should be read")), \
             patch.object(sys, "argv", argv + ["--dry-run"]), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(module.main(), 0)
            self.assertFalse(self.out.exists())
            with patch.object(sys, "argv", argv):
                self.assertEqual(module.main(), 0)
                first = pd.read_csv(self.out)
                self.assertEqual(module.main(), 0)
                second = pd.read_csv(self.out)
        self.assertEqual(len(first), 2)
        self.assertEqual(first["summary"].tolist(), second["summary"].tolist())
        self.assertEqual(first["request_hash"].tolist(), second["request_hash"].tolist())

    def test_08_rejects_unfinished_and_stale_summaries(self):
        # Load just the pure validation function, not script 08's build/write steps.
        tree = ast.parse((ROOT / "scripts/08-build_prompts.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_summaries")
        import hashlib, os
        ns = {"config": config, "pd": pd, "json": json, "hashlib": hashlib, "os": os}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "08", "exec"), ns)
        plan = self.plan()
        module.save(self.out, plan)
        with patch.object(config, "NEWS_FILE", self.news_path), \
             patch.object(config, "NEWS_SUMMARIES_FILE", self.out), patch.dict(os.environ, {"RUN": "2026-10-27"}):
            self.assertEqual(len(ns["load_summaries"](self.personas)), 2)
            changed = self.personas.copy()
            changed.loc[0, "persona_text"] = "Changed profile"
            with self.assertRaisesRegex(ValueError, "Stale"):
                ns["load_summaries"](changed)
            plan[0]["summary_method"] = "llm"
            module.save(self.out, plan)
            with self.assertRaisesRegex(ValueError, "Old LLM"):
                ns["load_summaries"](self.personas)
            plan[0]["summary_method"] = "concatenate_v1"
            plan[0]["status"] = "pending"
            module.save(self.out, plan)
            with self.assertRaisesRegex(ValueError, "Unfinished"):
                ns["load_summaries"](self.personas)


if __name__ == "__main__":
    unittest.main()
