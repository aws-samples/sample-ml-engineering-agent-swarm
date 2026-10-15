"""TUI Management Agent — interprets natural language and executes commands.

Instead of telling users "use /attach", this agent:
1. Classifies user intent from natural language
2. Extracts parameters (paths, targets, specs)
3. Returns (intent, params) so the engine can EXECUTE the appropriate handler
4. Returns real results — no instructions to "use X command"

Intent Categories:
    load_data    — "upload data", "load dataset", "use this file", "here's my data"
    run_pipeline — "run experiment", "start training", "analyze this"
    check_quality— "check data quality", "run EDA", "how's the data look"
    show_results — "show leaderboard", "best model", "show results"
    create_spec  — "define problem", "create spec", "I want to predict..."
    show_status  — "what's happening", "status", "how's it going"
    research     — "find algorithms", "what approaches work for..."
"""

from __future__ import annotations

import re
from typing import Any

# Intent patterns — ordered by priority (first match wins).
# Each entry: (intent_name, list_of_regex_strings, list_of_entity_names_to_extract)
INTENT_PATTERNS: list[tuple[str, list[str], list[str]]] = [
    (
        "load_data",
        [
            # verb + optional filler + data noun
            r"(?:upload|load|use|attach|import)\s+(?:some|this|my|the|a)?\s*(?:data|dataset|file|csv|parquet)",
            # data/dataset for/from experiment
            r"(?:data|dataset)\s+(?:for|from)\s+(?:an?\s+)?experiment",
            # "i want/need to upload/load data"
            r"i\s+(?:want|need)\s+to\s+(?:upload|load|use|import)\s+(?:some\s+)?data",
            # "here's/here is the/my/some data"
            r"(?:here(?:'s| is)\s+(?:the|my|some)\s+(?:data|dataset|file))",
            # verb + path-like token (catches "load /path" or "load ./path")
            r"(?:upload|load|use|import)\s+(?:[~.]?/|\.\./)[\w./\-]+",
            # verb + relative path with extension
            r"(?:upload|load|use|import)\s+[\w][\w./\-]*\.(?:csv|parquet|json|yaml|xlsx)",
            # "attach" followed by anything (attach is always data-related)
            r"attach\s+(?:my|this|the|some|a)?\s*\w+",
        ],
        ["path"],
    ),
    (
        "run_pipeline",
        [
            r"(?:run|start|execute|begin|kick\s*off)\s+(?:the\s+)?(?:pipeline|experiment|training|analysis|ml)",
            r"(?:analyze|train|classify|predict)\s+(?:this|the|my)",
            r"start\s+training",
        ],
        ["spec", "data"],
    ),
    (
        "check_quality",
        [
            r"(?:check|assess|evaluate|run)\s+(?:data\s+)?quality",
            r"(?:run|do)\s+(?:an?\s+)?(?:eda|exploratory|quality)",
            r"how(?:'s| does)\s+(?:the\s+)?data\s+(?:look|seem|appear)",
            r"data\s+quality",
        ],
        ["path"],
    ),
    (
        "show_results",
        [
            # "show/display/get/see [me] [the] results/leaderboard/..."
            r"(?:show|display|get|see)\s+(?:me\s+)?(?:the\s+)?(?:results|leaderboard|scores|models|rankings?)",
            r"(?:what'?s?|which)\s+(?:the\s+)?best\s+model",
            r"(?:model|experiment)\s+(?:results|scores|rankings?)",
        ],
        [],
    ),
    (
        "create_spec",
        [
            r"(?:define|create|write|make|generate)\s+(?:a\s+)?(?:spec|specification|problem|config)",
            r"i\s+(?:want|need)\s+to\s+(?:predict|classify|detect|forecast|identify)",
        ],
        ["description"],
    ),
    (
        "show_status",
        [
            r"(?:what'?s?|how'?s?)\s+(?:happening|going|the\s+status)",
            r"(?:show|get|check)\s+(?:the\s+)?status",
            r"(?:how\s+is|how's)\s+(?:it|the\s+(?:pipeline|training|experiment))\s+going",
        ],
        [],
    ),
    (
        "research",
        [
            r"(?:find|search|look\s+for|what)\s+(?:algorithms?|approaches?|methods?|techniques?)",
            r"(?:research|recommend|suggest)\s+(?:approaches?|algorithms?|methods?)",
            r"what\s+(?:algorithms?|approaches?|methods?)\s+(?:work|are good)\s+for",
        ],
        ["query"],
    ),
]


class IntentAgent:
    """Classifies natural language intent and extracts parameters.

    Uses compiled regex patterns for fast, local classification (no LLM needed).
    Designed to intercept messages that would otherwise get a "use /command" response
    and instead trigger direct execution.
    """

    def __init__(self) -> None:
        self._compiled: list[tuple[str, list[re.Pattern[str]], list[str]]] = [
            (intent, [re.compile(p, re.IGNORECASE) for p in patterns], entities)
            for intent, patterns, entities in INTENT_PATTERNS
        ]

    def classify(self, text: str) -> tuple[str, dict[str, Any]] | None:
        """Classify user text into an intent with extracted parameters.

        Returns:
            Tuple of (intent_name, extracted_params) if a match is found.
            None if no intent pattern matches the text.
        """
        for intent, patterns, entities in self._compiled:
            for pattern in patterns:
                if pattern.search(text):
                    params = self._extract_params(text, entities)
                    return (intent, params)
        return None

    def _extract_params(self, text: str, entities: list[str]) -> dict[str, Any]:
        """Extract named entities from the text based on intent requirements."""
        params: dict[str, Any] = {}
        if "path" in entities:
            params["path"] = self._extract_path(text)
        if "description" in entities:
            params["description"] = text
        if "query" in entities:
            params["query"] = text
        if "spec" in entities:
            params["spec"] = self._extract_path(text)
        if "data" in entities:
            params["data"] = self._extract_path(text)
        return params

    def _extract_path(self, text: str) -> str | None:
        """Extract a file or directory path from text.

        Handles:
        - Absolute paths: /path/to/project/archive
        - Home-relative: ~/data/train.csv
        - Dot-relative: ./data/train.csv, ../repos/data.csv
        - Relative with extensions: data/ai4i/bogie/archive/dataset1.csv
        - Relative directory paths: data/ai4i/bogie/archive
        """
        # 1. Match home-relative paths (~/...) — most specific prefix
        home_match = re.search(r'(~/(?:[\w.\-]+/)*[\w.\-]+)', text)
        if home_match:
            return home_match.group(1).rstrip(".")

        # 2. Match dot-relative paths (./... or ../...)
        dot_match = re.search(r'(\.\.?/(?:[\w.\-]+/)*[\w.\-]+)', text)
        if dot_match:
            return dot_match.group(1).rstrip(".")

        # 3. Match absolute paths — require whitespace or start-of-string before /
        abs_match = re.search(r'(?:^|\s)(/(?:[\w.\-]+/)*[\w.\-]+)', text)
        if abs_match:
            path = abs_match.group(1).rstrip(".")
            return path

        # 4. Match relative paths with known data extensions
        #    Use negative lookbehind to avoid matching verb prefixes
        ext_match = re.search(
            r'(?<![a-zA-Z])([\w][\w./\-]*\.(?:csv|parquet|json|yaml|yml|xlsx))\b', text
        )
        if ext_match:
            return ext_match.group(1)
        # Fallback without lookbehind — scan for path-like token after space
        ext_match2 = re.search(
            r'\s([\w][\w./\-]*\.(?:csv|parquet|json|yaml|yml|xlsx))', text
        )
        if ext_match2:
            return ext_match2.group(1)

        # 5. Match relative paths that look like directory structures
        #    (contain at least one slash and start with known prefixes)
        dir_match = re.search(
            r'((?:data|repos|specs|output|results|models)/[\w./\-]+)', text
        )
        if dir_match:
            return dir_match.group(1).rstrip(".")

        return None
