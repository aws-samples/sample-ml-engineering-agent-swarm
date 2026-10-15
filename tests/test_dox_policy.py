"""The recursive DOX contract is a blocking, dependency-free first CI check."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DoxPolicyTests(unittest.TestCase):
    def test_gitlab_runs_dox_before_verify_without_dependency_setup(self):
        text = (ROOT / ".gitlab-ci.yml").read_text()
        self.assertLess(text.index("  - dox"), text.index("  - verify"))
        block = text.split("dox:\n", 1)[1].split("\nverify:", 1)[0]
        self.assertIn("before_script: []", block)
        self.assertIn("python scripts/dox_lint.py . --recursive", block)
        self.assertNotIn("allow_failure", block)
        self.assertNotIn("rules:", block)

    def test_github_checks_dox_before_dependency_install(self):
        text = (ROOT / ".github/workflows/public-release.yml").read_text()
        self.assertLess(text.index("python scripts/dox_lint.py . --recursive"), text.index("python -m pip install"))
        self.assertNotIn("continue-on-error", text)

    def test_root_alias_and_framework_are_present(self):
        self.assertTrue((ROOT / "CLAUDE.md").is_symlink())
        self.assertEqual("AGENTS.md", (ROOT / "CLAUDE.md").readlink().as_posix())
        self.assertIn("# DOX framework", (ROOT / "AGENTS.md").read_text())
