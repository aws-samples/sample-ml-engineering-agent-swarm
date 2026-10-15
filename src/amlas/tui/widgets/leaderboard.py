"""Leaderboard table widget — sortable model performance ranking."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import DataTable, Static


class LeaderboardWidget(Static):
    """Sortable leaderboard showing ranked model results."""

    def compose(self) -> ComposeResult:
        yield DataTable(id="leaderboard-table")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Rank", "Algorithm", "f1_macro", "Accuracy", "Duration", "Run ID")
        # Data pre-sorted by f1_macro descending
        table.add_row("🥇 1", "XGBoost", "0.8923", "0.9012", "4.2s", "a1b2c3d4")
        table.add_row("🥈 2", "RandomForest", "0.8756", "0.8891", "3.8s", "e5f6g7h8")
        table.add_row("🥉 3", "GradientBoosting", "0.8621", "0.8745", "5.1s", "i9j0k1l2")
        table.add_row("  4", "LogisticRegression", "0.8134", "0.8301", "0.9s", "m3n4o5p6")
        table.add_row("  5", "SVM", "0.7892", "0.8012", "12.3s", "q7r8s9t0")
        table.cursor_type = "row"
