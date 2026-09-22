from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("evaluation.elo_tracker")

DEFAULT_LOG_PATH = Path(__file__).resolve().parents[2] / "data" / "lichess_matches.jsonl"


@dataclass
class MatchRecord:
    timestamp: str
    game_id: str
    opponent: str
    opponent_elo: int | None
    result: str  # "win" | "loss" | "draw"
    engine_elo_after: int | None
    time_control: str
    moves_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EloTracker:
    def __init__(self, log_path: Path | str | None = None) -> None:
        self.log_path = Path(log_path) if log_path else DEFAULT_LOG_PATH
        self.matches: list[MatchRecord] = []
        self.initial_elo: int | None = None
        self.latest_elo: int | None = None
        self._load_existing_matches()

    def _load_existing_matches(self) -> None:
        if not self.log_path.exists():
            return
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        record = MatchRecord(
                            timestamp=data.get("timestamp", ""),
                            game_id=data.get("game_id", ""),
                            opponent=data.get("opponent", "unknown"),
                            opponent_elo=data.get("opponent_elo"),
                            result=data.get("result", "draw"),
                            engine_elo_after=data.get("engine_elo_after"),
                            time_control=data.get("time_control", "-"),
                            moves_count=int(data.get("moves_count", 0)),
                        )
                        self.matches.append(record)
                        if record.engine_elo_after is not None:
                            if self.initial_elo is None:
                                self.initial_elo = record.engine_elo_after
                            self.latest_elo = record.engine_elo_after
                    except Exception as exc:
                        logger.warning("Failed to parse match line: %s (%s)", line, exc)
        except Exception as exc:
            logger.error("Error reading existing matches log %s: %s", self.log_path, exc)

    def set_initial_elo(self, elo: int | None) -> None:
        if elo is not None:
            if self.initial_elo is None:
                self.initial_elo = elo
            self.latest_elo = elo

    def record_match(
        self,
        *,
        game_id: str,
        opponent: str,
        opponent_elo: int | None,
        result: str,
        engine_elo_after: int | None,
        time_control: str,
        moves_count: int,
        timestamp: str | None = None,
    ) -> MatchRecord:
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        record = MatchRecord(
            timestamp=timestamp,
            game_id=game_id,
            opponent=opponent,
            opponent_elo=opponent_elo,
            result=result.lower(),
            engine_elo_after=engine_elo_after,
            time_control=time_control,
            moves_count=moves_count,
        )

        self.matches.append(record)
        if engine_elo_after is not None:
            if self.initial_elo is None:
                self.initial_elo = engine_elo_after
            self.latest_elo = engine_elo_after

        self._append_to_file(record)
        self.print_match_summary(record)
        return record

    def _append_to_file(self, record: MatchRecord) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
        except Exception as exc:
            logger.error("Failed to append match record to %s: %s", self.log_path, exc)

    def print_match_summary(self, record: MatchRecord) -> None:
        wins = sum(1 for m in self.matches if m.result == "win")
        losses = sum(1 for m in self.matches if m.result == "loss")
        draws = sum(1 for m in self.matches if m.result == "draw")
        total = len(self.matches)
        win_rate = (wins + 0.5 * draws) / max(1, total) * 100.0

        elo_delta_str = "N/A"
        if self.initial_elo is not None and self.latest_elo is not None:
            delta = self.latest_elo - self.initial_elo
            sign = "+" if delta > 0 else ""
            elo_delta_str = f"{self.latest_elo} ({sign}{delta})"
        elif self.latest_elo is not None:
            elo_delta_str = f"{self.latest_elo}"

        result_tag = record.result.upper()
        if record.result == "win":
            result_display = f"\033[92m{result_tag}\033[0m"
        elif record.result == "loss":
            result_display = f"\033[91m{result_tag}\033[0m"
        else:
            result_display = f"\033[93m{result_tag}\033[0m"

        divider = "=" * 65
        print(f"\n{divider}")
        print(f"[MATCH FINISHED] Game {record.game_id} | Result: {result_display}")
        print(f"{divider}")
        print(f" Opponent:     {record.opponent} (Rating: {record.opponent_elo if record.opponent_elo is not None else '?'})")
        print(f" Time Control: {record.time_control} | Moves: {record.moves_count}")
        print(f" Engine Rating:{elo_delta_str}")
        print(f" Session Stats:{total} games | W: {wins} | L: {losses} | D: {draws} | Score: {win_rate:.1f}%")
        print(f"{divider}\n")

    def get_summary(self) -> dict[str, Any]:
        wins = sum(1 for m in self.matches if m.result == "win")
        losses = sum(1 for m in self.matches if m.result == "loss")
        draws = sum(1 for m in self.matches if m.result == "draw")
        total = len(self.matches)
        win_rate = (wins + 0.5 * draws) / max(1, total) if total > 0 else 0.0

        elo_delta = 0
        if self.initial_elo is not None and self.latest_elo is not None:
            elo_delta = self.latest_elo - self.initial_elo

        return {
            "total_games": total,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": round(win_rate, 4),
            "initial_elo": self.initial_elo,
            "current_elo": self.latest_elo,
            "elo_delta": elo_delta,
        }

    @staticmethod
    def parse_rating_from_account(account_data: dict[str, Any], speed: str = "blitz") -> int | None:
        try:
            perfs = account_data.get("perfs", {})
            speed_perf = perfs.get(speed.lower()) or perfs.get("blitz") or perfs.get("rapid")
            if isinstance(speed_perf, dict):
                return speed_perf.get("rating")
        except Exception:
            pass
        return None
