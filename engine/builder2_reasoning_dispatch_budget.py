"""
Authoritative per-invocation OpenAI reasoning dispatch budget and ledger.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from engine.builder2_tournament_contracts import Builder2TournamentError

CALL_BUDGET_EXHAUSTED = "builder2_complete_ad_reasoning_resume_call_budget_exhausted"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _role_key(role: str) -> str:
    cleaned = str(role or "").strip()
    if cleaned.startswith("builder2_creator"):
        return "builder2_creator"
    if cleaned.startswith("builder2_judge"):
        return "builder2_judge"
    if cleaned.startswith("builder2_winner"):
        return "builder2_winner"
    return cleaned


class ControlledReasoningCallBudget:
    """
    Single authoritative ledger for all paid OpenAI reasoning dispatches in one invocation.
    """

    def __init__(self, *, max_calls: int) -> None:
        self.max_calls = max(1, int(max_calls))
        self._ledger: List[Dict[str, Any]] = []
        self._sequence = 0

    @property
    def reasoning_budget_limit(self) -> int:
        return self.max_calls

    @property
    def reasoning_budget_reserved(self) -> int:
        return len(self._ledger)

    @property
    def reasoning_budget_consumed(self) -> int:
        return len(self._ledger)

    @property
    def reasoning_budget_remaining(self) -> int:
        return max(0, self.max_calls - len(self._ledger))

    @property
    def actual_openai_dispatches_this_run(self) -> int:
        return len(self._ledger)

    @property
    def total_this_run(self) -> int:
        return len(self._ledger)

    @property
    def creator_calls_this_run(self) -> int:
        return sum(1 for entry in self._ledger if _role_key(str(entry.get("role") or "")) == "builder2_creator")

    @property
    def judge_calls_this_run(self) -> int:
        return sum(1 for entry in self._ledger if _role_key(str(entry.get("role") or "")) == "builder2_judge")

    @property
    def winner_calls_this_run(self) -> int:
        return sum(1 for entry in self._ledger if _role_key(str(entry.get("role") or "")) == "builder2_winner")

    @property
    def dispatch_ledger(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in self._ledger]

    @property
    def dispatches_by_role(self) -> Dict[str, int]:
        counts = {"builder2_creator": 0, "builder2_judge": 0, "builder2_winner": 0}
        for entry in self._ledger:
            role = _role_key(str(entry.get("role") or ""))
            if role in counts:
                counts[role] += 1
        return counts

    @property
    def dispatches_by_call_type(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for entry in self._ledger:
            call_type = str(entry.get("callType") or "normal").strip() or "normal"
            counts[call_type] = counts.get(call_type, 0) + 1
        return counts

    def assert_can_call(self, role: str) -> None:
        if self.reasoning_budget_remaining <= 0:
            raise Builder2TournamentError(f"{CALL_BUDGET_EXHAUSTED}:{role}")

    def reserve(
        self,
        role: str,
        *,
        call_type: str = "normal",
        candidate_id: Optional[str] = None,
        prototype_id: Optional[str] = None,
        judgment_id: Optional[str] = None,
        winner_attempt_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.assert_can_call(role)
        self._sequence += 1
        entry = {
            "sequence": self._sequence,
            "role": role,
            "callType": call_type,
            "candidateId": candidate_id or None,
            "prototypeId": prototype_id or None,
            "judgmentId": judgment_id or None,
            "winnerAttemptId": winner_attempt_id or None,
            "reservedAt": _utc_now_iso(),
            "httpDispatchBegan": False,
            "responseReceived": False,
            "terminalResult": None,
        }
        self._ledger.append(entry)
        return entry

    def mark_http_begun(self, entry: Dict[str, Any]) -> None:
        entry["httpDispatchBegan"] = True

    def mark_response_received(self, entry: Dict[str, Any]) -> None:
        entry["responseReceived"] = True

    def finalize(self, entry: Dict[str, Any], *, terminal_result: str) -> None:
        entry["terminalResult"] = str(terminal_result or "").strip() or None

    def record(self, role: str, *, call_type: str = "normal", **metadata: Any) -> Dict[str, Any]:
        entry = self.reserve(
            role,
            call_type=call_type,
            candidate_id=metadata.get("candidate_id") or metadata.get("candidateId"),
            prototype_id=metadata.get("prototype_id") or metadata.get("prototypeId"),
            judgment_id=metadata.get("judgment_id") or metadata.get("judgmentId"),
            winner_attempt_id=metadata.get("winner_attempt_id") or metadata.get("winnerAttemptId"),
        )
        self.mark_http_begun(entry)
        self.mark_response_received(entry)
        self.finalize(entry, terminal_result="accepted")
        return entry


def populate_report_reasoning_dispatch_budget(report: Dict[str, Any], budget: ControlledReasoningCallBudget) -> None:
    report["creatorCallsThisRun"] = budget.creator_calls_this_run
    report["judgeCallsThisRun"] = budget.judge_calls_this_run
    report["winnerCallsThisRun"] = budget.winner_calls_this_run
    report["totalReasoningCallsThisRun"] = budget.total_this_run
    report["maximumAllowedReasoningCalls"] = budget.reasoning_budget_limit
    report["reasoningBudgetLimit"] = budget.reasoning_budget_limit
    report["reasoningBudgetReserved"] = budget.reasoning_budget_reserved
    report["reasoningBudgetConsumed"] = budget.reasoning_budget_consumed
    report["reasoningBudgetRemaining"] = budget.reasoning_budget_remaining
    report["actualOpenAIDispatchesThisRun"] = budget.actual_openai_dispatches_this_run
    report["dispatchesByRole"] = budget.dispatches_by_role
    report["dispatchesByCallType"] = budget.dispatches_by_call_type
    report["dispatchLedger"] = budget.dispatch_ledger
