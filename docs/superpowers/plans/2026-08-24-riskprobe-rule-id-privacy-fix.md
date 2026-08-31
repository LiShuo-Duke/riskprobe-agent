# RiskProbe rule_id Privacy False-Positive Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent numeric-only generated `rule_id` values from being rejected as sensitive long-number payloads without weakening privacy checks.

**Architecture:** Fix the identifier at its generation source in `_make_rule`. Preserve every existing nonnumeric identifier, prefix only numeric-only 12-character digests, and leave privacy and Host state-machine behavior unchanged.

**Tech Stack:** Python 3.12, pytest, hashlib, RiskProbe stdio MCP

## Global Constraints

- Do not weaken `_LONG_NUMBER` or any privacy gate.
- Do not read or output entity values, sample rows, or dataset details.
- Do not edit immutable historical sidecars.
- Preserve existing nonnumeric `rule_id` values exactly.
- Add no dependencies or new abstraction.
- Do not create a Git commit unless the user explicitly requests one.

---

## File Structure

- Modify `src/riskprobe/rules/discovery.py`: make generated public rule IDs safe at the source.
- Modify `tests/rules/test_discovery.py`: cover numeric-only and existing alphanumeric digest behavior.

### Task 1: Make generated rule IDs privacy-safe

**Files:**
- Modify: `src/riskprobe/rules/discovery.py:65-72`
- Test: `tests/rules/test_discovery.py`

**Interfaces:**
- Consumes: `_make_rule(conditions: tuple[Condition, ...], origin: str) -> tuple[RiskRule, str]`
- Produces: the same signature and expression; only numeric-only generated IDs gain the `rule-` prefix.

- [ ] **Step 1: Write the failing regression test**

Update imports and add this test near the other deterministic ID checks:

```python
from types import SimpleNamespace

from riskprobe.models import Condition, RiskRule
from riskprobe.rules.discovery import _make_rule, discover_rules


@pytest.mark.parametrize(
    ("digest", "expected"),
    [
        ("123456789012" + "a" * 52, "rule-123456789012"),
        ("12ab567890cd" + "0" * 52, "12ab567890cd"),
    ],
)
def test_make_rule_prefixes_only_numeric_digest(
    monkeypatch: pytest.MonkeyPatch,
    digest: str,
    expected: str,
) -> None:
    monkeypatch.setattr(
        "riskprobe.rules.discovery.hashlib.sha256",
        lambda _payload: SimpleNamespace(hexdigest=lambda: digest),
    )

    rule, _expression = _make_rule(
        (Condition(feature="feature", operator=">", value=1.0),),
        "single",
    )

    assert rule.rule_id == expected
```

- [ ] **Step 2: Run the regression test and confirm the numeric case fails**

Run:

```bash
/opt/anaconda3/bin/python -m pytest tests/rules/test_discovery.py::test_make_rule_prefixes_only_numeric_digest -q
```

Expected before implementation: one numeric case fails because the result is `123456789012`; the alphanumeric case passes.

- [ ] **Step 3: Implement the minimum source fix**

Replace the existing ID assignment in `_make_rule` with:

```python
rule_id = hashlib.sha256(expression.encode("utf-8")).hexdigest()[:12]
if rule_id.isdigit():
    rule_id = f"rule-{rule_id}"
```

Do not change privacy checks, hashing input, digest length, ordering, or return types.

- [ ] **Step 4: Run targeted validation**

Run:

```bash
/opt/anaconda3/bin/python -m pytest tests/rules/test_discovery.py tests/test_privacy.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Restart and validate the MCP protocol**

Restart the `riskprobe` MCP server from Kiro's MCP Server view so the Python process loads the changed source. Use a new stable `idempotency_key` with `riskprobe_get_decision_context`. Assert the response is a bounded context rather than `agent_state_incomplete`; choose only policy-allowed actions, then submit the same key, exact `context_id`, and complete diagnosis evidence IDs. Assert the final response phase/status is `terminal`.

No Git commit is included because the user has not authorized one.
