"""Qwen3.6-27B renderer — the fan-OUT of the hourglass (DESIGN §2, PIPELINE 1.4).

Unified schema v0.1 record  ->  token ids + loss labels for SFT.

This module owns loss **mechanics** only. The schema owns loss **intent** (`trainable`), set
once by `normalize_core.py`; per SPEC.md "intent vs. mechanics" the renderer must NOT
re-derive intent from roles — it reads `trainable` and maps it onto token positions.

Everything here was verified against the real `chat_template.jinja` at the pinned revision
(hash `e84f32a2…`); see `tests/test_render_qwen36.py`. Three facts drive the design:

1. **Segment concatenation is only token-exact at special-token edges.** Splitting rendered
   text at arbitrary newlines changes tokenization (measured 331 vs 322 tokens on one
   trajectory) because `\\n\\n` is a SINGLE token (271), not two 198s. So every segment
   boundary below sits at a `<|im_start|>` / `<|im_end|>` / `<think>` / `</think>` /
   `<tool_call>` edge, or at a text boundary the template has already `|trim`-ed.

2. **Mask exactly what the scaffold supplies at inference; train what the model must emit.**
   With `add_generation_prompt=True` the scaffold force-feeds
   `<|im_start|>assistant\\n<think>\\n`, so those tokens are masked. The model is trained to
   produce everything after: reasoning text, `\\n</think>`, content, tool calls, `<|im_end|>`.

3. **Qwen3.6 is a HYBRID model, so "what the scaffold supplies" depends on the serving mode**
   (this is the subtle one). Template line 149 branches on `enable_thinking`:

       enable_thinking=True   ->  '<|im_start|>assistant\\n<think>\\n'
       enable_thinking=False  ->  '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n'

   With thinking OFF the scaffold *pre-closes* an empty think block, so the model never decides
   to skip thinking — it just resumes after `</think>`. With thinking ON the block is left open
   and the model must produce either reasoning or the closing tag itself.

   Rule 2 therefore has two correct answers, and applying the wrong one teaches the model to
   abort thinking in a mode where that transition is prefilled. So the mask boundary follows the
   RECORD's own reasoning content (`ThinkMode`), not a global flag:

       THINKING      mask '<|im_start|>assistant\\n<think>\\n'; train reasoning + '\\n</think>\\n\\n'
       NON_THINKING  mask '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' entirely

   The rendered TEXT is identical either way (`preserve_thinking=True` emits the tags for every
   historical turn regardless of `enable_thinking`, which only affects the generation prompt) —
   only the mask boundary and the tokenization of the empty span differ. Verified: the
   non-thinking prefix is an exact TOKEN prefix of the rendered turn
   (`['<|im_start|>','assistant','Ċ','<think>','ĊĊ','</think>','ĊĊ']`), so masking it aligns
   training with `enable_thinking=False` inference exactly.

   In THINKING mode an empty span must still be split as `<think>\\n` | `\\n</think>` =
   [248068, 198] + [198, 248069] rather than tokenized greedily as [248068, **271**, 248069],
   because inference feeds `<think>`,`\\n` and the model must then emit `\\n</think>`. That case
   is real: a thinking model may skip reasoning on individual turns.
   `EmptyThinkPolicy` keeps the greedy alternative available as an ablation.

Self-check: the segments we build are asserted to re-join into exactly the string
`apply_chat_template` produces, so any drift in the template (or in Jinja's `tojson`) fails
loudly instead of silently emitting wrong tokens.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schema"
CONTRACT_PATH = SCHEMA_DIR / "runtime_contract_v0.json"
TOOL_SET_DIR = SCHEMA_DIR / "tool_sets"

IGNORE_INDEX = -100

# Tool-call `command` values that modify files; used to bound the gold-patch leak check.
_EDIT_COMMANDS = {"str_replace", "create", "insert", "append", "write"}
_EDIT_TOOLS = {"file_edit", "file_write"}

# `provenance.tool_ontology` -> vendored tool definitions (pinned copies of each snapshot's own
# tools.json, so rendering is reproducible without the multi-GB snapshots on disk). Keys are
# ontology names, NOT source names: the fan-out never learns where a record came from.
TOOL_SET_FILES = {
    "openhands_open_swe": "openhands_open_swe.json",
    "openhands_nebius": "openhands_nebius.json",
    "swe_agent": "swe_agent.json",
    "monet_code": "monet_code.json",
}


class EmptyThinkPolicy(str, Enum):
    """How to tokenize an empty `<think>` span in THINKING mode (module docstring fact 3)."""

    GEN_PREFIX_CONSISTENT = "gen_prefix_consistent"  # [248068, 198, 198, 248069] — default
    TEMPLATE_GREEDY = "template_greedy"              # [248068, 271, 248069]


class ThinkMode(str, Enum):
    """Which serving mode a record is training, i.e. where the mask boundary falls.

    Qwen3.6 is hybrid: `enable_thinking` decides whether the scaffold pre-closes the think
    block. A record must be masked for the mode it actually represents, or training teaches a
    transition inference never asks for (module docstring fact 3).
    """

    THINKING = "thinking"          # reasoning present -> train reasoning + the closing tag
    NON_THINKING = "non_thinking"  # no reasoning -> the whole empty block is a prefilled prefix


class ThinkModePolicy(str, Enum):
    """How to choose `ThinkMode` per record."""

    # Default: follow the record's own content. Reasoning-bearing records train thinking mode,
    # reasoning-free records train non-thinking mode. Both are natively supported serving modes,
    # so a mixed corpus trains a mode-conditioned model rather than a confused one.
    PER_RECORD = "per_record"
    FORCE_THINKING = "force_thinking"          # ablation: render everything as thinking mode
    FORCE_NON_THINKING = "force_non_thinking"  # ablation: render everything as non-thinking


class RenderError(Exception):
    """A record cannot be rendered. Caller reports and drops it (never silently skips)."""


@dataclass
class RenderConfig:
    """Rendering knobs. Defaults come from the pinned runtime contract."""

    max_seq_len: int | None = None  # None = no limit (0e measures the histogram first)
    empty_think: EmptyThinkPolicy = EmptyThinkPolicy.GEN_PREFIX_CONSISTENT
    think_mode: ThinkModePolicy = ThinkModePolicy.PER_RECORD
    train_turn_end: bool = True     # train `<|im_end|>` so the model learns to stop
    verify_template_hash: bool = True
    check_gold_patch_leak: bool = True


@dataclass
class RenderedRecord:
    input_ids: list[int]
    labels: list[int]
    text: str
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def n_tokens(self) -> int:
        return len(self.input_ids)

    @property
    def n_trained_tokens(self) -> int:
        return sum(1 for l in self.labels if l != IGNORE_INDEX)


# ---------------------------------------------------------------------------
# Contract + tool sets
# ---------------------------------------------------------------------------

def load_contract(path: Path | str = CONTRACT_PATH) -> dict[str, Any]:
    with Path(path).open() as fh:
        return json.load(fh)


def _verify_template_hash(contract: dict[str, Any]) -> None:
    """Fail closed if the live template differs from the pinned one — the renderer is
    revision-specific (DESIGN §6)."""
    tpl = Path(contract["model"]["local_path"]) / contract["chat_template"]["source_file"]
    if not tpl.is_file():
        raise RenderError(f"chat template not found at {tpl}")
    live = hashlib.sha256(tpl.read_bytes()).hexdigest()
    pinned = contract["chat_template"]["sha256"]
    if live != pinned:
        raise RenderError(
            f"chat_template.jinja hash {live[:12]}… != pinned {pinned[:12]}…; "
            "the renderer is revision-specific — re-verify before rendering"
        )


def tool_set_key(record: dict[str, Any]) -> str:
    """Which tool ontology this record was generated under — READ, never inferred.

    The fan-out must not know any source's name. 'OpenHands' is not one fixed tool set (Nebius's
    OpenHands 0.54.0 adds `task_tracker`, DATA_SOURCES #5), so `scaffold` alone cannot resolve
    it — but the fix is for the NORMALIZER to declare `provenance.tool_ontology`, not for the
    renderer to sniff `provenance.source`. Same rule as `trainable`: the fan-in owns the
    decision, the fan-out consumes it. Adding a source must never edit this file.
    """
    prov = record.get("provenance") or {}
    key = prov.get("tool_ontology")
    if not key:
        raise RenderError(
            "provenance.tool_ontology is missing — the normalizer must declare which tool set "
            "the agent was offered; the renderer will not guess it from the source name"
        )
    if key not in TOOL_SET_FILES and key != "none":
        raise RenderError(f"unknown tool_ontology {key!r} (no vendored tool set)")
    return key


def load_tool_set(key: str) -> list[dict[str, Any]]:
    if key == "none":
        return []  # agentless: no tools offered, so no <tools> block
    try:
        fname = TOOL_SET_FILES[key]
    except KeyError:
        raise RenderError(f"unknown tool set {key!r}") from None
    with (TOOL_SET_DIR / fname).open() as fh:
        tools = json.load(fh)
    if not isinstance(tools, list):
        raise RenderError(f"tool set {key!r} is not a list of tool definitions")
    return tools


# ---------------------------------------------------------------------------
# Schema record -> chat messages (the authoritative text comes from the template)
# ---------------------------------------------------------------------------

def _blocks(event: dict[str, Any]) -> list[dict[str, Any]]:
    return event.get("blocks") or []


def _one_of(blocks: Iterable[dict[str, Any]], kind: str, *, where: str) -> dict[str, Any] | None:
    got = [b for b in blocks if b.get("kind") == kind]
    if len(got) > 1:
        # Fail loudly rather than invent a join rule that the template can't reproduce.
        raise RenderError(f"{where}: expected at most one {kind} block, got {len(got)}")
    return got[0] if got else None


def _tool_call_arguments(blk: dict[str, Any]) -> dict[str, Any]:
    """Arguments must be a mapping: the template iterates `tool_call.arguments|items`."""
    parsed = blk.get("arguments_parsed")
    if isinstance(parsed, dict):
        return parsed
    raw = blk.get("arguments_raw")
    if isinstance(raw, str) and raw.strip():
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RenderError(
                f"tool_call {blk.get('tool_name')!r} arguments are not valid JSON: {exc}"
            ) from None
        if isinstance(obj, dict):
            return obj
        raise RenderError(f"tool_call {blk.get('tool_name')!r} arguments are not an object")
    return {}


def record_to_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Map a v0.1 record onto the message list the chat template consumes."""
    msgs: list[dict[str, Any]] = []
    for i, ev in enumerate(record.get("trajectory", {}).get("events", [])):
        role = ev.get("role")
        bl = _blocks(ev)
        where = f"event[{i}] role={role}"
        if role in ("system", "user"):
            txt = _one_of(bl, "text", where=where)
            msgs.append({"role": role, "content": (txt or {}).get("text", "")})
        elif role == "assistant":
            reasoning = _one_of(bl, "reasoning", where=where)
            text = _one_of(bl, "text", where=where)
            calls = [b for b in bl if b.get("kind") == "tool_call"]
            m: dict[str, Any] = {
                "role": "assistant",
                # Always pass a string so the template takes the reasoning_content branch and
                # never tries to split `<think>` out of content.
                "reasoning_content": (reasoning or {}).get("text", "") or "",
                "content": (text or {}).get("text", "") or "",
            }
            if calls:
                m["tool_calls"] = [
                    {"type": "function", "function": {
                        "name": c.get("tool_name"), "arguments": _tool_call_arguments(c)}}
                    for c in calls
                ]
            msgs.append(m)
        elif role == "tool":
            for obs in [b for b in bl if b.get("kind") == "observation"]:
                msgs.append({"role": "tool", "name": obs.get("tool_name"),
                             "content": obs.get("text", "") or ""})
        else:
            raise RenderError(f"{where}: unexpected role")
    if not msgs:
        raise RenderError("record has no events")
    return msgs


# ---------------------------------------------------------------------------
# Segments — (text, trainable) pairs whose concatenation IS the template output
# ---------------------------------------------------------------------------

def _jinja_tojson(value: Any) -> str:
    """Match the template's `args_value | tojson` for non-string argument values.

    Only reached for non-string values; the joined-text assertion in `render_record` catches
    any divergence rather than letting it through.
    """
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


def _assistant_segments(
    msg: dict[str, Any],
    *,
    reasoning_trainable: bool,
    think_mode: ThinkMode,
    cfg: RenderConfig,
) -> list[tuple[str, bool]]:
    """Segments for one assistant turn, mirroring template lines 89-130.

    The think-span mask boundary follows `think_mode`, because Qwen3.6 is hybrid and the two
    serving modes prefill different amounts (module docstring fact 3).
    """
    segs: list[tuple[str, bool]] = []
    reasoning = (msg.get("reasoning_content") or "").strip()
    content = (msg.get("content") or "").strip()

    if think_mode is ThinkMode.NON_THINKING:
        # Training `enable_thinking=False` inference. The scaffold prefills the ENTIRE closed
        # empty block, so every token of it is masked: the model is never asked to decide to
        # skip thinking, it just resumes after `</think>`. Verified to be an exact token prefix
        # of the rendered turn, so this aligns training with that inference mode exactly.
        if reasoning:
            raise RenderError(
                "non-thinking mode cannot render a turn that has reasoning content — the "
                "reasoning would be silently dropped from the loss"
            )
        segs.append(("<|im_start|>assistant\n<think>\n\n</think>\n\n", False))
    elif reasoning or cfg.empty_think is EmptyThinkPolicy.GEN_PREFIX_CONSISTENT:
        # Training `enable_thinking=True` inference. The scaffold supplies only the OPEN think
        # tag, so the model must emit the reasoning and close the span itself. Splitting here is
        # also what makes an EMPTY think span tokenize as [<think>, \n, \n, </think>] instead of
        # [<think>, \n\n, </think>], matching what inference actually feeds.
        segs.append(("<|im_start|>assistant\n<think>\n", False))
        if reasoning:
            segs.append((reasoning, reasoning_trainable))
        segs.append(("\n</think>\n\n", True))
    else:
        # TEMPLATE_GREEDY ablation: keep the template's own `\n\n` single token.
        segs.append(("<|im_start|>assistant\n", False))
        segs.append(("<think>\n\n</think>\n\n", True))

    if content:
        segs.append((content, True))

    for i, call in enumerate(msg.get("tool_calls") or []):
        fn = call.get("function", call)
        name = fn.get("name")
        if i == 0:
            lead = "\n\n<tool_call>\n" if content else "<tool_call>\n"
        else:
            lead = "\n<tool_call>\n"
        segs.append((f"{lead}<function={name}>\n", True))
        for key, value in (fn.get("arguments") or {}).items():
            rendered = value if isinstance(value, str) else _jinja_tojson(value)
            segs.append((f"<parameter={key}>\n", True))
            segs.append((rendered, True))
            segs.append(("\n</parameter>\n", True))
        segs.append(("</function>\n</tool_call>", True))

    # `<|im_end|>` is the stop token the model must learn to emit; the following newline is a
    # separator the template adds when the NEXT turn is appended, so it stays masked.
    segs.append(("<|im_end|>", cfg.train_turn_end))
    segs.append(("\n", False))
    return segs


def build_segments(
    record: dict[str, Any],
    msgs: list[dict[str, Any]],
    full_text: str,
    cfg: RenderConfig,
) -> list[tuple[str, bool]]:
    """Build (text, trainable) segments covering the whole render.

    The system block (auto-generated `<tools>` JSON + format preamble + the source's own
    system prose) is taken VERBATIM from the template output rather than reimplemented: it is
    fully masked, so its internal structure is irrelevant to loss, and slicing avoids
    depending on Jinja's `tojson` for tool definitions.
    """
    segs: list[tuple[str, bool]] = []

    if full_text.startswith("<|im_start|>system"):
        end = full_text.find("<|im_end|>\n")
        if end == -1:
            raise RenderError("system block is not terminated by <|im_end|>")
        segs.append((full_text[: end + len("<|im_end|>\n")], False))
        body_msgs = msgs[1:] if msgs and msgs[0]["role"] == "system" else msgs
    else:
        body_msgs = msgs

    reasoning_trainable = _reasoning_trainable(record)
    think_mode = resolve_think_mode(record, cfg)

    prev_role: str | None = None
    for i, msg in enumerate(body_msgs):
        role = msg["role"]
        nxt = body_msgs[i + 1]["role"] if i + 1 < len(body_msgs) else None
        if role == "system":
            raise RenderError("system message must be first")
        if role == "user":
            segs.append((f"<|im_start|>user\n{(msg.get('content') or '').strip()}<|im_end|>\n", False))
        elif role == "assistant":
            segs.extend(_assistant_segments(
                msg, reasoning_trainable=reasoning_trainable, think_mode=think_mode, cfg=cfg))
        elif role == "tool":
            # Observations render as role `user` inside <tool_response>, and CONSECUTIVE
            # observations merge into one user turn (template lines 131-142). Always masked.
            chunk = ""
            if prev_role != "tool":
                chunk += "<|im_start|>user"
            chunk += f"\n<tool_response>\n{(msg.get('content') or '').strip()}\n</tool_response>"
            if nxt != "tool":
                chunk += "<|im_end|>\n"
            segs.append((chunk, False))
        else:
            raise RenderError(f"unexpected role {role!r}")
        prev_role = role
    return segs


def resolve_think_mode(record: dict[str, Any], cfg: RenderConfig) -> ThinkMode:
    """Which serving mode this record trains.

    Decided per RECORD, not per turn: the mode is a property of the generating model (does it
    have a thinking channel at all?), and `enable_thinking` is one flag for a whole rollout. A
    record mixing both would train the model to change mode mid-trajectory.

    Under PER_RECORD, any reasoning content anywhere makes it a thinking record. A thinking
    record may still have individual turns with no reasoning — that stays THINKING mode, and the
    empty span is split so the model learns it may close the block immediately.
    """
    if cfg.think_mode is ThinkModePolicy.FORCE_THINKING:
        return ThinkMode.THINKING
    if cfg.think_mode is ThinkModePolicy.FORCE_NON_THINKING:
        return ThinkMode.NON_THINKING
    for ev in record.get("trajectory", {}).get("events", []):
        for blk in _blocks(ev):
            if blk.get("kind") == "reasoning" and (blk.get("text") or "").strip():
                return ThinkMode.THINKING
    return ThinkMode.NON_THINKING


def _reasoning_trainable(record: dict[str, Any]) -> bool:
    """Read reasoning loss INTENT from the record (never re-derive it, per SPEC).

    v0.1 requires all reasoning blocks in a record to agree, so the first one decides. A
    record with no reasoning blocks (every Nebius record) has no intent to read; the value is
    unused because no reasoning segment is emitted.
    """
    for ev in record.get("trajectory", {}).get("events", []):
        for blk in _blocks(ev):
            if blk.get("kind") == "reasoning":
                return bool(blk.get("trainable"))
    return False


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_record(
    record: dict[str, Any],
    tokenizer,
    *,
    contract: dict[str, Any] | None = None,
    cfg: RenderConfig | None = None,
) -> RenderedRecord:
    """Render one v0.1 record to input_ids + labels.

    Raises RenderError for anything unrenderable (bad arguments, over length, template drift,
    gold-patch leak) so the caller can report-and-drop per PIPELINE 0e.
    """
    contract = contract or load_contract()
    cfg = cfg or RenderConfig(max_seq_len=contract["rendering"]["max_seq_len"])
    if cfg.verify_template_hash:
        _verify_template_hash(contract)

    msgs = record_to_messages(record)
    tools = load_tool_set(tool_set_key(record))
    kwargs = contract["chat_template"]["kwargs"]

    # The template is authoritative for text; our segments must reproduce it exactly.
    full_text = tokenizer.apply_chat_template(msgs, tools=tools, tokenize=False, **kwargs)

    segs = build_segments(record, msgs, full_text, cfg)
    joined = "".join(text for text, _ in segs)
    if joined != full_text:
        raise RenderError(
            "segment text != apply_chat_template output — the renderer no longer mirrors the "
            f"template (first divergence at char {_first_diff(joined, full_text)})"
        )

    if cfg.check_gold_patch_leak:
        _assert_no_gold_patch(record, full_text)

    input_ids: list[int] = []
    labels: list[int] = []
    for text, trainable in segs:
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False).input_ids
        input_ids.extend(ids)
        labels.extend(ids if trainable else [IGNORE_INDEX] * len(ids))

    # Segmented tokenization must equal one-shot tokenization of the same text, EXCEPT for the
    # deliberate empty-think split (which adds one token per empty span, same text).
    _assert_token_consistency(tokenizer, full_text, input_ids, cfg)

    if cfg.max_seq_len is not None and len(input_ids) > cfg.max_seq_len:
        raise RenderError(f"over_length: {len(input_ids)} > max_seq_len {cfg.max_seq_len}")

    n_trained = sum(1 for l in labels if l != IGNORE_INDEX)
    n_reasoning_blocks = sum(
        1 for ev in record.get("trajectory", {}).get("events", [])
        for b in _blocks(ev) if b.get("kind") == "reasoning" and (b.get("text") or "").strip()
    )
    n_assistant = sum(
        1 for ev in record.get("trajectory", {}).get("events", []) if ev.get("role") == "assistant"
    )
    return RenderedRecord(
        input_ids=input_ids,
        labels=labels,
        text=full_text,
        stats={
            "n_tokens": len(input_ids),
            "n_trained_tokens": n_trained,
            "trained_fraction": (n_trained / len(input_ids)) if input_ids else 0.0,
            "n_assistant_turns": n_assistant,
            "n_reasoning_blocks": n_reasoning_blocks,
            # PIPELINE 0e / STATUS: per-record stat that keeps the thinking-mask variant
            # testable as an ablation without re-normalizing.
            "reasoning_coverage": (n_reasoning_blocks / n_assistant) if n_assistant else 0.0,
            "reasoning_trainable": _reasoning_trainable(record),
            "tool_set": tool_set_key(record),
            "empty_think_policy": cfg.empty_think.value,
            # Which serving mode this record trains. Carried per-record so the mixture ratio is
            # measurable and so eval knows which `enable_thinking` each record was rendered for.
            "think_mode": resolve_think_mode(record, cfg).value,
        },
    )


def _first_diff(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def _assert_token_consistency(tokenizer, full_text: str, input_ids: list[int], cfg: RenderConfig) -> None:
    """Segment-wise tokenization must encode the same thing as one-shot tokenization.

    Compared *through* the tokenizer on both sides rather than against `full_text` directly,
    because the tokenizer applies Unicode NFC: real trajectories contain decomposed sequences
    (e.g. `A`+U+030A in an observation) that encode to the precomposed token, so a decoded
    string legitimately differs from the source bytes even with no segmentation error. Going
    through the tokenizer on both sides cancels that and still catches real drift.

    The deliberate empty-think split changes ids but not text, so it passes this check by
    construction (that is exactly why the check is on decoded text, not on the id sequence).
    """
    ours = tokenizer.decode(input_ids, skip_special_tokens=False)
    theirs = tokenizer.decode(
        tokenizer(full_text, add_special_tokens=False).input_ids, skip_special_tokens=False
    )
    if ours != theirs:
        raise RenderError(
            "segment-wise tokenization differs from one-shot tokenization of the rendered text "
            f"(first divergence at char {_first_diff(ours, theirs)})"
        )


def _assert_no_gold_patch(record: dict[str, Any], text: str) -> None:
    """The gold patch is verifier-only and must never reach model-visible input the agent did
    not produce (DATA_SOURCES #8).

    Checked against the PROMPT PREFIX only — everything up to the agent's first file-modifying
    tool call. After that point the gold patch legitimately appears in `git diff` observations
    (measured in ~3% of resolved Open-SWE trajectories: it is the agent's own diff echoed back,
    which coincides with the gold patch precisely because the trajectory resolved the issue).
    Rejecting those would discard valid data; a patch present *before* any edit is a real leak.
    """
    ref = (record.get("verification") or {}).get("reference_patch")
    gold = (ref or {}).get("patch") if isinstance(ref, dict) else None
    if not isinstance(gold, str):
        return
    # Compare stripped: the template `|trim`s every message, so a patch that ends in a newline
    # would not match verbatim even when it is fully present in the prompt.
    gold = gold.strip()
    if not gold:
        return
    prefix = text[: _first_edit_char_offset(record, text)]
    if gold in prefix:
        raise RenderError("gold_patch_leak: reference_patch appears in the prompt before any edit")


def _first_edit_char_offset(record: dict[str, Any], text: str) -> int:
    """Character offset in the rendered text of the agent's first file-modifying tool call
    (len(text) if it never edits, i.e. check the whole render)."""
    for ev in record.get("trajectory", {}).get("events", []):
        for blk in _blocks(ev):
            if blk.get("kind") != "tool_call":
                continue
            args = blk.get("arguments_parsed") or {}
            is_named_edit = blk.get("tool_name") in _EDIT_TOOLS
            is_command_edit = (isinstance(args, dict)
                               and str(args.get("command", "")) in _EDIT_COMMANDS)
            if not (is_named_edit or is_command_edit):
                continue
            marker = f"<function={blk.get('tool_name')}>"
            idx = text.find(marker)
            return idx if idx != -1 else len(text)
    return len(text)
