#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: engine_runner.py
Version: 2.1.0
Last Modified: 2026-01-20

Purpose:
- Core function `run_flow` executes a flow from an in-memory object.
- Wrapper `run_flow_file` runs stripped prompt-flow JSON files for CLI usage.
- Uses the ai_models stack via ai_model_factory.
- Preserves guard behavior, <chomp_prompt>, <chompme>.
- Supports step types: llm_call, branch_on_input, tool_call, set_context.
- Saves final output and step-by-step debug logs when run from CLI.

###### DO NOT REMOVE ######
# JSON instructions, and user_request must not contain real linebreaks for Siemplify.
# Here we’re desktop-only; we pass them as-is to the model, but we keep guard escaping.
# SIEMPLIFY placeholders may appear; we resolve {previous_output} and {<step>_output}.
# Guarded spans are string-escaped safely without altering inner content semantics.
# Never use single backticks inside instruction/user_request/input_value strings.
# <chompme></chompme> and <chomp_prompt>{some_step_output}...</chomp_prompt> are honored.
# STICK To the SPECIFIC TASK WHEN UPDATING CODE.
######
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- Your AI stack ---
from ai_models import AIModel, VertexAIModel, LocalModel, LOCAL_MODEL_NAME  # noqa: F401
# Lucidum tool (optional but requested)
try:
    from lucidum_enricher import lucidum_enrich_and_pair  # noqa: F401
    _LUCIDUM_AVAILABLE = True
except Exception:
    _LUCIDUM_AVAILABLE = False

# ----------------------------
# AI Model Factory
# ----------------------------

def ai_model_factory(model_name: str, system_instruction: str, debug: bool = False) -> AIModel:
    """Creates an instance of an AI model based on the provided name."""
    # Per our spec, we only support these two models from the factory.
    if model_name not in ["gemini-2.5-pro", "gemini-2.5-flash"]:
        raise ValueError(f"Unsupported model name: {model_name}. Supported models are 'gemini-2.5-pro' and 'gemini-2.5-flash'.")
    return VertexAIModel(model_name=model_name, system_instruction=system_instruction, debug=debug)

# ----------------------------
# Defaults / Tunables
# ----------------------------

SCRIPT_NAME = "Desktop_Gemini_Flow_Engine_ai_models"
TOP_P_DEFAULT = 0.3
MAX_RETRIES_DEFAULT = 2
MAX_RECURSION_EXECUTIONS_PER_INDEX = 3  # loop guard

GUARD_START_DEFAULT = "###A1B1C1###"
GUARD_END_DEFAULT = "###A2B2C2###"

# ----------------------------
# Logging
# ----------------------------

def setup_logger(verbosity: int) -> logging.Logger:
    """Configures and returns a logger instance."""
    logger = logging.getLogger(SCRIPT_NAME)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(logging.DEBUG if verbosity > 0 else logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    return logger

# ----------------------------
# Guard helpers
# ----------------------------

def escape_guarded_json_blocks(raw: str, start_tag: str, end_tag: str) -> str:
    """Replaces guarded content with JSON-escaped inner content, then strips tags."""
    result = ""
    while True:
        s = raw.find(start_tag)
        if s == -1:
            result += raw
            break
        result += raw[: s + len(start_tag)]
        raw = raw[s + len(start_tag):]
        e = raw.find(end_tag)
        if e == -1:
            result += raw
            raw = ""
            break
        guarded = raw[:e]
        suffix = raw[e + len(end_tag):]
        escaped = json.dumps(guarded, ensure_ascii=False)[1:-1]
        result += escaped
        raw = suffix
    return result.replace(start_tag, "").replace(end_tag, "")

def apply_guard_to_fields(step: Dict[str, Any], start_tag: str, end_tag: str) -> None:
    """Applies guard escaping to common string fields in a step dictionary, in-place."""
    for key in ("instruction", "user_request", "input_value", "meta_description"):
        if key in step and isinstance(step[key], str):
            step[key] = escape_guarded_json_blocks(step[key], start_tag, end_tag)

# ----------------------------
# Template / placeholder processing
# ----------------------------

def resolve_placeholders_in_segment(text: str, step_outputs: Dict[str, Any], previous_output: Optional[str]) -> str:
    """Replaces {step_output}, {previous_output}, and initial context placeholders in a string."""
    if not isinstance(text, str):
        return text
    s = text
    s = s.replace("{previous_output}", "" if previous_output is None else str(previous_output))
    for step_key, output in step_outputs.items():
        s = s.replace(f"{{{step_key}}}", "" if output is None else str(output)) # For initial context like {USER_INPUT}
        s = s.replace(f"{{{step_key}_output}}", "" if output is None else str(output)) # For step outputs
    return s

def parse_chompme_list(ignore_list_string: str) -> set:
    """Parses a comma or newline-separated string into a set of lowercase items."""
    if not ignore_list_string:
        return set()
    items = [x.strip().lower() for x in re.split(r"[,\n]+", ignore_list_string) if x.strip()]
    return set(items)

def process_prompt_template(template_string: str,
                            previous_output: Optional[str],
                            step_outputs: Dict[str, Any],
                            ignore_set: set,
                            logger: logging.Logger) -> str:
    """Implements <chomp_prompt> and <chompme> logic, then resolves placeholders."""
    if not isinstance(template_string, str):
        return template_string

    chomp_prompt_pattern = re.compile(r"<chomp_prompt>\s*\{+([\w\d_]+_output)\}+\s*:?\s*(.*?)</chomp_prompt>", re.DOTALL)

    def _chomp_prompt_repl(m):
        key_full = m.group(1)
        body = m.group(2)
        step_name_key = key_full.replace("_output", "")
        val = step_outputs.get(step_name_key, None)
        val_s = "" if val is None else str(val).strip().lower()
        keep = (val is not None) and (val_s not in ignore_set)
        logger.debug(f"[chomp_prompt] gate={key_full} keep={keep}")
        return body if keep else ""

    after_cp = chomp_prompt_pattern.sub(_chomp_prompt_repl, template_string)

    chomp_pattern = re.compile(r"<chompme>(.*?)</chompme>", re.DOTALL)
    internal_ph = re.compile(r"\{([\w\d_]+_output)\}")

    out_parts = []
    last = 0
    for m in chomp_pattern.finditer(after_cp):
        out_parts.append(after_cp[last:m.start()])
        content = m.group(1)
        keep_block = True
        for p in internal_ph.findall(content):
            step_name = p.replace("_output", "")
            val = step_outputs.get(step_name, None)
            val_s = "" if val is None else str(val).strip().lower()
            if val is None or val_s in ignore_set:
                keep_block = False
                break
        logger.debug(f"[chompme] keep={keep_block}")
        if keep_block:
            out_parts.append(content)
        last = m.end()
    out_parts.append(after_cp[last:])
    s = "".join(out_parts)

    s = resolve_placeholders_in_segment(s, step_outputs, previous_output)
    return s

# ----------------------------
# Local tools (incl. Lucidum)
# ----------------------------
def tool_get_server_time() -> str:
    """Returns the current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()

def tool_echo(value: str = "") -> str:
    """Returns the provided value as a string."""
    return str(value)


LOCAL_TOOLS = {
    "get_server_time": tool_get_server_time,
    "echo": tool_echo
}

# ----------------------------
# Engine core
# ----------------------------

def _tasking_brief(steps: List[Dict[str, Any]], i: int, meta_flow_description: str) -> str:
    """Generates a contextual briefing for an LLM step."""
    parts = ["**--- TASKING BRIEF ---**", ""]
    if meta_flow_description:
        parts += ["**OVERALL MISSION OBJECTIVE:**", meta_flow_description, ""]
    if i > 0:
        parts.append("**PROJECT HISTORY (Completed Tasks):**")
        for j in range(i):
            s = steps[j]
            nm = s.get("name", f"index_{j}")
            md = (s.get("meta_description") or "").strip()
            if md:
                parts.append(f"- **From `{nm}`**: {md}")
        parts.append("")
    nm = steps[i].get("name", f"index_{i}")
    parts.append(f"**YOUR CURRENT ASSIGNMENT: `{nm}`**")
    parts.append("It is your turn to contribute. Your task is outlined in the prompt that follows this briefing.")
    parts.append("")
    nxt = []
    s = steps[i]
    if "next_step_if_contains" in s:
        cond = s["next_step_if_contains"]
        if isinstance(cond, list):
            for c in cond:
                if isinstance(c, dict):
                    if c.get("then"): nxt.append(c["then"])
        elif isinstance(cond, dict):
            for _, arr in cond.items():
                if isinstance(arr, list):
                    for c in arr:
                        if c.get("then"): nxt.append(c["then"])
    if s.get("goto_step"): nxt.append(s["goto_step"])
    if not nxt and (i + 1) < len(steps): nxt.append(steps[i + 1].get("name"))
    nxt = sorted({x for x in nxt if x})
    if nxt:
        parts.append("**UPCOMING TASKS (Handoff to Next Team Member(s)):**")
        for n in nxt:
            try:
                target = next(t for t in steps if t.get("name") == n)
                md = (target.get("meta_description") or "").strip()
                parts.append(f"- **Potential Handoff to `{n}`**: {md}" if md else f"- Handoff to `{n}`")
            except StopIteration:
                parts.append(f"- Handoff to `{n}` (description not found)")
        parts.append("\n**IMPORTANT: Your responsibility is strictly limited to your assignment. Do not perform the upcoming tasks.**")
        parts.append("")
    parts.append("--- TASKING DETAILS BEGIN ---")
    return "\n".join(parts) + "\n\n"

def run_flow(
    flow_content: Dict | List,
    initial_context: Dict[str, Any],
    *,
    default_model: str,
    meta_prompting: bool = False,
    guard_start: str = GUARD_START_DEFAULT,
    guard_end: str = GUARD_END_DEFAULT,
    top_p_default: float = TOP_P_DEFAULT,
    max_retries: int = MAX_RETRIES_DEFAULT,
    debug_mode: bool = False,
    verbosity: int = 0,
    ignore_list_content: str = ""
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Executes a flow defined by a Python object, returning an execution trace.
    """
    logger = setup_logger(verbosity)
    execution_log: List[Dict[str, Any]] = []

    if isinstance(flow_content, list):
        steps = flow_content
        meta_flow_description = ""
    elif isinstance(flow_content, dict) and "steps" in flow_content:
        steps = list(flow_content.get("steps") or [])
        meta_flow_description = flow_content.get("meta_flow_description", "") or ""
    else:
        raise ValueError("Flow content must be a list of steps or a dict with a 'steps' key.")

    final_output: str = ""
    step_outputs: Dict[str, Any] = initial_context.copy()
    previous_output: str = str(initial_context.get("USER_INPUT", ""))
    loop_counts: Dict[int, int] = {}
    current_index = 0

    ignore_set = parse_chompme_list(ignore_list_content)
    if ignore_set:
        logger.info(f"Loaded ignore set with {len(ignore_set)} items.")

    while current_index < len(steps):
        step = steps[current_index]
        name = step.get("name", f"index_{current_index}")
        step_type = step.get("step_type", "llm_call")
        loop_counts[current_index] = loop_counts.get(current_index, 0) + 1
        if loop_counts[current_index] > MAX_RECURSION_EXECUTIONS_PER_INDEX:
            msg = f"Infinite loop detected on step index {current_index} ('{name}')."
            execution_log.append({"step_index": current_index, "step_name": name, "event": "loop_guard_triggered", "message": msg})
            final_output = msg
            break

        apply_guard_to_fields(step, guard_start, guard_end)
        t_start = time.time()
        dbg: Dict[str, Any] = {"ts": datetime.now(timezone.utc).isoformat(), "step_index": current_index, "step_name": name, "step_type": step_type}

        try:
            if step_type == "tool_call":
                tool_name = step.get("tool_name")
                if not tool_name or tool_name not in LOCAL_TOOLS:
                    raise ValueError(f"Unknown local tool '{tool_name}'")

                call_kwargs = {}
                raw_args = step.get("tool_args") or step.get("tool_inputs") or {}
                for k, v in raw_args.items():
                    call_kwargs[k] = resolve_placeholders_in_segment(str(v), step_outputs, previous_output)
                out = LOCAL_TOOLS[tool_name](**call_kwargs)
                step_outputs[name] = out
                previous_output = out
                final_output = out
                dbg.update({"tool_name": tool_name, "tool_inputs": call_kwargs, "tool_output": out})

            elif step_type == "set_context":
                scope = step.get("context_scope")
                key = resolve_placeholders_in_segment(str(step.get("context_key")), step_outputs, previous_output)
                val = resolve_placeholders_in_segment(str(step.get("context_value")), step_outputs, previous_output)
                step_outputs[name] = f"{scope}:{key}={val}"
                dbg.update({"context_scope": scope, "context_key": key, "context_value": val})

            elif step_type == "branch_on_input":
                raw = step.get("input_value")
                if raw is None: raise ValueError("'input_value' is required for branch_on_input")
                value_to_check = resolve_placeholders_in_segment(str(raw), step_outputs, previous_output)
                step_outputs[name] = value_to_check
                previous_output = value_to_check
                final_output = value_to_check
                dbg["branch_input_value"] = value_to_check

                jump = None
                conds = step.get("next_step_if_contains")
                if isinstance(conds, list):
                    for c in conds:
                        if "includes" in c and c.get("then") and str(c["includes"]).lower() in value_to_check.lower():
                            jump = c["then"]; break
                        if "else" in c and c["else"]:
                            jump = c["else"]; break
                if jump:
                    try: current_index = next(i for i, s in enumerate(steps) if s.get("name") == jump)
                    except StopIteration: current_index += 1
                else:
                    current_index += 1
                t_end = time.time()
                dbg.update({"duration_ms": int((t_end - t_start) * 1000)})
                execution_log.append(dbg)
                if step.get("sleep_seconds"): time.sleep(int(step["sleep_seconds"]))
                continue

            elif step_type == "llm_call":
                instruction = step.get("instruction", "")
                user_request = step.get("user_request", "")
                model_name = step.get("model", default_model)
                step_temp = float(step.get("temperature", 0.7))
                step_top_p = float(step.get("top_p", top_p_default))
                max_tokens = int(step.get("output_tokens", 64000))

                compiled_instruction = process_prompt_template(
                    (_tasking_brief(steps, current_index, meta_flow_description) + instruction) if meta_prompting else instruction,
                    previous_output, step_outputs, ignore_set, logger)
                compiled_user = process_prompt_template(
                    user_request, previous_output, step_outputs, ignore_set, logger)

                model = ai_model_factory(model_name, system_instruction=compiled_instruction, debug=debug_mode)

                attempt, ok, last_err = 0, False, None
                ai_result = {}
                while attempt <= max_retries and not ok:
                    try:
                        ai_result = model.generate(compiled_user, temperature=step_temp, top_p=step_top_p, max_output_tokens=max_tokens)
                        ok = True
                    except Exception as e:
                        last_err, attempt = e, attempt + 1
                        if attempt <= max_retries: time.sleep(2 * attempt)
                if not ok: raise RuntimeError(f"Model call failed after {max_retries} retries: {last_err}")

                resp_text = ai_result.get("text", "")
                step_outputs[name] = resp_text
                previous_output = resp_text
                final_output = resp_text
                dbg.update({"model": model_name, "temperature": step_temp, "top_p": step_top_p, "max_tokens": max_tokens, "compiled_instruction": compiled_instruction, "compiled_user_request": compiled_user, "ai_text": resp_text, "input_tokens": ai_result.get("input_tokens", 0), "output_tokens": ai_result.get("output_tokens", 0), "full_response_log": ai_result.get("full_response_log", []) if debug_mode else None})

            else:
                raise ValueError(f"Unsupported step_type '{step_type}' in step '{name}'")

            jump_general = None
            if "next_step_if_contains" in step and step_type != "branch_on_input":
                mapping = step["next_step_if_contains"]
                if isinstance(mapping, dict):
                    for output_key_template, conditions in mapping.items():
                        val = str(step_outputs.get(name, "")) if output_key_template == "{this_step_output}" else str(step_outputs.get(output_key_template.replace("_output", ""), ""))
                        for cond in conditions:
                            if "includes" in cond and cond.get("then") and str(cond["includes"]).lower() in val.lower():
                                jump_general = cond["then"]; break
                            if "else" in cond and cond["else"]:
                                jump_general = cond["else"]; break
                        if jump_general: break
            if not jump_general and step.get("goto_step"): jump_general = step["goto_step"]

            if jump_general:
                try: current_index = next(i for i, s in enumerate(steps) if s.get("name") == jump_general)
                except StopIteration: current_index += 1
            else:
                current_index += 1

            if step.get("sleep_seconds"): time.sleep(int(step["sleep_seconds"]))

        except Exception as ex:
            dbg.update({"error": str(ex)})
            execution_log.append(dbg)
            raise

        t_end = time.time()
        dbg.update({"duration_ms": int((t_end - t_start) * 1000)})
        execution_log.append(dbg)

        if bool(step.get("halt_after_step")): break

    return final_output, step_outputs, execution_log

def run_flow_file(
    infile: Path,
    outdir: Path,
    debugdir: Path,
    *,
    default_model: str,
    meta_prompting: bool = False,
    guard_start: str = GUARD_START_DEFAULT,
    guard_end: str = GUARD_END_DEFAULT,
    top_p_default: float = TOP_P_DEFAULT,
    max_retries: int = MAX_RETRIES_DEFAULT,
    debug_mode: bool = False,
    verbosity: int = 0
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Executes one flow JSON file, handling all file I/O.
    """
    case_id = infile.stem.replace(f"run_{int(time.time())}", "cli_run") # Clean up temp names
    outdir.mkdir(parents=True, exist_ok=True)
    debugdir.mkdir(parents=True, exist_ok=True)

    flow_content_str = infile.read_text(encoding="utf-8")
    flow_content = json.loads(flow_content_str)

    initial_context = {"USER_INPUT": ""}

    ignore_content = ""
    ignore_csv = infile.with_suffix(".ignore.csv")
    if ignore_csv.exists():
        ignore_content = ignore_csv.read_text(encoding="utf-8")

    try:
        final_output, step_outputs, execution_log = run_flow(
            flow_content=flow_content,
            initial_context=initial_context,
            default_model=default_model,
            meta_prompting=meta_prompting,
            guard_start=guard_start,
            guard_end=guard_end,
            top_p_default=top_p_default,
            max_retries=max_retries,
            debug_mode=debug_mode,
            verbosity=verbosity,
            ignore_list_content=ignore_content
        )

        final_path = outdir / f"{case_id}.txt"
        final_path.write_text(final_output, encoding="utf-8")

        debug_path = debugdir / f"{case_id}.jsonl"
        with debug_path.open("w", encoding="utf-8") as f_debug:
            for record in execution_log:
                f_debug.write(json.dumps(record, ensure_ascii=False) + "\n")

        return case_id, final_output, step_outputs

    except Exception as e:
        (outdir / f"{case_id}.txt").write_text(f"Engine error: {e}", encoding="utf-8")
        raise

# ----------------------------
# CLI
# ----------------------------

def main():
    """Main entry point for CLI execution."""
    ap = argparse.ArgumentParser(description="Run stripped prompt-flow JSON files with ai_models.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="Run a single flow file (JSON).")
    src.add_argument("--indir", help="Run all *.json in this directory (non-recursive).")

    ap.add_argument("--outdir", default="ran-flow-folder", help="Dir for final outputs.")
    ap.add_argument("--debugdir", default="ran_steps_debug", help="Dir for per-step JSONL.")
    ap.add_argument("--model", default="gemini-2.5-flash", choices=["gemini-2.5-flash", "gemini-2.5-pro"], help="Default model.")
    ap.add_argument("--meta", action="store_true", help="Enable meta prompting header (Tasking Brief).")
    ap.add_argument("--guard-start", default=GUARD_START_DEFAULT)
    ap.add_argument("--guard-end", default=GUARD_END_DEFAULT)
    ap.add_argument("--top-p", type=float, default=TOP_P_DEFAULT)
    ap.add_argument("--retries", type=int, default=MAX_RETRIES_DEFAULT)
    ap.add_argument("--debug", action="store_true", help="Verbose model logging.")
    ap.add_argument("-v", "--verbose", action="count", default=0)

    args = ap.parse_args()
    logger = setup_logger(args.verbose)

    outdir, debugdir = Path(args.outdir), Path(args.debugdir)
    outdir.mkdir(parents=True, exist_ok=True)
    debugdir.mkdir(parents=True, exist_ok=True)

    files = [Path(args.file)] if args.file else sorted(Path(args.indir).glob("*.json"))

    if not files:
        logger.error("No input files found.")
        sys.exit(1)

    ok, fail = 0, 0
    for f in files:
        try:
            case_id, _, _ = run_flow_file(
                infile=f, outdir=outdir, debugdir=debugdir, default_model=args.model,
                meta_prompting=args.meta, guard_start=args.guard_start, guard_end=args.guard_end,
                top_p_default=args.top_p, max_retries=args.retries, debug_mode=args.debug, verbosity=args.verbose
            )
            ok += 1
            logger.info(f"[DONE] {case_id} -> {outdir / (case_id + '.txt')}")
        except Exception as e:
            fail += 1
            logger.error(f"[FAIL] {f.name}: {e}", exc_info=args.debug)

    logger.info(f"Complete. Succeeded: {ok}, Failed: {fail}, Outdir: {outdir}, Debug: {debugdir}")
    sys.exit(0 if fail == 0 else 2)

if __name__ == "__main__":
    main()