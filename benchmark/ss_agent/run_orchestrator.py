#!/usr/bin/env python3
"""
SWE-bench Lite orchestrator (planner + executor) that writes predictions.jsonl.

This version supports servers that ALWAYS respond with SSE (Content-Type: text/event-stream)
even when stream=false is requested (common behind some OpenAI-compatible gateways).

It aggregates delta.content from SSE chunks and returns a final text string.
"""

import argparse
import json
import os
import re
import sys
import time
import hashlib
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def write_jsonl_line(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_done_instance_ids(out_path: str) -> set:
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                iid = o.get("instance_id")
                if iid:
                    done.add(iid)
            except Exception:
                continue
    return done


def now_ms() -> int:
    return int(time.time() * 1000)


def stable_run_id(instance: Dict[str, Any]) -> str:
    base = (instance.get("instance_id") or "") + "|" + (instance.get("repo") or "") + "|" + (instance.get("base_commit") or "")
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]


def _aggregate_sse_text(r: requests.Response) -> str:
    """
    Aggregate delta content from SSE stream.

    Supports common schemas:
    - choices[0].delta.content  (your server)
    - choices[0].message.content (less common in SSE, but handle)
    - choices[0].text (classic completions)
    """
    parts: List[str] = []
    for raw in r.iter_lines(decode_unicode=True):
        if not raw:
            continue
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except Exception:
            continue

        choices = obj.get("choices") or []
        if not choices:
            continue
        ch0 = choices[0] or {}

        # Your gateway: {"choices":[{"delta":{"content":"..."}}]}
        delta = ch0.get("delta") or {}
        if isinstance(delta, dict):
            piece = delta.get("content")
            if piece:
                parts.append(piece)
                continue

        # Some variants
        msg = ch0.get("message") or {}
        if isinstance(msg, dict):
            piece = msg.get("content")
            if piece:
                parts.append(piece)
                continue

        piece = ch0.get("text")
        if piece:
            parts.append(piece)
            continue

    return "".join(parts)


def chat_completion(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    api_key: Optional[str] = None,
    timeout_s: int = 120,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
    retries: int = 3,
    retry_sleep: float = 2.0,
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (content, meta). Works with:
    - application/json normal responses
    - text/event-stream SSE responses (aggregates delta content)
    """
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,  # keep explicit; some servers ignore and stream anyway
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if extra:
        payload.update(extra)
        payload["stream"] = False

    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        t0 = now_ms()
        try:
            # stream=True lets us read SSE reliably and avoids buffering issues
            with requests.post(url, headers=headers, json=payload, timeout=(10, timeout_s), stream=True) as r:
                ctype = (r.headers.get("content-type") or "").lower()

                if r.status_code >= 400:
                    body = (r.text or "")[:1200]
                    raise RuntimeError(f"HTTP {r.status_code} ctype={ctype} body[:1200]={body}")

                if "text/event-stream" in ctype:
                    content = _aggregate_sse_text(r)
                    latency = now_ms() - t0
                    return content, {"latency_ms": latency, "ctype": ctype, "streamed": True}

                # non-stream JSON
                text = r.text or ""
                if not text.strip():
                    raise RuntimeError(f"Empty response body (HTTP {r.status_code}, ctype={ctype})")

                try:
                    data = json.loads(text)
                except Exception as e:
                    raise RuntimeError(f"Non-JSON response ctype={ctype} body[:800]={text[:800]} err={e}")

                # Chat completions schema
                if data.get("choices"):
                    ch0 = data["choices"][0]
                    if isinstance(ch0, dict):
                        msg = ch0.get("message") or {}
                        if isinstance(msg, dict) and "content" in msg:
                            latency = now_ms() - t0
                            return msg["content"], {"latency_ms": latency, "ctype": ctype, "streamed": False}
                        if "text" in ch0:
                            latency = now_ms() - t0
                            return ch0["text"], {"latency_ms": latency, "ctype": ctype, "streamed": False}

                raise RuntimeError(f"Unexpected response schema ctype={ctype} body[:800]={text[:800]}")

        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(retry_sleep * attempt)
            else:
                raise
    raise last_err  # type: ignore


DIFF_RE = re.compile(r"(?s)(diff --git .+)")

def extract_patch(text: str) -> str:
    if not text:
        return ""
    fences = re.findall(r"```(?:diff|patch|git)?\s*(.*?)```", text, flags=re.S | re.I)
    for block in fences:
        b = block.strip()
        m = DIFF_RE.search(b)
        if m:
            return m.group(1).strip() + "\n"
        if b.startswith("diff --git "):
            return b + "\n"
    m = DIFF_RE.search(text)
    if m:
        return m.group(1).strip() + "\n"
    if text.strip().startswith("diff --git "):
        return text.strip() + "\n"
    return ""


def is_plausible_patch(patch: str) -> bool:
    if not patch:
        return False
    if not patch.lstrip().startswith("diff --git "):
        return False
    return ("\n@@ " in patch) or ("\n@@@" in patch)


def build_problem_text(instance: Dict[str, Any]) -> str:
    iid = instance.get("instance_id", "")
    repo = instance.get("repo", "")
    base_commit = instance.get("base_commit", "")
    problem = instance.get("problem_statement", "")
    hints = instance.get("hints_text", "")
    parts = [
        f"instance_id: {iid}",
        f"repo: {repo}",
        f"base_commit: {base_commit}",
        "",
        "Problem statement:",
        problem or "(missing)",
    ]
    if hints:
        parts += ["", "Hints:", hints]
    return "\n".join(parts).strip()


def planner_messages(instance: Dict[str, Any]) -> List[Dict[str, str]]:
    problem = build_problem_text(instance)
    system = (
        "You are a senior software engineer. Create a concise, actionable repair plan.\n"
        "Constraints:\n"
        "- You do NOT have repository files.\n"
        "- Provide a search strategy and likely files/functions to inspect based on the problem.\n"
        "- Provide a minimal change approach and test strategy.\n"
        "Output format:\n"
        "1) Diagnosis hypotheses\n"
        "2) Candidate files/areas\n"
        "3) Step-by-step plan\n"
        "4) Patch guidance (what to change)\n"
        "5) How to validate (tests/commands)\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": problem}]


def executor_messages(instance: Dict[str, Any], plan: str, feedback: Optional[str] = None) -> List[Dict[str, str]]:
    problem = build_problem_text(instance)
    system = (
        "You are an expert code patch generator.\n"
        "You must output ONLY a valid unified diff starting with 'diff --git'.\n"
        "Do not include explanations.\n"
        "Make minimal changes to fix the described issue.\n"
    )
    user = (
        problem
        + "\n\nPlanner plan:\n"
        + (plan or "(no plan)")
        + "\n\nTask: Produce a patch as a unified diff. Output only the diff."
    )
    if feedback:
        user += "\n\nVerifier feedback from previous attempt:\n" + feedback + "\n\nRevise the patch accordingly. Output only the diff."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def run_verify_cmd(cmd: str, timeout: int = 600) -> Tuple[int, str]:
    try:
        p = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return p.returncode, (p.stdout or "")
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if hasattr(e, "stdout") else ""
        return 124, out + "\n[verify] TIMEOUT\n"
    except Exception as e:
        return 125, f"[verify] ERROR: {e}\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")

    ap.add_argument("--planner-base-url", required=True)
    ap.add_argument("--planner-model", required=True)
    ap.add_argument("--executor-base-url", required=True)
    ap.add_argument("--executor-model", required=True)

    ap.add_argument("--planner-api-key", default=os.environ.get("PLANNER_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--executor-api-key", default=os.environ.get("EXECUTOR_API_KEY") or os.environ.get("OPENAI_API_KEY"))

    ap.add_argument("--planner-timeout", type=int, default=180)
    ap.add_argument("--executor-timeout", type=int, default=300)
    ap.add_argument("--planner-temp", type=float, default=0.2)
    ap.add_argument("--executor-temp", type=float, default=0.0)
    ap.add_argument("--executor-max-tokens", type=int, default=2400)
    ap.add_argument("--max-attempts", type=int, default=2)

    ap.add_argument("--verify-cmd", default="")
    ap.add_argument("--verify-timeout", type=int, default=600)

    ap.add_argument("--log", default="")
    args = ap.parse_args()

    done = load_done_instance_ids(args.out) if args.resume else set()

    n_seen = 0
    n_skipped = 0
    n_written = 0

    for instance in read_jsonl(args.instances):
        n_seen += 1
        if args.limit and n_seen > args.limit:
            break

        iid = instance.get("instance_id")
        if not iid:
            continue
        if iid in done:
            n_skipped += 1
            continue

        run_id = stable_run_id(instance)
        t0 = now_ms()

        plan = ""
        planner_err = ""
        planner_meta: Dict[str, Any] = {}
        try:
            plan, planner_meta = chat_completion(
                base_url=args.planner_base_url,
                model=args.planner_model,
                messages=planner_messages(instance),
                api_key=args.planner_api_key,
                timeout_s=args.planner_timeout,
                temperature=args.planner_temp,
            )
        except Exception as e:
            planner_err = str(e)

        patch = ""
        exec_err = ""
        exec_meta: Dict[str, Any] = {}
        verifier_feedback = None

        for _attempt in range(1, args.max_attempts + 1):
            try:
                resp, exec_meta = chat_completion(
                    base_url=args.executor_base_url,
                    model=args.executor_model,
                    messages=executor_messages(instance, plan, verifier_feedback),
                    api_key=args.executor_api_key,
                    timeout_s=args.executor_timeout,
                    temperature=args.executor_temp,
                    max_tokens=args.executor_max_tokens,
                )

                candidate = extract_patch(resp)
                if not is_plausible_patch(candidate):
                    verifier_feedback = "Patch invalid: must be a unified diff starting with 'diff --git' and contain hunks (@@). Output only the diff."
                    continue

                patch = candidate

                if args.verify_cmd:
                    code, out = run_verify_cmd(args.verify_cmd, timeout=args.verify_timeout)
                    if code == 0:
                        break
                    verifier_feedback = "Local verify failed:\n" + out[-4000:]
                    patch = ""
                    continue

                break

            except Exception as e:
                exec_err = str(e)
                verifier_feedback = f"Executor error: {e}"
                patch = ""
                continue

        record = {
            "instance_id": iid,
            "patch": patch,
            "_meta": {
                "run_id": run_id,
                "planner": {
                    "base_url": args.planner_base_url,
                    "model": args.planner_model,
                    "error": planner_err,
                    "meta": planner_meta,
                },
                "executor": {
                    "base_url": args.executor_base_url,
                    "model": args.executor_model,
                    "error": exec_err,
                    "meta": exec_meta,
                },
                "attempts": args.max_attempts,
                "ms": now_ms() - t0,
            },
        }
        write_jsonl_line(args.out, record)
        n_written += 1

        if args.log:
            write_jsonl_line(
                args.log,
                {
                    "ts_ms": now_ms(),
                    "instance_id": iid,
                    "run_id": run_id,
                    "planner_err": planner_err,
                    "exec_err": exec_err,
                    "planner_streamed": planner_meta.get("streamed"),
                    "executor_streamed": exec_meta.get("streamed"),
                    "patch_len": len(patch or ""),
                },
            )

        print(f"[{n_seen}] wrote instance_id={iid} patch_len={len(patch)}", flush=True)

    print(f"done. seen={n_seen} skipped={n_skipped} written={n_written} out={args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())