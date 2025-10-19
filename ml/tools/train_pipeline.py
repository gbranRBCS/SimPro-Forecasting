#!/usr/bin/env python3
"""Automated training helper for the local ML service.

This script replicates the manual workflow:
1. (Optionally) rebuilds training data via the Node.js helper.
2. Validates the generated JSON (catching trailing log lines).
3. Summarises label counts and detects available temporal fields.
4. Sends a POST request to `/train` with the requested options.
5. Handles common failure cases with retries and rich diagnostics.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


DEFAULT_CUTOFF_DATE = "2016-10-17"
DEFAULT_USE_TEXT = False
DEFAULT_CALIBRATE = True
DEFAULT_RARE_TOP_K = 20
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_THRESHOLDS = {"low": 0.44, "high": 0.64}
ERROR_DUMP_PATH = Path("/tmp/train_error.txt")
TRAIN_DATA_PATH = Path("ml/train.json")
BUILD_SCRIPT = ["node", "ml/tools/build-train.js"]
ISO_DATE_KEYS = (
    "dateIssued",
    "DateIssued",
    "dateCompleted",
    "DateCompleted",
    "completedDate",
    "completedOn",
    "dueDate",
    "DueDate",
    "date",
    "issued",
)


@dataclass
class TrainingConfig:
    url: str
    cutoff_date: Optional[str]
    use_text: bool
    calibrate: bool
    rare_top_k: int
    test_size: float
    random_state: int
    thresholds: Optional[Dict[str, float]]

    def to_payload(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "data": data,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "rare_top_k": self.rare_top_k,
            "use_text": self.use_text,
            "calibrate": self.calibrate,
        }
        if self.cutoff_date:
            payload["cutoff_date"] = self.cutoff_date
        if self.thresholds:
            payload["thresholds"] = self.thresholds
        return payload


class TrainingError(Exception):
    """Raised when training cannot be completed."""


def run_command(command: List[str], cwd: Path) -> None:
    print(f"› Running: {' '.join(command)}")
    try:
        subprocess.run(command, cwd=str(cwd), check=True)
    except FileNotFoundError:
        raise TrainingError(f"Required command not found: {command[0]}") from None
    except subprocess.CalledProcessError as exc:
        raise TrainingError(
            f"Command {' '.join(command)} failed with exit code {exc.returncode}."
        ) from exc


def load_training_data(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        offending_line = fetch_line(path, exc.lineno)
        print("✗ Failed to parse ml/train.json (potential trailing log line).", file=sys.stderr)
        if offending_line is not None:
            print(f"  Problematic line {exc.lineno}: {offending_line.rstrip()}", file=sys.stderr)
        raise TrainingError("Training data JSON is invalid.") from exc

    if isinstance(raw, list):
        return raw, {}

    if isinstance(raw, dict):
        rows = raw.get("data")
        if not isinstance(rows, list):
            raise TrainingError("Training data JSON must contain a 'data' array.")
        metadata = {k: v for k, v in raw.items() if k != "data"}
        return rows, metadata

    raise TrainingError("Training data JSON must be an array or an object with a 'data' array.")


def fetch_line(path: Path, line_number: int) -> Optional[str]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for current, line in enumerate(handle, start=1):
                if current == line_number:
                    return line
    except OSError:
        return None
    return None


def derive_label(row: Dict[str, Any], thresholds: Dict[str, float]) -> Optional[str]:
    p = to_num(row.get("netMarginPct"))
    if p is None:
        revenue = to_num(row.get("revenue"))
        cost = to_num(row.get("cost_total"))
        if revenue is not None and cost is not None and revenue > 0:
            p = (revenue - cost) / revenue
    if p is None:
        return None
    if p > thresholds.get("high", DEFAULT_THRESHOLDS["high"]):
        return "High"
    if p >= thresholds.get("low", DEFAULT_THRESHOLDS["low"]):
        return "Medium"
    return "Low"


def to_num(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        candidate = re.sub(r"[^0-9eE.+-]", "", candidate)
        try:
            return float(candidate)
        except ValueError:
            return None
    return None


def summarise_classes(data: List[Dict[str, Any]], thresholds: Dict[str, float]) -> None:
    from collections import Counter

    counts = Counter()
    for row in data:
        label = row.get("profitability_class")
        if not label:
            label = derive_label(row, thresholds)
        counts[label or "Unknown"] += 1

    total = sum(counts.values())
    print("✔ Training rows per class:")
    for label in sorted(counts.keys()):
        print(f"  {label:>7}: {counts[label]} rows")
    print(f"  Total : {total} rows")


def discover_date_fields(data: Iterable[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    found: List[str] = []
    for row in data:
        for key in ISO_DATE_KEYS:
            if key in row and looks_like_iso_date(row[key]):
                if key not in found:
                    found.append(key)
    return (len(found) > 0, found)


def looks_like_iso_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    candidate = value.strip()
    if not candidate:
        return False
    candidate = candidate.replace("Z", "+00:00")
    try:
        datetime.fromisoformat(candidate)
        return True
    except ValueError:
        return False


def ensure_service_alive(base_url: str) -> None:
    health_url = urljoin(base_url, "/health")
    req = Request(health_url, method="GET")
    try:
        with urlopen(req, timeout=10) as response:
            if response.status >= 400:
                raise TrainingError(
                    f"ML service health-check returned status {response.status}."
                )
    except URLError as exc:
        raise TrainingError(
            f"Unable to reach ML service at {health_url}: {exc}"
        ) from exc


def post_train_request(config: TrainingConfig, data: List[Dict[str, Any]]) -> Tuple[int, Dict[str, str], str]:
    payload = config.to_payload(data)
    encoded = json.dumps(payload).encode("utf-8")
    req = Request(
        config.url,
        data=encoded,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=120) as response:
            status = response.status
            headers = dict(response.getheaders())
            body = response.read().decode("utf-8", errors="replace")
            return status, headers, body
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        headers = dict(exc.headers.items()) if exc.headers else {}
        return exc.code, headers, body


def parse_json(body: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def save_error_dump(body: str, headers: Dict[str, str], status: int) -> None:
    ERROR_DUMP_PATH.write_text(body, encoding="utf-8")
    print("✗ Response was not valid JSON (saved to /tmp/train_error.txt)")
    print(f"  HTTP {status}")
    for key, value in headers.items():
        print(f"  {key}: {value}")


def analyse_failure(
    status: int,
    payload: Optional[Dict[str, Any]],
    text: str,
) -> Tuple[str, Optional[str]]:
    """Return (reason, message) for logging and retry decisions."""
    if payload is None:
        return "non-json", None

    if payload.get("ok") is True:
        return "success", None

    message = payload.get("error") or payload.get("message") or "Unknown error"
    if "Temporal split failed" in message:
        return "temporal", message
    if status == 500 or "empty vocabulary" in message.lower():
        return "server", message
    if "Need at least 3 samples per class" in message:
        return "too-few-classes", message
    if status >= 400:
        return "client", message
    return "unknown", message


def adjust_config_for_retry(
    current: TrainingConfig,
    reason: str,
    status: int,
    message: Optional[str],
    retries: Dict[str, bool],
) -> Optional[TrainingConfig]:
    updated = current
    if reason in {"server", "too-few-classes"} and not retries.get("reduced_test_size"):
        if current.test_size > 0.11:
            print("⚠ Retrying with smaller test_size=0.1 to retain more labelled rows.")
            retries["reduced_test_size"] = True
            updated = replace(current, test_size=0.1)
            return updated
    if reason in {"server", "non-json"} and current.use_text and not retries.get("disabled_text"):
        print("⚠ Retrying with use_text=false to avoid TF-IDF vocabulary issues.")
        retries["disabled_text"] = True
        updated = replace(current, use_text=False)
        return updated
    if status == 500 and not retries.get("second_pass"):
        retries["second_pass"] = True
        print("⚠ Retrying request one more time with existing parameters.")
        return current
    return None


def pretty_print_metrics(payload: Dict[str, Any]) -> None:
    metrics = payload.get("metrics")
    if not metrics:
        print("No metrics available in response.")
        return
    print("Training metrics:")
    print(json.dumps(metrics, indent=2))


def parse_thresholds(args: argparse.Namespace) -> Optional[Dict[str, float]]:
    thresholds = {}
    if args.threshold_low is not None:
        thresholds["low"] = args.threshold_low
    if args.threshold_high is not None:
        thresholds["high"] = args.threshold_high
    return thresholds or None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the ML model reliably.")
    parser.add_argument("--host", default="http://localhost:8000", help="Base URL for the ML service")
    parser.add_argument("--cutoff-date", dest="cutoff_date", default=DEFAULT_CUTOFF_DATE)
    parser.add_argument("--use-text", dest="use_text", action="store_true", default=DEFAULT_USE_TEXT)
    parser.add_argument("--no-use-text", dest="use_text", action="store_false")
    parser.add_argument("--calibrate", dest="calibrate", action="store_true", default=DEFAULT_CALIBRATE)
    parser.add_argument("--no-calibrate", dest="calibrate", action="store_false")
    parser.add_argument("--rare-top-k", type=int, default=DEFAULT_RARE_TOP_K)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--threshold-low", type=float)
    parser.add_argument("--threshold-high", type=float)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Regenerate ml/train.json via the Node helper before training",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]

    if ERROR_DUMP_PATH.exists():
        try:
            ERROR_DUMP_PATH.unlink()
        except OSError:
            pass

    if args.rebuild:
        run_command(BUILD_SCRIPT, cwd=repo_root)
    else:
        print("ℹ Using existing ml/train.json (pass --rebuild to regenerate).")

    data_path = repo_root / TRAIN_DATA_PATH
    if not data_path.exists():
        raise TrainingError(f"Training data not found at {data_path}")

    rows, file_metadata = load_training_data(data_path)
    if not rows:
        raise TrainingError("Training data must contain at least one row.")

    user_thresholds = parse_thresholds(args)
    file_thresholds = (
        file_metadata.get("thresholds")
        if isinstance(file_metadata.get("thresholds"), dict)
        else None
    )
    thresholds_for_summary = user_thresholds or file_thresholds or DEFAULT_THRESHOLDS
    summarise_classes(rows, thresholds_for_summary)

    has_dates, date_fields = discover_date_fields(rows)
    cutoff_date = args.cutoff_date
    if not has_dates:
        print("⚠ No ISO date fields detected; falling back to random stratified split.")
        cutoff_date = None
    else:
        print(f"✔ Detected temporal fields: {', '.join(date_fields)}")

    ensure_service_alive(args.host)

    thresholds_for_payload = user_thresholds or file_thresholds

    def file_value(key: str, expected_type: Tuple[type, ...]) -> Optional[Any]:
        value = file_metadata.get(key)
        if isinstance(value, expected_type):
            return value
        return None

    test_size_value = args.test_size
    file_test_size = file_value("test_size", (int, float))
    if args.test_size == DEFAULT_TEST_SIZE and file_test_size is not None:
        test_size_value = float(file_test_size)

    random_state_value = args.random_state
    file_random_state = file_value("random_state", (int,))
    if args.random_state == DEFAULT_RANDOM_STATE and file_random_state is not None:
        random_state_value = int(file_random_state)

    rare_top_k_value = args.rare_top_k
    file_rare_top_k = file_value("rare_top_k", (int,))
    if args.rare_top_k == DEFAULT_RARE_TOP_K and file_rare_top_k is not None:
        rare_top_k_value = int(file_rare_top_k)

    use_text_value = args.use_text
    file_use_text = file_value("use_text", (bool,))
    if args.use_text == DEFAULT_USE_TEXT and file_use_text is not None:
        use_text_value = file_use_text

    calibrate_value = args.calibrate
    file_calibrate = file_value("calibrate", (bool,))
    if args.calibrate == DEFAULT_CALIBRATE and file_calibrate is not None:
        calibrate_value = file_calibrate

    config = TrainingConfig(
        url=urljoin(args.host, "/train"),
        cutoff_date=cutoff_date,
        use_text=use_text_value,
        calibrate=calibrate_value,
        rare_top_k=rare_top_k_value,
        test_size=test_size_value,
        random_state=random_state_value,
        thresholds=thresholds_for_payload,
    )

    retries: Dict[str, bool] = {}
    attempt = 1
    last_status: Optional[int] = None

    while True:
        print("―" * 60)
        print(f"Attempt {attempt} → cutoff_date={config.cutoff_date!r}, test_size={config.test_size}, "
              f"use_text={config.use_text}, calibrate={config.calibrate}, rare_top_k={config.rare_top_k}")
        status, headers, body = post_train_request(config, rows)
        last_status = status
        payload = parse_json(body)
        reason, message = analyse_failure(status, payload, body)

        if reason == "success":
            print(f"✔ Training completed with HTTP {status}.")
            pretty_print_metrics(payload)
            print("Model trained successfully, metrics saved to model_meta.json")
            print(f"Status code: {status}")
            return 0

        if payload and payload.get("ok") is False:
            print(f"✗ Training error: {payload.get('error') or payload.get('message')}")
        elif message:
            print(f"✗ Training error: {message}")

        if reason == "temporal":
            print("Hint: adjust --cutoff-date earlier or later to ensure both sides have data.")
            break

        if reason == "non-json":
            save_error_dump(body, headers, status)

        next_config = adjust_config_for_retry(config, reason, status, message, retries)
        if next_config is None:
            if message:
                print(f"No further retries available. Last error: {message}")
            break

        config = next_config
        attempt += 1

    print("Training failed, see /tmp/train_error.txt for details." if ERROR_DUMP_PATH.exists() else "Training failed.")
    final_status = last_status if last_status is not None else "unknown"
    print(f"Status code: {final_status}")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except TrainingError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(130)
