#!/usr/bin/env python3
"""
SimPRO ML Training Pipeline
---------------------------
Controls the training workflow for the SimPRO Machine Learning service.

Workflow:
1. Rebuilds training dataset using `build-train.js` (syncs/transforms raw data).
2. Validates the JSON training data structure.
3. Analyses data distribution (classes, temporal fields) to guide training config.
4. Submits training request to the running ML service (`POST /train`).
5. Monitors progress, retries on errors, and reports final metrics.

Usage:
  python ml/tools/train_pipeline.py --host http://localhost:5001 --rebuild --user <U> --pass <P>
  python ml/tools/train_pipeline.py --test-size 0.2 --no-use-text
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

# --- Configuration Constants ---

DEFAULT_CUTOFF_DATE = "2016-10-17"  # Historical fallback if dynamic detection fails
DEFAULT_USE_TEXT = False
DEFAULT_CALIBRATE = True
DEFAULT_RARE_TOP_K = 20
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_THRESHOLDS = {"low": 0.44, "high": 0.64}

ERROR_DUMP_PATH = Path("/tmp/train_error.txt")
TRAIN_DATA_PATH = Path("ml/train.json")
BUILD_SCRIPT = ["node", "ml/tools/build-train.js"]

# Fields that likely contain ISO dates for temporal splitting
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
    """Encapsulates all hyperparameters and settings for a training run."""
    url: str
    cutoff_date: Optional[str]
    use_text: bool
    calibrate: bool
    rare_top_k: int
    test_size: float
    random_state: int
    thresholds: Optional[Dict[str, float]]

    def to_payload(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Converts config and data into the JSON payload expected by the API."""
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
    """Raised when the training pipeline encounters a fatal error."""


# --- Helper Functions ---

def run_command(command: List[str], cwd: Path) -> None:
    """Executes a shell command in the specified directory."""
    print(f"› Executing: {' '.join(command)}")
    try:
        subprocess.run(command, cwd=str(cwd), check=True)
    except FileNotFoundError:
        raise TrainingError(f"Command not found: {command[0]}") from None
    except subprocess.CalledProcessError as exc:
        raise TrainingError(
            f"Process failed (Exit Code {exc.returncode}): {' '.join(command)}"
        ) from exc


def load_training_data(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Reads and validates the training data JSON.
    Supports both simple array format and object format with metadata.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        offending_line = fetch_line(path, exc.lineno)
        print("✗ JSON Parse Error in ml/train.json", file=sys.stderr)
        if offending_line:
            print(f"  Line {exc.lineno}: {offending_line.rstrip()}", file=sys.stderr)
        raise TrainingError("Invalid JSON in training data.") from exc

    if isinstance(raw, list):
        return raw, {}

    if isinstance(raw, dict):
        rows = raw.get("data")
        if not isinstance(rows, list):
            raise TrainingError("Training data file missing 'data' array.")
        metadata = {k: v for k, v in raw.items() if k != "data"}
        return rows, metadata

    raise TrainingError("Unknown format for training data (expected list or object).")


def fetch_line(path: Path, line_number: int) -> Optional[str]:
    """Retrieves a specific line from a file safely."""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for current, line in enumerate(handle, start=1):
                if current == line_number:
                    return line
    except OSError:
        return None
    return None


def derive_label(row: Dict[str, Any], thresholds: Dict[str, float]) -> Optional[str]:
    """Calculates profitability label if missing, based on financial metrics."""
    p = to_num(row.get("netMarginPct"))
    if p is None:
        revenue = to_num(row.get("revenue"))
        cost = to_num(row.get("cost_total"))
        if revenue is not None and cost is not None and revenue > 0:
            p = (revenue - cost) / revenue
    
    if p is None:
        return None
    
    # Classification logic
    high = thresholds.get("high", DEFAULT_THRESHOLDS["high"])
    low = thresholds.get("low", DEFAULT_THRESHOLDS["low"])
    
    if p > high: return "High"
    if p >= low: return "Medium"
    return "Low"


def to_num(value: Any) -> Optional[float]:
    """Robust conversion to float, handling currency strings etc."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[^0-9eE.+-]", "", value.strip())
        if not cleaned: return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def summarise_dataset(data: List[Dict[str, Any]], thresholds: Dict[str, float]) -> None:
    """Prints a summary of the dataset class distribution."""
    from collections import Counter

    counts = Counter()
    for row in data:
        label = row.get("profitability_class")
        if not label:
            label = derive_label(row, thresholds)
        counts[label or "Unknown"] += 1

    total = sum(counts.values())
    print("\nDataset Summary:")
    for label in sorted(counts.keys()):
        print(f"  • {label:<8}: {counts[label]:>4} rows")
    print(f"  • Total   : {total:>4} rows\n")


def discover_date_fields(data: Iterable[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """Scans for time-series compatible fields to allow temporal splitting."""
    found: List[str] = []
    # Check max 100 rows to be fast
    check_limit = 100
    for i, row in enumerate(data):
        if i >= check_limit: break
        for key in ISO_DATE_KEYS:
            if key in row and looks_like_iso_date(row[key]):
                if key not in found:
                    found.append(key)
    return (len(found) > 0, found)


def looks_like_iso_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    # Quick check for YYYY-MM
    if not re.match(r'^\d{4}-\d{2}', value):
        return False
    try:
        # replace Z with +00:00 for strict ISO parsing
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def check_service_health(base_url: str) -> None:
    """Ensures the ML Service is running before attempting training."""
    health_url = urljoin(base_url, "/health")
    try:
        with urlopen(health_url, timeout=5) as response:
            if response.status >= 400:
                raise TrainingError(f"Health check failed: HTTP {response.status}")
    except URLError as exc:
        raise TrainingError(
            f"Cannot connect to ML Service at {base_url}.\n"
            f"  Ensure the Python ML server is running (e.g. uvicorn ml.profitability:app)."
        ) from exc


def post_train_request(config: TrainingConfig, data: List[Dict[str, Any]]) -> Tuple[int, Dict[str, str], str]:
    """Submits the training job to the API."""
    try:
        payload = config.to_payload(data)
        encoded = json.dumps(payload).encode("utf-8")
        
        req = Request(
            config.url,
            data=encoded,
            headers={"Content-Type": "application/json", "User-Agent": "simpro-train-pipeline/1.0"},
        )
        
        # Long timeout because training can be slow
        with urlopen(req, timeout=120) as response:
            return response.status, dict(response.getheaders()), response.read().decode("utf-8", errors="replace")
            
    except HTTPError as exc:
        return exc.code, dict(exc.headers.items()), exc.read().decode("utf-8", errors="replace")
    except URLError as exc:
        raise TrainingError(f"Network error during training request: {exc}")


def analyse_failure(status: int, payload: Optional[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
    """Diagnoses why a training request failed."""
    if payload is None:
        return "non-json", None

    if payload.get("ok") is True:
        return "success", None

    message = payload.get("error") or payload.get("message") or "Unknown error"
    
    # Categorize known errors
    if "Temporal split failed" in message: return "temporal", message
    if "empty vocabulary" in message.lower(): return "server", message  # NLP issue
    if "Need at least 3 samples" in message: return "too-few-classes", message
    if status >= 500: return "server", message
    if status >= 400: return "client", message
    
    return "unknown", message


def adjust_config_for_retry(current: TrainingConfig, reason: str, retries: Dict[str, bool]) -> Optional[TrainingConfig]:
    """Return a Modified config to work around specific errors, or None to stop retrying."""
    
    # Case 1: Not enough data for strict splitting -> Reduce test size
    if reason in {"server", "too-few-classes"} and not retries.get("reduced_test_size"):
        if current.test_size > 0.11:
            print("  ↻ Retry: Reducing test_size to 0.1 to keep more training data.")
            retries["reduced_test_size"] = True
            return replace(current, test_size=0.1)

    # Case 2: Empty Vocab -> Disable Text features
    if reason in {"server", "non-json"} and current.use_text and not retries.get("disabled_text"):
        print("  ↻ Retry: Disabling text features.")
        retries["disabled_text"] = True
        return replace(current, use_text=False)

    # Case 3: Random 500 error -> Simple one-time retry
    if reason == "server" and not retries.get("second_pass"):
        print("  ↻ Retry: Attempting one more time (transient server error).")
        retries["second_pass"] = True
        return current

    return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SimPRO ML Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    group_conn = parser.add_argument_group("Connection")
    group_conn.add_argument("--host", default="http://localhost:5001", help="Base URL for ML service")
    
    group_model = parser.add_argument_group("Model Hyperparameters")
    group_model.add_argument("--cutoff-date", dest="cutoff_date", default=DEFAULT_CUTOFF_DATE, help="Split date")
    group_model.add_argument("--use-text", dest="use_text", action="store_true", default=DEFAULT_USE_TEXT)
    group_model.add_argument("--no-use-text", dest="use_text", action="store_false")
    group_model.add_argument("--calibrate", dest="calibrate", action="store_true", default=DEFAULT_CALIBRATE)
    group_model.add_argument("--rare-top-k", type=int, default=DEFAULT_RARE_TOP_K)
    group_model.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    group_model.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    
    group_data = parser.add_argument_group("Data Generation")
    group_data.add_argument("--rebuild", action="store_true", help="Run build-train.js before training")
    group_data.add_argument("--user", help="SimPro Username")
    group_data.add_argument("--pass", dest="password", help="SimPro Password")
    group_data.add_argument("--token", help="SimPro JWT Token")
    
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]

    # Clean up previous dumps
    if ERROR_DUMP_PATH.exists():
        try:
            ERROR_DUMP_PATH.unlink()
        except OSError: pass

    # 1. Build Data (Optional)
    if args.rebuild:
        cmd = list(BUILD_SCRIPT)
        if args.user: cmd.extend(["--user", args.user])
        if args.password: cmd.extend(["--pass", args.password])
        if args.token: cmd.extend(["--token", args.token])
        
        print("\n--- Step 1: Building Dataset ---")
        run_command(cmd, cwd=repo_root)
    else:
        print("\n--- Step 1: Dataset Skipped (Using existing) ---")

    # 2. Load & Validate
    data_path = repo_root / TRAIN_DATA_PATH
    if not data_path.exists():
        raise TrainingError(f"Training data not found at {data_path}. Run with --rebuild first.")

    rows, file_metadata = load_training_data(data_path)
    if not rows:
        raise TrainingError("Dataset is empty.")

    # 3. Summarise
    # Merge CLI thresholds with File thresholds or defaults
    file_thresholds = file_metadata.get("thresholds") if isinstance(file_metadata.get("thresholds"), dict) else None
    thresholds = file_thresholds or DEFAULT_THRESHOLDS
    
    summarise_dataset(rows, thresholds)
    
    has_dates, date_fields = discover_date_fields(rows)
    config_cutoff = args.cutoff_date
    
    if not has_dates:
        print("Note: No temporal fields found. Forcing random split strategies.")
        config_cutoff = None
    else:
        print(f"Temporal fields detected: {', '.join(date_fields[:3])}...")

    # 4. Check Connection
    print(f"\n--- Step 2: Connection Check ({args.host}) ---")
    check_service_health(args.host)
    print("✔ Service is online.")

    # 5. Config Setup
    # Prefer CLI arguments, fall back to file data if CLI defaults not enterred
    
    def get_cfg(cli_val, file_key, default, cast_type=str):
        file_val = file_metadata.get(file_key)
        # If CLI is default and file has value -> use file
        if cli_val == default and file_val is not None:
             try: return cast_type(file_val)
             except: return cli_val
        return cli_val

    test_size = get_cfg(args.test_size, "test_size", DEFAULT_TEST_SIZE, float)
    random_state = get_cfg(args.random_state, "random_state", DEFAULT_RANDOM_STATE, int)
    rare_top_k = get_cfg(args.rare_top_k, "rare_top_k", DEFAULT_RARE_TOP_K, int)
    use_text = get_cfg(args.use_text, "use_text", DEFAULT_USE_TEXT, bool)
    
    config = TrainingConfig(
        url=urljoin(args.host, "/train"),
        cutoff_date=config_cutoff,
        use_text=use_text,
        calibrate=args.calibrate,
        rare_top_k=rare_top_k,
        test_size=test_size,
        random_state=random_state,
        thresholds=thresholds,
    )

    # 6. Training Loop
    print("\n--- Step 3: Training Model ---")
    retries: Dict[str, bool] = {}
    attempt = 1

    while True:
        print(f"Attempt {attempt}: text={config.use_text}, test={config.test_size}, split={config.cutoff_date or 'random'}")
        
        status, headers, body = post_train_request(config, rows)
        payload = parse_json(body) if body else None
        reason, message = analyse_failure(status, payload)

        if reason == "success":
            print(f"\n✔ Training Success (HTTP {status})")
            if payload and "metrics" in payload:
                print(json.dumps(payload["metrics"], indent=2))
            return 0

        # Failure Handling
        print(f"✗ Failure: {message}")
        
        if reason == "non-json":
            save_error_dump(body, headers, status)

        # Retry Logic
        next_config = adjust_config_for_retry(config, reason, retries)
        if not next_config:
            print("\nAborting after failed retries.")
            return 1

        config = next_config
        attempt += 1

def parse_json(body: str) -> Optional[Dict[str, Any]]:
    try: return json.loads(body)
    except: return None

def save_error_dump(body: str, headers: Dict[str, str], status: int) -> None:
    ERROR_DUMP_PATH.write_text(body, encoding="utf-8")
    print(f"  -> Dumped response to {ERROR_DUMP_PATH}")

if __name__ == "__main__":
    try:
        sys.exit(main())
    except TrainingError as exc:
        print(f"\nFATAL: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nOperation Cancelled.")
        sys.exit(130)
