"""Look up a total vision delay in the registered S6 delay grid.

Total delay D = camera latency upper bound + detector processing median +
radio term. The camera upper bound comes from a ``zephyr-vision-latency-1``
file (``tools/vision_latency.py analyze``). The processing median comes from
a ``zephyr-processing-time-1`` file or is typed in by hand. The radio term is
the upper end of the one-way mean delay interval in a ``zephyr-espnow-fit-1``
file when one is given; otherwise it is a 40 ms default that is assumed and
simulated, not measured. The two radio terms are different statistics: the
default is a tail value (the worst simulated seed's p95 delay), while the fit
term bounds the MEAN one-way delay and leaves out jitter and the tail, so
giving a fit usually lowers D. The result records this under ``limits``.

D is rounded UP to the next S6 grid delay, and that row's pass or fail per
arm is read from the registered S6 summary, which this tool only reads. A D
above the largest grid delay is reported as off the grid. The result is
labelled by its weakest input: synthetic stays synthetic, and an assumed
radio term means the total is at most "partly measured".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# One provenance rule for measured processing-time files, kept in the link converter.
from tools.vision_latency_to_link import LinkConversionError, _require_processing_provenance  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = REPO_ROOT / "sim" / "runs" / "s6-recovery" / "summary.json"

SCHEMA = "zephyr-latency-budget-1"
LATENCY_SCHEMA = "zephyr-vision-latency-1"
PROCESSING_SCHEMA = "zephyr-processing-time-1"
FIT_SCHEMA = "zephyr-espnow-fit-1"
S6_SCHEMA = "zephyr-s6-recovery-1"

# The single worst seed's p95 delay in the EXP-017 tolerance run: a pessimistic
# placeholder from the simulation, not a measurement.
DEFAULT_RADIO_S = 0.040
DEFAULT_RADIO_LABEL = "assumed, simulated"
DEFAULT_RADIO_STATUS = "assumed (simulated)"

# A sum of float seconds can land a hair above a grid point (0.05 + 0.01 + 0.04
# is not exactly 0.1). Anything within a nanosecond of a grid point counts as on it.
GRID_TOLERANCE_S = 1e-9

MOTIONS = ("constant_velocity", "constant_turn")
SYNTHETIC_STATUSES = ("synthetic", "fixture")
RADIO_FIT_NOTE = ("the fit term bounds the mean one-way delay and leaves out jitter and the tail, while the "
                  "40 ms default is a tail value (worst simulated seed's p95), so the two are not like for like")
PASS_RULE = "a cell passes when fewer than half of its seeds fail (the cell rule in the S6 summary)"


class LatencyBudgetError(ValueError):
    pass


def _seconds(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise LatencyBudgetError(f"{name} must be a finite number")
    if float(value) < 0:
        raise LatencyBudgetError(f"{name} must be zero or more")
    return float(value)


def _object(data: Any, schema: str, what: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise LatencyBudgetError(f"{what} must be a JSON object")
    if data.get("schema") != schema:
        raise LatencyBudgetError(f"{what} must have schema {schema!r}, got {data.get('schema')!r}")
    return data


def camera_term(latency: Any) -> dict[str, Any]:
    """Camera latency upper bound from a vision latency analysis."""
    data = _object(latency, LATENCY_SCHEMA, "latency input")
    camera = data.get("camera_latency_s")
    if not isinstance(camera, dict) or "high" not in camera:
        raise LatencyBudgetError("latency input has no camera_latency_s.high")
    if isinstance(camera["high"], (int, float)) and not isinstance(camera["high"], bool) and camera["high"] < 0:
        raise LatencyBudgetError("camera_latency_s.high is below zero: the apparent latency is smaller than the "
                                 "smallest assumed display lag, so check the display lag range or the run")
    high = _seconds(camera["high"], "camera_latency_s.high")
    apparent = data.get("apparent_latency_s") if isinstance(data.get("apparent_latency_s"), dict) else {}
    settings = data.get("settings") if isinstance(data.get("settings"), dict) else {}
    return {
        "value_s": high,
        "rule": "camera latency upper bound (apparent mean plus its stated error, minus the smallest assumed display lag)",
        "camera_low_s": camera.get("low"),
        "apparent_mean_s": apparent.get("mean"),
        "apparent_stated_error_s": apparent.get("stated_error"),
        "jitter_std_s": apparent.get("jitter_std_from_straddles"),
        "display_lag_assumed_s": settings.get("display_lag_s"),
        "status": str(data.get("status", "unknown")),
    }


def processing_term(processing: Any) -> dict[str, Any]:
    """Detector processing median from a processing-time file."""
    data = _object(processing, PROCESSING_SCHEMA, "processing input")
    if data.get("status") == "measured":
        try:
            _require_processing_provenance(data)
        except LinkConversionError as exc:
            raise LatencyBudgetError(str(exc)) from exc
    if "median_s" not in data:
        raise LatencyBudgetError("processing input has no median_s")
    median = _seconds(data["median_s"], "median_s")
    n = data.get("n")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise LatencyBudgetError("processing input n must be a whole number of at least 1")
    return {
        "value_s": median,
        "rule": "processing median",
        "p95_s": data.get("p95_s"),
        "n": n,
        "status": str(data.get("status", "unknown")),
    }


def typed_processing_term(median_s: float) -> dict[str, Any]:
    return {"value_s": _seconds(median_s, "processing median"), "rule": "processing median",
            "status": "typed in by hand"}


def radio_term(fit: Any = None) -> dict[str, Any]:
    """Radio term from an ESP-NOW fit, or the assumed 40 ms default."""
    if fit is None:
        return {"value_s": DEFAULT_RADIO_S, "label": DEFAULT_RADIO_LABEL, "status": DEFAULT_RADIO_STATUS,
                "rule": "default: the worst simulated seed's p95 delay in the EXP-017 tolerance run"}
    data = _object(fit, FIT_SCHEMA, "fit input")
    delay = data.get("delay")
    if not isinstance(delay, dict):
        raise LatencyBudgetError("fit input has no delay estimate (no echoes)")
    try:
        one_way = delay["one_way_estimate"]["delay_s"]
    except (KeyError, TypeError) as exc:
        raise LatencyBudgetError("fit input has no delay.one_way_estimate.delay_s") from exc
    value = _seconds(one_way.get("value"), "one-way delay_s")
    ci = one_way.get("ci")
    if isinstance(ci, list) and len(ci) == 2 and ci[1] is not None:
        term = _seconds(ci[1], "one-way delay_s interval upper end")
        rule = "upper end of the fit's one-way mean delay interval"
    else:
        term = value
        rule = "fit's one-way mean delay (the fit gave no interval)"
    trace = data.get("source_trace") if isinstance(data.get("source_trace"), dict) else {}
    status = str(trace.get("status", "unknown"))
    jitter = delay["one_way_estimate"].get("delay_jitter_s")
    return {"value_s": term, "label": f"from a {status} ESP-NOW fit", "status": status, "rule": rule,
            "one_way_mean_s": value, "one_way_mean_ci_s": ci,
            "one_way_jitter_s": jitter.get("value") if isinstance(jitter, dict) else None,
            "note": RADIO_FIT_NOTE, "trace_id": trace.get("trace_id")}


def overall_status(statuses: list[str]) -> str:
    if any(s in SYNTHETIC_STATUSES for s in statuses):
        return "synthetic"
    if all(s == "measured" for s in statuses):
        return "measured"
    return "partly measured"


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _row_cells(summary: dict[str, Any], noise_std_m: float) -> list[dict[str, Any]]:
    cells = summary.get("cells")
    if not isinstance(cells, list):
        raise LatencyBudgetError("S6 summary has no cells list")
    return [c for c in cells
            if isinstance(c, dict) and c.get("slice") == "main" and c.get("loss_model") == "independent"
            and c.get("loss_probability") == 0.0 and c.get("jitter_s") == 0.0
            and math.isclose(c.get("noise_std_m", -1.0), noise_std_m, abs_tol=1e-12)]


def delay_grid(summary: dict[str, Any], noise_std_m: float) -> tuple[float, ...]:
    """The S6 delays run for this noise level with no loss and no jitter."""
    try:
        return tuple(sorted({float(c["delay_s"]) for c in _row_cells(summary, noise_std_m)}))
    except (KeyError, TypeError, ValueError) as exc:
        raise LatencyBudgetError(f"S6 summary has a cell without a usable delay_s: {exc}") from exc


def round_up_to_grid(delay_s: float, grid: tuple[float, ...]) -> float | None:
    """The smallest grid delay at or above ``delay_s``, or None above the grid."""
    for point in sorted(grid):
        if delay_s <= point + GRID_TOLERANCE_S:
            return point
    return None


def read_rows(summary: dict[str, Any], delay_s: float, noise_std_m: float) -> list[dict[str, Any]]:
    rows = []
    cells = _row_cells(summary, noise_std_m)
    for motion in MOTIONS:
        match = [c for c in cells if c["motion"] == motion and c["delay_s"] == delay_s]
        if len(match) != 1:
            raise LatencyBudgetError(f"expected one S6 cell for {motion} at {delay_s} s, found {len(match)}")
        cell = match[0]
        try:
            row = {"motion": motion, "cell_id": cell["cell_id"]}
            for arm in ("naive", "recovery"):
                a = cell["arms"][arm]
                row[arm] = {"label": a["label"], "fail_count": a["fail_count"], "n_seeds": a["n_seeds"]}
        except (KeyError, TypeError) as exc:
            raise LatencyBudgetError(f"S6 cell {cell.get('cell_id')!r} is missing {exc}") from exc
        rows.append(row)
    return rows


def budget(latency: Any, processing: Any = None, fit: Any = None, *, summary: dict[str, Any] | None = None,
           summary_sha256: str | None = None, noise_std_m: float = 0.05,
           processing_term_override: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute D and read its S6 row. See the module docstring."""
    camera = camera_term(latency)
    proc = processing_term_override if processing_term_override is not None else processing_term(processing)
    radio = radio_term(fit)
    if summary is None:
        summary = load_summary(DEFAULT_SUMMARY)
        summary_sha256 = hashlib.sha256(DEFAULT_SUMMARY.read_bytes()).hexdigest()
    if not isinstance(summary, dict) or summary.get("schema") != S6_SCHEMA:
        raise LatencyBudgetError(f"S6 summary must have schema {S6_SCHEMA!r}")
    grid = delay_grid(summary, noise_std_m)
    if not grid:
        raise LatencyBudgetError(f"the S6 summary has no no-loss main cells at noise {noise_std_m} m")
    total = camera["value_s"] + proc["value_s"] + radio["value_s"]
    point = round_up_to_grid(total, grid)
    if point is None:
        rows: list[dict[str, Any]] = []
        verdict = (f"D = {1000 * total:.1f} ms is off the grid: above the largest S6 delay "
                   f"({1000 * grid[-1]:.0f} ms), so no row applies. The S6 delay axis needs extending first.")
    else:
        rows = read_rows(summary, point, noise_std_m)
        verdict = f"D = {1000 * total:.1f} ms rounds up to the {1000 * point:.0f} ms S6 row."
    limits = ["the S6 rows are simulated link and sensor, not flight performance",
              "the camera bound assumes the display lag range recorded in the latency file"]
    if fit is not None:
        limits.append(RADIO_FIT_NOTE)
    return {
        "schema": SCHEMA,
        "status": overall_status([camera["status"], proc["status"], radio["status"]]),
        "total_delay_s": total,
        "formula": "D = camera upper bound + processing median + radio term, rounded up to the S6 delay grid",
        "components": {"camera": camera, "processing": proc, "radio": radio},
        "grid_s": list(grid),
        "grid_point_s": point,
        "on_grid": point is not None,
        "noise_std_m": noise_std_m,
        "rows": rows,
        "verdict": verdict,
        "pass_rule": PASS_RULE,
        "summary": {"schema": summary.get("schema"), "status": summary.get("status"), "sha256": summary_sha256},
        "limits": limits,
    }


def _load_json(path: Path, what: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LatencyBudgetError(f"cannot read {what} {path}: {exc}") from exc


def _check_output(path: Path) -> Path:
    resolved = Path(path).resolve()
    runs = (REPO_ROOT / "sim" / "runs").resolve()
    if resolved == runs or runs in resolved.parents:
        raise LatencyBudgetError("output must not be inside sim/runs")
    return resolved


def format_report(result: dict[str, Any]) -> str:
    c = result["components"]
    lines = [
        f"Status of the result: {result['status']}",
        f"Camera upper bound: {1000 * c['camera']['value_s']:.1f} ms ({c['camera']['status']})",
        f"Processing median:  {1000 * c['processing']['value_s']:.1f} ms ({c['processing']['status']})",
        f"Radio term:         {1000 * c['radio']['value_s']:.1f} ms ({c['radio']['label']})",
        result["verdict"],
    ]
    for row in result["rows"]:
        parts = [f"{arm} {row[arm]['label']} ({row[arm]['fail_count']} of {row[arm]['n_seeds']} seeds fail)"
                 for arm in ("naive", "recovery")]
        lines.append(f"  {row['motion']}: " + ", ".join(parts))
    if result["rows"]:
        lines.append(f"Row read at {100 * result['noise_std_m']:.0f} cm noise, no loss, no jitter. "
                     f"Rule: {result['pass_rule']}.")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--latency", required=True, type=Path, help="latency.json from vision_latency.py analyze")
    proc = parser.add_mutually_exclusive_group(required=True)
    proc.add_argument("--processing", type=Path, help="a zephyr-processing-time-1 JSON file")
    proc.add_argument("--processing-median-ms", type=float, help="processing median typed in by hand, in ms")
    parser.add_argument("--fit", type=Path, help="a zephyr-espnow-fit-1 JSON file (default: 40 ms assumed)")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="registered S6 summary (read only)")
    parser.add_argument("--noise-std-m", type=float, default=0.05, help="which S6 noise row to read (default 0.05)")
    parser.add_argument("--out", type=Path, help="write the result JSON here (never inside sim/runs)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except (OSError, LatencyBudgetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    out = _check_output(args.out) if args.out is not None else None
    latency = _load_json(args.latency, "latency file")
    if args.processing is not None:
        processing, override = _load_json(args.processing, "processing file"), None
    else:
        processing, override = None, typed_processing_term(args.processing_median_ms / 1000.0)
    fit = _load_json(args.fit, "fit file") if args.fit is not None else None
    summary_bytes = Path(args.summary).read_bytes()
    try:
        summary = json.loads(summary_bytes)
    except json.JSONDecodeError as exc:
        raise LatencyBudgetError(f"cannot read S6 summary {args.summary}: {exc}") from exc
    result = budget(latency, processing, fit, summary=summary,
                    summary_sha256=hashlib.sha256(summary_bytes).hexdigest(), noise_std_m=args.noise_std_m,
                    processing_term_override=override)
    print(format_report(result))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
