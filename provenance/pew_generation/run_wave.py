from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import os
import threading
import time
import json

from visd.Wave import Wave
from visd.Demo import Demo, Profile
from visd.utils.llm import LLM, LLMCallMetrics
from visd.utils.prompt import make_prompt
from visd.utils.filenames import get_model_costs_filepath, get_response_filepath
from visd.utils.env import setup_keys

_PRINT_LOCK = threading.Lock()

def _log(*args, **kwargs) -> None:
    """Thread-safe print."""
    with _PRINT_LOCK:
        print(*args, **kwargs)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class RunOneWaveInput:
    wave_id: str
    demographic_id: str
    run_id: str
    model_name: str
    config_dir: str
    base_output_dir: str
    data_dir: str = "data/use_data"
    prompts_dir: str = "prompts"
    base_url: str = DEFAULT_BASE_URL
    temperature: float = 0.0
    demo_info: bool = True


@dataclass
class SingleResponse:
    wave_id: str
    demo_id: str
    respondent: Profile
    success: bool
    answer: dict | None
    metrics: LLMCallMetrics
    randomization: dict | None = None


def response_to_dict(output) -> dict:
    """Convert a Pydantic model (or dict) into a plain dict."""
    if hasattr(output, "model_dump"):
        return output.model_dump()
    if isinstance(output, dict):
        return output
    return {}


def load_wave_index(data_dir: str = "data/use_data") -> list[dict]:
    """Read data/use_data/index.jsonl and return the list of wave entries."""
    index_path = Path(data_dir) / "index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Wave index not found: {index_path}. "
            f"Generate it with scripts/data_collection/build_index.py."
        )
    entries = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def run_one_wave(
    input: RunOneWaveInput,
    max_workers: int = 1,
) -> tuple[float, list[str]]:
    """Run the survey for every respondent in a demographic group.

    Parameters
    ----------
    max_workers:
        Number of respondents to survey concurrently.  The default (1) keeps
        the original sequential behaviour.  Higher values parallelise API
        calls with a thread pool — useful when rate limits allow it.

    Returns
    -------
    (total_cost, fail_ids)
    """
    wave = Wave(input.wave_id, data_dir=input.data_dir, prompts_dir=input.prompts_dir)
    # Demo now lives under data/demographic/{demo_id}.csv
    # input.data_dir may be "data/use_data"; we strip back to "data" for Demo.
    demo_base = str(Path(input.data_dir).parent) if Path(input.data_dir).name == "use_data" else input.data_dir
    demo = Demo(demo_id=input.demographic_id, data_dir=demo_base)

    llm = LLM(
        model_name=input.model_name,
        base_url=input.base_url,
        model_cost_file=str(get_model_costs_filepath(input.config_dir)),
        temperature=input.temperature,
    )

    # Pre-build these once — they are read-only and safe to share across threads
    schema       = wave.get_scheme()
    questions    = wave.get_schema()
    wave_id      = wave.id
    wave_subpath = wave.runs_subpath  # mirrors the prompt's sub-folder layout
    demo_id      = demo.demo_id

    if input.demo_info:
        profiles = demo.get_profiles()
    else:
        # Use the demo CSV only to obtain the field dates, then synthesise
        # 1000 blank profiles that carry only that date (no demographics).
        first = demo.get_profiles()
        dates_value = (
            first[0].attributes.get("dates", "") if first else ""
        ) or wave.info.get("field_dates", "")
        profiles = [
            Profile(
                demo_id=demo_id,
                respondent_id=f"blank_{i:04d}",
                weight=None,
                attributes={"dates": dates_value} if dates_value else {},
            )
            for i in range(1, 1001)
        ]

    # ── per-respondent worker ──────────────────────────────────────────
    def process_respondent(respondent: Profile) -> tuple[float, str | None]:
        """Return (cost, fail_message | None) for one respondent."""
        respondent_id = respondent.respondent_id
        _log(
            f"[INFO] wave={wave_id}  demo={demo_id}  respondent={respondent_id}"
        )

        system, survey, randomization = make_prompt(
            respondent, wave, prompts_dir=input.prompts_dir, demo_info=input.demo_info
        )

        response_path = get_response_filepath(
            base_runs_dir=input.base_output_dir,
            wave_id=wave_id,
            demo_id=demo_id,
            respondent_id=respondent_id,
            run_id=input.run_id,
            wave_subpath=wave_subpath,
        )
        if response_path.exists():
            _log(f"[SKIP] Already exists: {response_path}")
            return 0.0, None

        start_time = time.perf_counter()
        llm_out = llm.call_structured_json(
            prompt=survey,
            system_prompt=system,
            response_model=schema,
            questions=questions,
        )
        elapsed = time.perf_counter() - start_time

        metrics = LLMCallMetrics.build(llm_out, llm, input.run_id, elapsed)
        cost = metrics.input_cost + metrics.output_cost

        if llm_out.success:
            answer = response_to_dict(llm_out.output)
            sr = SingleResponse(
                wave_id=wave_id,
                demo_id=demo_id,
                respondent=respondent,
                success=True,
                answer=answer,
                metrics=metrics,
                randomization=randomization if randomization else None,
            )
            response_path.parent.mkdir(parents=True, exist_ok=True)
            with response_path.open("w", encoding="utf-8") as f:
                json.dump(asdict(sr), f, ensure_ascii=False, indent=4)
            _log(f"[OK]   Saved → {response_path}")
            return cost, None
        else:
            fail_msg = (
                f"Failed: wave={wave_id}  demo={demo_id}  respondent={respondent_id}"
            )
            _log(f"[FAIL] {fail_msg}")
            return cost, fail_msg

    # ── dispatch ───────────────────────────────────────────────────────
    total_cost = 0.0
    fail_ids: list[str] = []

    if max_workers <= 1:
        for respondent in profiles:
            cost, fail = process_respondent(respondent)
            total_cost += cost
            if fail:
                fail_ids.append(fail)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_respondent, r): r for r in profiles
            }
            for future in as_completed(futures):
                try:
                    cost, fail = future.result()
                except Exception as exc:
                    r = futures[future]
                    fail = (
                        f"Failed: wave={wave_id}  demo={demo_id} "
                        f"respondent={r.respondent_id}  exception={exc}"
                    )
                    cost = 0.0
                    _log(f"[ERROR] {fail}")
                total_cost += cost
                if fail:
                    fail_ids.append(fail)

    return total_cost, fail_ids


def run_all_waves(
    run_id: str,
    model_name: str,
    config_dir: str = "config",
    base_output_dir: str = "runs",
    data_dir: str = "data/use_data",
    prompts_dir: str = "prompts",
    base_url: str = DEFAULT_BASE_URL,
    wave_ids: list[str] | None = None,
    wave_workers: int = 1,
    respondent_workers: int = 1,
    temperature: float = 0.0,
) -> tuple[float, list[str]]:
    """Run the survey for every wave in data/use_data/index.jsonl.

    For each wave the demographic_id defaults to the wave_id (the CSV at
    data/demographic/{wave_id}.csv contains that wave's respondents).

    Parameters
    ----------
    run_id : str
        Identifier for this run, e.g. "gpt-4o-mini_0".
    model_name : str
        Which LLM to call.
    wave_ids : list[str] | None
        Restrict to these waves. If None, run every wave listed in the index.
    wave_workers : int
        Number of waves to process concurrently (default 1 = sequential).
        Each wave runs in its own thread, so this multiplies with
        *respondent_workers* for the total number of in-flight API calls.
    respondent_workers : int
        Number of respondents to survey concurrently *within* each wave
        (forwarded to ``run_one_wave``).

    Returns
    -------
    (grand_total_cost, all_fail_ids)
    """
    index = load_wave_index(data_dir)
    if wave_ids is not None:
        index = [e for e in index if e["wave_id"] in wave_ids]
        missing = set(wave_ids) - {e["wave_id"] for e in index}
        if missing:
            _log(f"[WARN] wave_ids not found in index: {sorted(missing)}")

    grand_total = 0.0
    all_fails: list[str] = []

    def _run_entry(entry: dict) -> tuple[float, list[str], str]:
        wave_id = entry["wave_id"]
        _log(
            f"\n========== wave={wave_id}  n={entry.get('total_n')} =========="
        )
        wave_input = RunOneWaveInput(
            wave_id=wave_id,
            demographic_id=wave_id,
            run_id=run_id,
            model_name=model_name,
            config_dir=config_dir,
            base_output_dir=base_output_dir,
            data_dir=data_dir,
            prompts_dir=prompts_dir,
            base_url=base_url,
            temperature=temperature,
        )
        cost, fails = run_one_wave(wave_input, max_workers=respondent_workers)
        _log(f"[SUMMARY] wave={wave_id}  cost=${cost:.4f}  fails={len(fails)}")
        return cost, fails, wave_id

    if wave_workers <= 1:
        for entry in index:
            cost, fails, _ = _run_entry(entry)
            grand_total += cost
            all_fails.extend(fails)
    else:
        with ThreadPoolExecutor(max_workers=wave_workers) as executor:
            futures = {executor.submit(_run_entry, e): e for e in index}
            for future in as_completed(futures):
                try:
                    cost, fails, wave_id = future.result()
                except Exception as exc:
                    wave_id = futures[future].get("wave_id", "?")
                    _log(f"[ERROR] wave={wave_id}  exception={exc}")
                    cost, fails = 0.0, [f"wave={wave_id} crashed: {exc}"]
                grand_total += cost
                all_fails.extend(fails)

    _log(f"\n========== GRAND TOTAL ==========")
    _log(f"cost     = ${grand_total:.4f}")
    _log(f"failures = {len(all_fails)}")
    return grand_total, all_fails


# ── CLI entrypoint ──────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Pew ATP surveys through an LLM, either for a single "
                    "wave or for every wave listed in data/use_data/index.jsonl."
    )
    p.add_argument(
        "--model-name", required=True,
        choices=[
            "claude-4-5-sonnet",
            "gemini-2.5-pro",
            "gemini-2.0-flash-001",
            "gpt-4.omini",
        ],
        help="Which LLM to call.",
    )
    p.add_argument(
        "--run-id", required=True,
        help="Run identifier, e.g. 'gpt-4.omini_0'. Used as the output filename.",
    )
    p.add_argument(
        "--wave-id", default=None,
        help="If set, run only this wave. Otherwise run every wave in the index.",
    )
    p.add_argument(
        "--demographic-id", default=None,
        help="Override the demographic CSV to use (default: same as wave-id). "
             "E.g. pass 'test' to use data/demographic/test.csv.",
    )
    p.add_argument(
        "--demographic-ids", nargs="+", default=None, metavar="DEMO_ID",
        help="Run the survey for multiple demographic groups sequentially. "
             "Each DEMO_ID refers to data/demographic/{DEMO_ID}.csv. "
             "Overrides --demographic-id when provided.",
    )
    p.add_argument(
        "--base-url", default=None,
        help="API base URL (default: read AI_API_BASE_URL from env file, "
             "or fall back to OpenRouter).",
    )
    p.add_argument(
        "--env-file", default="_env.json",
        help="Path to JSON file containing API keys / base URL (default: _env.json).",
    )
    p.add_argument("--config-dir",      default="config")
    p.add_argument("--base-output-dir", default="runs")
    p.add_argument("--data-dir",        default="data/use_data")
    p.add_argument("--prompts-dir",     default="prompts")
    p.add_argument(
        "--respondent-workers", type=int, default=1, metavar="N",
        help="Number of respondents to survey concurrently within each wave "
             "(default: 1 = sequential).  Increase to parallelise API calls.",
    )
    p.add_argument(
        "--wave-workers", type=int, default=1, metavar="N",
        help="Number of waves to run concurrently (default: 1 = sequential). "
             "Only relevant when no --wave-id is given (run_all_waves path). "
             "Total concurrent API calls = wave-workers × respondent-workers.",
    )
    p.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature for the LLM (default: 0.0 = deterministic). "
             "Set to 1.0 for meaningful variance across runs.",
    )
    p.add_argument(
        "--no-demo-info", action="store_true", default=False,
        help="Omit demographic profile from the system prompt. The demo CSV is "
             "still used to obtain the field dates; 1000 blank respondents are "
             "synthesised in place of the real respondent list.",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    # Load API keys and base URL from the env file first, then let explicit
    # --base-url override if provided.
    setup_keys(args.env_file)
    base_url = args.base_url or os.environ.get("AI_API_BASE_URL") or DEFAULT_BASE_URL

    # Resolve the list of demographic IDs to process.
    # --demographic-ids takes precedence over --demographic-id.
    if args.demographic_ids:
        demo_id_list = args.demographic_ids
    elif args.demographic_id:
        demo_id_list = [args.demographic_id]
    elif args.wave_id:
        demo_id_list = [args.wave_id]
    else:
        demo_id_list = [None]  # None → run_all_waves uses each wave's own demo

    if args.wave_id:
        grand_cost = 0.0
        grand_fails: list[str] = []

        for demo_id in demo_id_list:
            effective_demo = demo_id or args.wave_id
            _log(
                f"\n========== wave={args.wave_id}  demo={effective_demo} =========="
            )
            wave_input = RunOneWaveInput(
                wave_id=args.wave_id,
                demographic_id=effective_demo,
                run_id=args.run_id,
                model_name=args.model_name,
                config_dir=args.config_dir,
                base_output_dir=args.base_output_dir,
                data_dir=args.data_dir,
                prompts_dir=args.prompts_dir,
                base_url=base_url,
                temperature=args.temperature,
                demo_info=not args.no_demo_info,
            )
            cost, fails = run_one_wave(wave_input, max_workers=args.respondent_workers)
            grand_cost += cost
            grand_fails.extend(fails)
            _log(
                f"[DONE] wave={args.wave_id}  demo={effective_demo}  "
                f"cost=${cost:.4f}  fails={len(fails)}"
            )

        if len(demo_id_list) > 1:
            _log(f"\n========== ALL DEMOS DONE ==========")
            _log(f"total cost = ${grand_cost:.4f}")
            _log(f"total fails = {len(grand_fails)}")
    else:
        run_all_waves(
            run_id=args.run_id,
            model_name=args.model_name,
            config_dir=args.config_dir,
            base_output_dir=args.base_output_dir,
            data_dir=args.data_dir,
            prompts_dir=args.prompts_dir,
            base_url=base_url,
            wave_workers=args.wave_workers,
            respondent_workers=args.respondent_workers,
            temperature=args.temperature,
        )


if __name__ == "__main__":
    main()
