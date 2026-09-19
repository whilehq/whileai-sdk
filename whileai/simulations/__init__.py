"""Generate training conversations from an intent or an agent.

Input is a system prompt, tools, or both. The SDK sets a world from that
spec, writes human requests across a grid (simple, complex, vague, ordinary,
malicious), and rolls the agent. Grade 0/1 later. Optimize for post-training.

Five calls, spec to gated dataset:

    import whileai.simulations as wai
    data = wai.simulate(agent="openai:gpt-4.1-mini", tools=my_tools,
                        system_prompt=my_policy, mode="rl",
                        situations=200, repeats=8)                  # generate
    scored = data.grade(judge=my_judge)                             # grade 0/1
    print(scored.pass_at); wai.judge_trust(scored.rows, judge=my_judge)  # trust
    rows, report = wai.optimize(scored, mode="rl")                  # prune
    wai.push_rows(rows, "my-agent-rl-v1", gate=True, mode="rl")     # publish, gated

Everything else exported here is one layer down from those five.
"""

from __future__ import annotations

import logging as _logging

from . import data, schema, simulation, verify
from .data import SimulationData, conversation, grade, grade_llm, llm_grade, rank
from .environment import build_tasks, export_environment, load_environment
from .export import (
    export_dataset,
    export_preference,
    export_training,
    loss_mask,
    to_trl,
    tool_call_roundtrip,
    training_rows,
)
from .generate.adapters import AgentProfile, claude_code, connect, inspect
from .generate.agents import hosted_model, local_model
from .generate.diversity import adaptive_allocator, allocator_slot_counts
from .generate.generator import ModelSimulator, write_scene_brief
from .generate.offline_agent import SEEDED_BEHAVIORS, World, seeded_agent, world
from .generate.scenarios import (
    build_dimensions,
    novelty,
    open_ended_probes,
    policy_sections,
    scenario_regions,
)
from .ingest.otel import rows_from_otel
from .ingest.platform import (
    agents,
    catalog,
    datasets,
    delete_empty_datasets,
    hf_publish,
    hf_publish_run,
    hf_status,
    import_hf,
    issue_delegated_credential,
    preview,
    profile,
    publish,
    pull,
    purge_agent,
    push_file,
    push_rows,
    refresh_delegated_credential,
    register_agent,
    unpublish,
    update_dataset,
)
from .ingest.platform import (
    delete as delete_dataset,
)
from .ingest.traces import (
    dimensions_from_traces,
    drop_leaky_rows,
    flaw_rows,
    format_trace_report,
    infer_harness,
    leakage_report,
    load_traces,
    mine_traces,
    opening_share,
    simulate_from_traces,
    split_pseudo_production,
    tools_from_traces,
    trace_report,
)
from .monitor import HackMonitor, format_hack_monitor
from .schema import (
    SCHEMA_VERSION,
    Calibration,
    Dataset,
    Judgment,
    Marker,
    Rollout,
    Task,
    calibration_of,
    from_row,
    to_row,
    validate,
)
from .score.agreement import judge_agreement
from .score.audit import audit_grades, format_audit
from .score.checklist import expected_outcome, outcome_check, privileged_context, task_checklist
from .score.curriculum import curriculum, format_curriculum, retire_solved
from .score.delta import delta_report, format_delta_report
from .score.grading import behavior_signature, conduct_grade
from .score.grounding import (
    argument_grounding,
    grounding_report,
    mark_grounding,
    ungrounded_arguments,
)
from .score.hack_scan import format_hack_scan, format_hack_scan_diff, hack_scan, hack_scan_diff
from .score.hygiene import (
    HACK_THRESHOLD,
    coverage_warnings,
    dedupe_groups,
    length_report,
    near_duplicate_prompts,
    reward_correlations,
)
from .score.judge_trust import format_judge_trust, judge_probes, judge_trust
from .score.judging import (
    ScoredData,
    build_preference_pairs,
    evaluate,
    normalize_judge_result,
    run_judge,
)
from .score.labels import annotator_agreement, attach_labels
from .score.logprobs import logprob_report, mean_kl, staleness_report
from .score.markers import STOCK_MARKERS, behavioral_markers, format_markers, mark_rows, row_markers
from .score.optimize import (
    DEFAULT_BAND,
    filter_rl_rows,
    group_signal,
    next_round,
    optimize,
    recommend,
    select_for_rl,
    select_for_sft,
    trim_out_of_band,
    trim_unanimous_groups,
)
from .score.pairwise import judge_pairs, pairwise_judge
from .score.passat import PassAt, pass_at
from .score.preflight import (
    FAILURE_CLASSES,
    classify_failure,
    coverage_gap,
    dataset_report,
    format_coverage_gap,
    preflight,
)
from .score.privileged import format_leak_report, leak_report
from .score.publish_gate import PublishGateError, calibrate, publish_gate
from .score.quality import rank_rows, score_row
from .score.reference import reference_logprobs
from .score.rubric import (
    Criterion,
    Rubric,
    attach_rubric,
    rubric_judge,
    rubric_of,
    write_rubrics,
)
from .score.spec import Spec, Trait, load_spec, spec_version, stamp_spec
from .score.stage import STAGES, format_stages, stage_of, stage_report, stamp_stage
from .score.stats import (
    compare_runs,
    decontaminate,
    detectable_effect,
    eval_variance,
    holdout_size,
    marker_names,
    marker_summary,
    metric_summary,
    task_key,
)
from .score.style import refusal_report, style_markers, style_report
from .score.trace import trace_flag_report, trace_flags, trace_markers
from .simulation import resolve_topology, simulate
from .training import (
    TrainerCallback,
    TrainingRun,
    attach_delta,
    attach_holdout,
    delete_model,
    delete_run,
    get_run,
    list_runs,
    models,
    reward_model,
    serve,
    train,
    training_run,
    unserve,
)
from .verify import Verifier, verifier
from .world.sandbox import MockEnvironment, WorldOptions

# Library convention: emit under "whileai.simulations", never configure
# the root logger. Callers opt in with logging.basicConfig() or a handler.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

__all__ = [
    "DEFAULT_BAND",
    "FAILURE_CLASSES",
    "HACK_THRESHOLD",
    "SCHEMA_VERSION",
    "SEEDED_BEHAVIORS",
    "STAGES",
    "STOCK_MARKERS",
    "AgentProfile",
    "Calibration",
    "Criterion",
    "Dataset",
    "HackMonitor",
    "Judgment",
    "Marker",
    "MockEnvironment",
    "ModelSimulator",
    "PassAt",
    "PublishGateError",
    "Rollout",
    "Rubric",
    "ScoredData",
    "SimulationData",
    "Spec",
    "Task",
    "TrainerCallback",
    "TrainingRun",
    "Trait",
    "Verifier",
    "World",
    "WorldOptions",
    "adaptive_allocator",
    "agents",
    "allocator_slot_counts",
    "annotator_agreement",
    "argument_grounding",
    "attach_delta",
    "attach_holdout",
    "attach_labels",
    "attach_rubric",
    "audit_grades",
    "behavior_signature",
    "behavioral_markers",
    "build_dimensions",
    "build_preference_pairs",
    "build_tasks",
    "calibrate",
    "calibration_of",
    "catalog",
    "classify_failure",
    "claude_code",
    "compare_runs",
    "conduct_grade",
    "connect",
    "conversation",
    "coverage_gap",
    "curriculum",
    "dataset_report",
    "datasets",
    "decontaminate",
    "dedupe_groups",
    "delete_dataset",
    "delete_empty_datasets",
    "delete_model",
    "delete_run",
    "delta_report",
    "detectable_effect",
    "dimensions_from_traces",
    "drop_leaky_rows",
    "eval_variance",
    "evaluate",
    "expected_outcome",
    "export_dataset",
    "export_environment",
    "export_preference",
    "export_training",
    "filter_rl_rows",
    "flaw_rows",
    "format_audit",
    "format_coverage_gap",
    "format_curriculum",
    "format_delta_report",
    "format_hack_monitor",
    "format_hack_scan",
    "format_hack_scan_diff",
    "format_judge_trust",
    "format_leak_report",
    "format_markers",
    "format_stages",
    "format_trace_report",
    "from_row",
    "get_run",
    "grade",
    "grade_llm",
    "grounding_report",
    "group_signal",
    "hack_scan",
    "hack_scan_diff",
    "hf_publish",
    "hf_publish_run",
    "hf_status",
    "holdout_size",
    "hosted_model",
    "import_hf",
    "infer_harness",
    "inspect",
    "issue_delegated_credential",
    "judge_agreement",
    "judge_pairs",
    "judge_probes",
    "judge_trust",
    "leak_report",
    "leakage_report",
    "length_report",
    "list_runs",
    "llm_grade",
    "load_environment",
    "load_spec",
    "load_traces",
    "local_model",
    "logprob_report",
    "loss_mask",
    "mark_grounding",
    "mark_rows",
    "marker_names",
    "marker_summary",
    "mean_kl",
    "metric_summary",
    "mine_traces",
    "models",
    "near_duplicate_prompts",
    "next_round",
    "normalize_judge_result",
    "novelty",
    "open_ended_probes",
    "opening_share",
    "optimize",
    "outcome_check",
    "pairwise_judge",
    "pass_at",
    "policy_sections",
    "preflight",
    "preview",
    "privileged_context",
    "profile",
    "publish",
    "publish_gate",
    "pull",
    "purge_agent",
    "push_file",
    "push_rows",
    "rank",
    "rank_rows",
    "recommend",
    "reference_logprobs",
    "refresh_delegated_credential",
    "refusal_report",
    "register_agent",
    "resolve_topology",
    "retire_solved",
    "reward_correlations",
    "reward_model",
    "row_markers",
    "rows_from_otel",
    "rubric_judge",
    "rubric_of",
    "run_judge",
    "scenario_regions",
    "score_row",
    "seeded_agent",
    "select_for_rl",
    "select_for_sft",
    "serve",
    "simulate",
    "simulate_from_traces",
    "spec_version",
    "split_pseudo_production",
    "stage_of",
    "stage_report",
    "staleness_report",
    "stamp_spec",
    "stamp_stage",
    "style_markers",
    "style_report",
    "task_checklist",
    "task_key",
    "to_row",
    "to_trl",
    "tool_call_roundtrip",
    "tools_from_traces",
    "trace_flag_report",
    "trace_flags",
    "trace_markers",
    "trace_report",
    "train",
    "training_rows",
    "training_run",
    "trim_out_of_band",
    "trim_unanimous_groups",
    "ungrounded_arguments",
    "unpublish",
    "unserve",
    "update_dataset",
    "validate",
    "verifier",
    "verify",
    "world",
    "write_rubrics",
    "write_scene_brief",
]
