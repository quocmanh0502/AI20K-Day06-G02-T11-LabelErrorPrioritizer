"""YOUR PoC PIPELINE — this is the file your group owns.

Lab contract = 5 automated steps, run by `python -m src.main ...`:
    01 load      input + provenance check (manifest.csv)
    02 process   your algorithm / model (the research part)
    03 rank      prioritise for a fixed review budget B
    04 review    human-in-the-loop: queue -> reviewer decisions -> overrides + audit log
    05 decide    GO / HOLD / REWORK (or track equivalent) with reason_code + owner, write outputs

Required output: out/<pred_file> with the columns listed for your track in
docs/tracks/TXX.md (also in src/metrics/tracks.py::TRACKS). Then evaluate with
    python -m src.evaluate --track tXX --pred out/<pred_file>

Rules: never read fixtures/*/gt/ here (the GT guard will stop you); seed everything via ctx.rng;
no network calls; fail loudly on missing models instead of downloading.
See src/examples/random_baseline.py for a minimal end-to-end wiring example.
"""
from __future__ import annotations
import pandas as pd

from .common.context import RunContext
from .common.decision import Decision, write_decision
from .common.manifest import verify_manifest
from .common.review import load_review_decisions, make_review_queue, write_review_queue
from .metrics.tracks import get_track
from .t11_review import apply_label_fixes, build_workbench_items, decide_policy, render_workbench, review_stats
from .t11_scoring import DEFAULT_PARAMS, build_scores


# ---------------------------------------------------------------- 01 load
def load(ctx: RunContext) -> dict:
    if ctx.track != "t11":
        raise NotImplementedError(f"this group pipeline only implements t11, not {ctx.track}")
    manifest = ctx.input_dir / "manifest.csv"
    problems = verify_manifest(manifest) if manifest.exists() else [{"problem": "no_manifest"}]
    if problems:
        ctx.audit.log(actor="pipeline", action="provenance_problems", item_id=str(manifest), after=problems[:20])
        if ctx.config.get("strict_provenance", True):
            raise ValueError(f"provenance check failed ({len(problems)} problems), first: {problems[:3]}")
    data = {"manifest": pd.read_csv(manifest) if manifest.exists() else None}
    d = ctx.input_dir
    data["ann"] = pd.read_csv(d / "annotations.csv")
    data["pred"] = pd.read_csv(d / "model_predictions.csv")
    data["images"] = pd.read_csv(d / "images.csv")
    ctx.stats["n_ann"] = len(data["ann"])
    ctx.stats["n_pred"] = len(data["pred"])

    return data


# ---------------------------------------------------------------- 02 process
def process(ctx: RunContext, data: dict) -> pd.DataFrame:
    params = {k: ctx.param(k) for k in DEFAULT_PARAMS if ctx.param(k) is not None}
    items = build_scores(data["ann"], data["pred"], data["images"], params)
    ctx.stats["t11_params"] = {**DEFAULT_PARAMS, **params}
    ctx.stats["n_flagged"] = int((items.score > 0.2).sum())
    return items



# ---------------------------------------------------------------- 03 rank
def rank(ctx: RunContext, scored: pd.DataFrame) -> pd.DataFrame:
    """Default: top-B by `score`. Adapt id/score/reason columns to your track."""
    budget = int(ctx.param("budget", 20))
    if "score" not in scored.columns:
        raise ValueError("rank(): process() output needs a numeric `score` column (higher = review first), "
                         "or adapt rank() to your track")
    q = make_review_queue(scored, id_col=scored.columns[0], score_col="score", budget=budget,
                          reason_col="reason" if "reason" in scored.columns else None,
                          evidence_col="path" if "path" in scored.columns else None)
    write_review_queue(q, ctx.out("review_queue.csv"))
    return q


# ---------------------------------------------------------------- 04 review
def review(ctx: RunContext, scored: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    """Human-in-the-loop.

    Lần 1 (chưa có file duyệt): sinh out/<run>/contact_sheet.html để người kiểm tra duyệt top-B rồi Export CSV.
    Lần 2 (có out/<run>/review_queue_reviewed.csv): áp sửa nhãn lên BẢN SAO annotations, ghi
    corrected_annotations.csv + label_changes.csv, mọi quyết định vào audit_log.jsonl.
    `scored` (ranking) được trả nguyên vẹn: review không đổi điểm/thứ tự đã nộp.
    """
    d = ctx.input_dir
    ann = pd.read_csv(d / "annotations.csv")
    pred = pd.read_csv(d / "model_predictions.csv")
    images = pd.read_csv(d / "images.csv")

    reviewed_path = ctx.output_dir / "review_queue_reviewed.csv"
    reviewed = load_review_decisions(reviewed_path) if reviewed_path.exists() else None

    items = build_workbench_items(queue, ann, pred, images, d, reviewed)
    render_workbench(items, ctx.out("contact_sheet.html"))

    changes = None
    if reviewed is not None:
        fixed, changes = apply_label_fixes(ann, reviewed, ctx.audit)
        fixed.to_csv(ctx.out("corrected_annotations.csv"), index=False)
        changes.to_csv(ctx.out("label_changes.csv"), index=False)
        ctx.stats["n_ann_before"], ctx.stats["n_ann_after"] = len(ann), len(fixed)
    ctx.stats.update(review_stats(reviewed, changes))
    return scored


# ---------------------------------------------------------------- 05 decide
def decide(ctx: RunContext, final: pd.DataFrame) -> Decision:
    """Chính sách chốt trước (src/t11_review.py::decide_policy): HOLD / REWORK / GO."""
    status, code, reason = decide_policy(ctx.stats)
    evidence = [str(ctx.out(f)) for f in ("ranking.csv", "review_queue.csv", "contact_sheet.html",
                                          "review_queue_reviewed.csv", "label_changes.csv",
                                          "corrected_annotations.csv", "audit_log.jsonl") if ctx.out(f).exists()]
    keys = ("n_reviewed", "n_confirmed", "n_false_positive", "n_escalated", "n_fixed")
    return Decision(status=status, reason_code=code, reason=reason, owner=ctx.config.get("owner", "TODO-owner"),
                    scope=f"{ctx.track}:annotations.csv", evidence=evidence,
                    metrics={k: ctx.stats.get(k) for k in keys})


def run(ctx: RunContext) -> None:
    spec = get_track(ctx.track)
    data = load(ctx)
    scored = process(ctx, data)
    ctx.stats["n_items"] = len(scored)
    queue = rank(ctx, scored)
    final = review(ctx, scored, queue)
    final[spec.pred_columns + [c for c in final.columns if c not in spec.pred_columns]].to_csv(ctx.out(spec.pred_file), index=False)
    write_decision(decide(ctx, final), ctx.out("decision.json"), ctx.out("decision_log.csv"))
