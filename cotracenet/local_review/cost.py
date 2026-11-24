"""Audit single-image inference by executing the archived implementation on a training image.

Arithmetic is counted per output element. One add/subtract/multiply/divide is one
FLOP; k-term dot products use k multiplications and k-1 additions. Weighted
histogram increments are additions. Special functions, comparisons, sign/abs,
integer arithmetic, unweighted histogram increments and data movement are not
FLOPs. No CPU instruction counts or hardware FLOP claims are implied.
"""

import argparse
import ast
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
import hashlib
import inspect
import json
import operator
from pathlib import Path
import sys
import textwrap

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode


records = defaultdict(Counter)
stage = "unassigned"


@contextmanager
def scope(name):
    global stage
    old = stage
    stage = name
    try:
        yield
    finally:
        stage = old


def add(category, count):
    records[stage][category] += int(count)


def numpy_value(x):
    return isinstance(x, (np.ndarray, np.generic))


OPS = dict(
    Add=operator.add,
    Sub=operator.sub,
    Mult=operator.mul,
    Div=operator.truediv,
    FloorDiv=operator.floordiv,
    Pow=operator.pow,
    MatMult=operator.matmul,
    BitAnd=operator.and_,
    BitXor=operator.xor,
    BitOr=operator.or_,
    Gt=operator.gt,
    GtE=operator.ge,
    Lt=operator.lt,
    LtE=operator.le,
    Eq=operator.eq,
    NotEq=operator.ne,
    Invert=operator.invert,
    USub=operator.neg,
)
COMPARISONS = {"Gt", "GtE", "Lt", "LtE", "Eq", "NotEq", "BitAnd", "BitXor", "BitOr", "Invert"}


def traced_op(name, *args):
    out = OPS[name](*args)
    if not numpy_value(out) or not any(numpy_value(x) for x in args):
        return out
    if name in COMPARISONS:
        add("comparison_or_boolean", out.size)
    elif name == "MatMult":
        k = args[-1].shape[0]
        add("fp_arithmetic", (2 * k - 1) * out.size)
    else:
        add("fp_arithmetic" if out.dtype.kind == "f" else "integer_arithmetic", out.size)
    return out


def traced_call(name, func, *args, **kw):
    out = func(*args, **kw)
    ref = getattr(func, "__self__", None)
    value = ref if numpy_value(ref) else args[0] if args else None
    if not numpy_value(value) or not numpy_value(out):
        return out
    n, m = value.size, out.size
    arithmetic = "fp_arithmetic" if out.dtype.kind == "f" else "integer_arithmetic"
    if name == "bincount":
        add(
            (
                "weighted_histogram_add"
                if kw.get("weights") is not None
                else "integer_histogram_increment"
            ),
            n,
        )
    elif name == "diff":
        add(arithmetic, m)  # all model calls use the default first difference
    elif name == "cumsum":
        axis = args[0] if numpy_value(ref) and args else kw["axis"]
        add(arithmetic, n - n // value.shape[axis])
    elif name == "sum":
        add(arithmetic, n - m)
    elif name == "mean":
        add(arithmetic, n - m + m)
    elif name == "std":
        # Population SD: mean, sample centering, squaring, mean, sqrt.
        add(arithmetic, 4 * n)
        add("sqrt", m)
    elif name in ("quantile", "percentile", "median"):
        add("order_statistic_calls", 1)
        add("order_statistic_input_elements", n)
    elif name in ("minimum", "maximum"):
        add("comparison_or_boolean", m)
    elif name in ("max", "min", "argmax", "argmin", "any"):
        add("comparison_or_boolean", n - m)
    elif name == "clip":
        low = kw.get("a_min", args[0] if numpy_value(ref) else args[1])
        high = kw.get(
            "a_max",
            (
                args[1]
                if numpy_value(ref) and len(args) > 1
                else args[2] if not numpy_value(ref) and len(args) > 2 else None
            ),
        )
        add("comparison_or_boolean", m * (int(low is not None) + int(high is not None)))
    elif name in ("abs", "absolute", "sign"):
        add("float_abs_or_sign" if value.dtype.kind == "f" else "integer_abs_or_sign", m)
    elif name in ("sqrt", "tanh", "expit"):
        add(name, m)
    return out


CALLS = {
    "bincount",
    "diff",
    "cumsum",
    "sum",
    "mean",
    "std",
    "quantile",
    "percentile",
    "median",
    "minimum",
    "maximum",
    "max",
    "min",
    "any",
    "clip",
    "abs",
    "absolute",
    "sign",
    "sqrt",
    "tanh",
    "expit",
    "argmax",
    "argmin",
}


class Arithmetic(ast.NodeTransformer):
    def wrap(self, node, helper, tag, args, kw=()):
        wrapped = ast.Call(ast.Name(helper, ast.Load()), [ast.Constant(tag), *args], list(kw))
        return ast.copy_location(wrapped, node)

    def visit_Call(self, node):
        self.generic_visit(node)
        name = getattr(node.func, "attr", getattr(node.func, "id", None))
        if name in CALLS:
            return self.wrap(node, "_cost_call", name, [node.func, *node.args], node.keywords)
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        tag = type(node.op).__name__
        return self.wrap(node, "_cost_op", tag, [node.left, node.right]) if tag in OPS else node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        tag = type(node.op).__name__
        return self.wrap(node, "_cost_op", tag, [node.operand]) if tag in OPS else node

    def visit_Compare(self, node):
        self.generic_visit(node)
        tag = type(node.ops[0]).__name__
        if len(node.ops) == 1 and tag in OPS:
            return self.wrap(node, "_cost_op", tag, [node.left, node.comparators[0]])
        return node

    def visit_FunctionDef(self, node):
        node.decorator_list = []
        return self.generic_visit(node)


def instrument(module, names, group):
    for name in names.split():
        fn = getattr(module, name)
        code = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        code = ast.fix_missing_locations(Arithmetic().visit(code))
        env = vars(module)
        env.update(_cost_op=traced_op, _cost_call=traced_call)
        exec(compile(code, inspect.getfile(fn), "exec"), env)
        transformed = env[name]

        def grouped(*args, _fn=transformed, **kwargs):
            with scope(group):
                return _fn(*args, **kwargs)

        setattr(module, name, grouped)


class TorchArithmetic(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if not isinstance(out, torch.Tensor):
            return out
        name = func._schema.name.split("::")[-1]
        source = args[0] if args else None
        m = out.numel()
        n = source.numel() if isinstance(source, torch.Tensor) else 0
        if name in ("add", "sub", "mul", "div"):
            add("fp_arithmetic" if out.is_floating_point() else "integer_arithmetic", m)
        elif name == "pow":
            assert args[1] == 2
            add("fp_arithmetic", m)
        elif name in ("matmul", "mm", "mv", "dot"):
            add("fp_arithmetic", (2 * source.shape[-1] - 1) * m)
        elif name == "sum":
            add("fp_arithmetic", n - m)
        elif name == "mean":
            add("fp_arithmetic", n)
        elif name == "linalg_vector_norm":
            assert len(args) < 2 or args[1] == 2
            add("fp_arithmetic", 2 * n - m)
            add("sqrt", m)
        elif name in ("tanh", "sqrt"):
            add(name, m)
        elif name == "amax":
            add("comparison_or_boolean", n - m)
        elif name in ("gt", "eq", "bitwise_xor", "__xor__", "relu"):
            add("comparison_or_boolean", m)
        elif name == "clamp":
            add("comparison_or_boolean", 2 * m)
        elif name not in (
            "lift_fresh",
            "view",
            "reshape",
            "slice",
            "select",
            "index_select",
            "permute",
            "t",
            "where",
            "unsqueeze",
            "cat",
            "stack",
            "expand",
            "alias",
            "detach",
            "to",
            "_to_copy",
            "clone",
            "empty",
            "empty_strided",
            "copy_",
            "ones",
            "as_strided",
            "squeeze",
            "resolve_conj",
            "resolve_neg",
        ):
            raise ValueError(f"Unaccounted torch operation {func}")
        return out


def setup(args):
    from cotracenet import triples, fourth, regions, detector, lite
    from cotracenet.binocular import attention, prepare, readout, predict
    from . import patch_features

    detector_instance = predict.PolicyDetector(args.attention, args.policy, args.train)
    from .active_review import ReviewModel

    current = ReviewModel(args.current / args.train)
    images = [args.image]
    if args.manifest:
        with args.manifest.open(encoding="utf-8-sig", newline="") as file:
            pairs = [row for row in csv.DictReader(file) if row["split"] == "train"]
        pairs.sort(
            key=lambda row: hashlib.sha256(("cost-audit-73/" + row["source"]).encode()).digest()
        )
        images += [
            args.manifest.parent / row[role]
            for row in pairs[: args.pairs]
            for role in ("cover", "stego")
        ]
    reference = {path: detector_instance.predict(path)["score"] for path in images}
    instrument(triples, "_quantize _triple_counts _merge extract_features", "triple_residuals")
    instrument(fourth, "local_texture counts symmetric_blocks features", "texture_fourth_order")
    instrument(regions, "uniform histogram sections features", "cost_partition_fourth_order")
    instrument(detector, "image_features", "descriptor_square_root")
    lite.image_features = detector.image_features
    instrument(prepare, "hellinger normalize_root bin_statistics", "binocular_evidence")
    instrument(attention, "build_inputs forward", "binocular_gates")
    instrument(
        readout,
        "state scale_design reviewer gate_features probabilities decide",
        "old_policy_readout",
    )
    instrument(patch_features, "double_midrank select_tile", "local_tile_selection")
    instrument(patch_features, "describe_tile", "local_tile_histograms")
    return detector_instance, current, images, reference


def trace_base(detector_instance, current, path):
    from cotracenet.binocular import prepare
    from . import patch_features
    from cotracenet.data import load_image

    with torch.inference_mode(), TorchArithmetic():
        with scope("anchor"):
            output = detector_instance.predict(path)
    # predict.py performs a NumPy affine projection and two in-place centers in
    # its top-level control flow (intentionally left uninstrumented). Count once.
    k = len(detector_instance.backbone.weights)
    with scope("lite_affine"):
        add("fp_arithmetic", (2 * k - 1) + 1)
    with scope("binocular_evidence"):
        add("fp_arithmetic", 2 * len(prepare.MATCH))
    gray = load_image(path)
    tile = patch_features.select_tile(gray)
    p = patch_features.describe_tile(gray, tile)
    # Run both locked new operations, even for a fast-exit image, to measure the conditional path.
    s = output["score"]
    assert np.isfinite(current.refine(s, p))
    current.should_query(s)
    return s


def costs(rows):
    return sum(v.get("fp_arithmetic", 0) + v.get("weighted_histogram_add", 0) for v in rows)


def path_costs():
    # 158 standardizations, then 158-dim affine+bias and bounded residual.
    # The review standardization includes both the selected image descriptor and anchor cues.
    with scope("local_reviewer"):
        add("fp_arithmetic", 2 * 158 + 2 * 158 + 3)
        add("comparison_or_boolean", 2 * 158)
        add("tanh", 1)
        add("float_abs_or_sign", 1)
    with scope("pre_gate"):
        add("float_abs_or_sign", 1)
        add("searchsorted_five_thresholds", 1)
        add("comparison_or_boolean", 2)
        add("integer_arithmetic", 2)
    base = [
        v
        for k, v in records.items()
        if k not in ("local_tile_histograms", "local_tile_selection", "local_reviewer")
    ]
    fixed = base + [records["local_tile_histograms"], records["local_reviewer"]]
    active = fixed + [records["local_tile_selection"]]
    rows = {}
    for key, parts in (("fast", base), ("fixed_review", fixed), ("active_review", active)):
        counts = sum((Counter(row) for row in parts), Counter())
        rows[key] = {"FLOPs": costs(parts), "counts": dict(counts)}
    return rows


def expected(root, domain, rows):
    summary = {}
    for target in ("WOW", "HILL"):
        path = root / domain / (domain + "-to-" + target + "-local.npz")
        if not path.exists():
            continue
        target_rows = {}
        with np.load(path, allow_pickle=False) as f:
            for gate in ("gate", "uncertainty"):
                actions = np.asarray(f[gate], dtype=bool)
                assert actions.shape == (6000,)
                p = actions.mean()
                modes = {}
                for key in ("fixed_review", "active_review"):
                    a, b = rows["fast"], rows[key]
                    expected_counts = {
                        k: v + p * (b["counts"][k] - v) for k, v in a["counts"].items()
                    }
                    if gate == "uncertainty":
                        # A confidence threshold uses abs+compare; no lookup or bin arithmetic.
                        expected_counts.pop("searchsorted_five_thresholds")
                        expected_counts["comparison_or_boolean"] -= 1
                        expected_counts["integer_arithmetic"] -= 2
                    modes[key] = {
                        "FLOPs": a["FLOPs"] + p * (b["FLOPs"] - a["FLOPs"]),
                        "counts": expected_counts,
                    }
                target_rows[gate] = {
                    "images": len(actions),
                    "queries": int(actions.sum()),
                    "query_rate": float(p),
                    "expected_per_image": modes,
                }
        summary[target] = target_rows
    return summary


def write_summary(folder):
    reports = {
        d: json.loads((folder / ("cost-" + d + ".json")).read_text(encoding="utf8"))
        for d in ("WOW", "HILL")
    }
    rows = []
    for train, r in reports.items():
        for test, results in r["expected_on_locked_test_actions"].items():
            learned, check = results["gate"], results["uncertainty"]
            base = learned["expected_per_image"]["active_review"]
            active = check["expected_per_image"]["active_review"]
            fixed = check["expected_per_image"]["fixed_review"]
            rows.append(
                {
                    "train": train,
                    "test": test,
                    "images": check["images"],
                    "reward_gate_queries": learned["queries"],
                    "reward_gate_FLOPs": base["FLOPs"],
                    "low_conf_queries": check["queries"],
                    "low_conf_query_rate": check["query_rate"],
                    "low_conf_selected_FLOPs": active["FLOPs"],
                    "low_conf_fixed_FLOPs": fixed["FLOPs"],
                    "low_conf_selected_integer_arithmetic": active["counts"]["integer_arithmetic"],
                    "low_conf_fixed_integer_arithmetic": fixed["counts"]["integer_arithmetic"],
                    "low_conf_selected_integer_histogram_votes": active["counts"][
                        "integer_histogram_increment"
                    ],
                    "low_conf_selected_sqrt": active["counts"]["sqrt"],
                    "low_conf_selected_tanh": active["counts"]["tanh"],
                    "low_conf_selected_sigmoid": active["counts"]["expit"],
                }
            )
    if len(rows) != 4:
        raise ValueError("Four locked train/test directions are required")
    with (folder / "cost-summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("attention", "policy", "current", "image", "out"):
        cli.add_argument("--" + name, type=Path, required=True)
    cli.add_argument("--train", choices=("WOW", "HILL"), default="HILL")
    cli.add_argument(
        "--manifest", type=Path, help="Optional training-split fold to verify path cost invariance"
    )
    cli.add_argument("--pairs", type=int, default=16)
    args = cli.parse_args()
    torch.set_num_threads(1)
    model, current, paths, reference = setup(args)
    measurements = []
    errors = []
    for path in paths:
        records.clear()
        score = trace_base(model, current, path)
        errors.append(abs(score - reference[path]))
        measurements.append(path_costs())
        if path == paths[0]:
            raw = {k: dict(v) for k, v in records.items()}
    assert max(errors) < 1e-10, max(errors)
    rows = measurements[0]
    ranges = {
        k: {
            m: [
                min(r[k]["counts"][m] for r in measurements),
                max(r[k]["counts"][m] for r in measurements),
            ]
            for m in rows[k]["counts"]
        }
        for k in rows
    }
    flop_range = {
        k: [min(r[k]["FLOPs"] for r in measurements), max(r[k]["FLOPs"] for r in measurements)]
        for k in rows
    }
    report = {
        "input": "one 256x256 grayscale image",
        "train": args.train,
        "probe": "first designated training image; variability audited on independent preselected training pairs",
        "training_images": len(paths),
        "instrumented_vs_native_max_score_difference": float(max(errors)),
        "path_FLOP_ranges": flop_range,
        "all_count_ranges": ranges,
        "raw_per_stage": raw,
        "single_image_paths": rows,
        "expected_on_locked_test_actions": expected(args.current, args.train, rows),
        "counting": "Add/subtract/multiply/divide cost one scalar operation, including weighted histogram adds. "
        "Separate: sqrt/tanh/expit, abs/sign, comparisons, integer arithmetic and histogram increments. "
        "No input parsing, loads/stores, casts, table lookups, hardware instructions, quantile/median "
        "selection or interpolation internals. Algorithm-level reduction uses n-1 adds. "
        "Includes actually executed archived code's Dperm, triple context and extra supervised probability.",
        "base_parameters": {
            "lite": len(model.backbone.weights) + 1,
            "SE": len(model.backbone.gates["se"][0]),
            "brain": len(model.backbone.gates["brain"][0]),
            "old_reviewer": len(model.parameters["rw"]),
            "old_actor": len(model.parameters["wp"]),
            "extra_control_only_actor_executed": len(model.parameters["ws"]),
        },
        "parameter_scope": "16,828 upstream output coefficients + 159 local-head coefficients + "
        "12 fitted bin rewards = 16,999 decision-relevant fitted scalars for a single-arm release. "
        "Five bin edges and one selected cutoff are additional. Current deployment loads the requested "
        "patch head on demand. Evaluating other arms in the same process can load their own control "
        "heads; those weights are not part of the single-arm model. The archived upstream predictor "
        "also evaluates 13 supervised comparison coefficients whose output is unused.",
        "new_parameters": {
            "local_head_including_intercept": int(current.head["w"].size),
            "fitted_rewards": int(current.values.size),
            "quantile_edges": int(current.edges.size),
            "policy_threshold": 1,
            "additional_low_confidence_control_threshold": 1,
        },
        "frozen_noncoefficient_state": {
            "selected_feature_indices": int(model.backbone.indices.size),
            "archived_source_feature_mean_before_prefolding": int(
                model.backbone.dense_weights.numel()
            ),
            "prefolded_center_terms_per_family": int(
                model.backbone.texture_center.size + model.backbone.cost_center.size
            ),
            "attention_standardization_by_gate": {
                k: sum(v.size for v in info[1].values()) for k, info in model.backbone.gates.items()
            },
            "old_head_standardizers": {
                k: int(v.size)
                for k, v in model.parameters.items()
                if k in ("gmu", "gsd", "rmu", "rsd")
            },
            "local_standardizers": int(current.head["mu"].size + current.head["sd"].size),
        },
    }
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print({k: v["FLOPs"] for k, v in rows.items()})


if __name__ == "__main__":
    if sys.argv[1:2] == ["summarize"]:
        parser = argparse.ArgumentParser(
            description="Summarize the four locked action rates and corresponding mean arithmetic"
        )
        parser.add_argument("folder", type=Path)
        write_summary(parser.parse_args(sys.argv[2:]).folder)
    else:
        main()
