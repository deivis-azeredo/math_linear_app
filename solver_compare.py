"""
solver_compare.py — MathModel AI
Solve a linear model in Continuous (LP) mode, Integer (MILP) mode, or compare both.

Model format (the same dict your AI step should produce):

model = {
    "name": "Production planning",
    "sense": "max",                       # "max" or "min"
    "variables": [
        {"name": "x1", "lb": 0, "ub": None, "integer": True},   # integer in MILP mode
        {"name": "x2", "lb": 0, "ub": None, "integer": False},  # stays continuous always
    ],
    "objective": {"x1": 3, "x2": 5},
    "constraints": [
        {"name": "Machine A", "coeffs": {"x1": 1},          "sense": "<=", "rhs": 4},
        {"name": "Machine B", "coeffs": {"x2": 2},          "sense": "<=", "rhs": 12},
        {"name": "Labor",     "coeffs": {"x1": 3, "x2": 2}, "sense": "<=", "rhs": 18},
    ],
}
"""

import time
import pulp

LP = "LP"
MILP = "MILP"
TOL = 1e-6


def _make_var(prob, name, lb, ub, cat):
    """Create a variable on both old (LpVariable) and new (PuLP 4.x add_variable) APIs."""
    if hasattr(prob, "add_variable"):  # PuLP 4.x
        return prob.add_variable(name, lowBound=lb, upBound=ub, cat=cat)
    return pulp.LpVariable(name, lowBound=lb, upBound=ub, cat=cat)


def solve_model(model: dict, mode: str = LP) -> dict:
    """Solve the model.

    mode = 'LP'   -> every variable is continuous.
    mode = 'MILP' -> variables with "integer": True become integer; those with
                     "integer": False stay continuous. If the key is missing,
                     the variable is treated as integer.
    """
    if mode not in (LP, MILP):
        raise ValueError("mode must be 'LP' or 'MILP'")

    sense = pulp.LpMaximize if model["sense"] == "max" else pulp.LpMinimize
    prob = pulp.LpProblem(model.get("name", "model").replace(" ", "_"), sense)

    vars_ = {
        v["name"]: _make_var(
            prob, v["name"], v.get("lb", 0), v.get("ub"),
            "Integer" if (mode == MILP and v.get("integer", True)) else "Continuous",
        )
        for v in model["variables"]
    }

    prob += pulp.lpSum(c * vars_[n] for n, c in model["objective"].items())

    ops = {
        "<=": lambda lhs, rhs: lhs <= rhs,
        ">=": lambda lhs, rhs: lhs >= rhs,
        "=": lambda lhs, rhs: lhs == rhs,
    }
    cnames = []
    for i, con in enumerate(model["constraints"]):
        lhs = pulp.lpSum(c * vars_[n] for n, c in con["coeffs"].items())
        cname = f"c{i}"
        prob += ops[con["sense"]](lhs, con["rhs"]), cname
        cnames.append((cname, con.get("name", cname)))

    start = time.perf_counter()
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    elapsed = time.perf_counter() - start

    status = pulp.LpStatus[prob.status]
    result = {
        "mode": mode,
        "status": status,
        "objective": None,
        "values": {},
        "time_s": elapsed,
        "shadow_prices": {},  # only meaningful in LP mode
    }
    if status == "Optimal":
        result["objective"] = pulp.value(prob.objective)
        result["values"] = {n: v.value() for n, v in vars_.items()}
        if mode == LP:
            result["shadow_prices"] = {
                label: prob.constraints[c].pi for c, label in cnames
            }
    return result


def compare_models(model: dict) -> dict:
    """Solve in both modes and build a side-by-side comparison."""
    lp = solve_model(model, LP)
    milp = solve_model(model, MILP)

    comparison = {"lp": lp, "milp": milp, "rows": [], "gap": None, "gap_pct": None,
                  "lp_already_integer": None, "note": ""}

    if lp["status"] != "Optimal" or milp["status"] != "Optimal":
        comparison["note"] = (
            f"LP status: {lp['status']} | MILP status: {milp['status']}. "
            "Comparison is only available when both are Optimal."
        )
        return comparison

    int_names = [v["name"] for v in model["variables"] if v.get("integer", True)]
    for name in lp["values"]:
        a, b = lp["values"][name], milp["values"][name]
        comparison["rows"].append(
            {"Variable": name + ("" if name in int_names else " (continuous)"),
             "LP (continuous)": round(a, 4),
             "MILP (integer)": round(b, 4), "Difference": round(b - a, 4)}
        )

    gap = abs(lp["objective"] - milp["objective"])
    comparison["gap"] = gap
    denom = abs(lp["objective"]) if abs(lp["objective"]) > TOL else 1.0
    comparison["gap_pct"] = 100 * gap / denom
    int_names = [v["name"] for v in model["variables"] if v.get("integer", True)]
    comparison["lp_already_integer"] = all(
        abs(lp["values"][n] - round(lp["values"][n])) <= TOL for n in int_names
    )

    if comparison["lp_already_integer"]:
        comparison["note"] = (
            "The LP solution already satisfies the integer requirements, so "
            "requiring integrality costs nothing here."
        )
    else:
        better = "lower" if model["sense"] == "max" else "higher"
        comparison["note"] = (
            f"Requiring integer values makes the objective {better} by "
            f"{gap:.4f} ({comparison['gap_pct']:.2f}%). This is the price of "
            "indivisibility."
        )
    return comparison


# ----------------------------------------------------------------------------
# Streamlit section — call render_solver_section(model) after the model is built
# ----------------------------------------------------------------------------
def render_solver_section(model: dict):
    import pandas as pd
    import streamlit as st

    st.subheader("Solve")
    choice = st.radio(
        "Solution mode",
        ["Continuous (LP)", "Integer (MILP)", "Compare both"],
        horizontal=True,
        help="LP allows fractional values. MILP forces integer variables to whole numbers.",
    )

    if not st.button("Solve", type="primary"):
        return

    if choice == "Compare both":
        cmp = compare_models(model)
        lp, milp = cmp["lp"], cmp["milp"]
        if not cmp["rows"]:
            st.warning(cmp["note"])
            return

        c1, c2, c3 = st.columns(3)
        c1.metric("LP objective", f"{lp['objective']:.4f}")
        c2.metric("MILP objective", f"{milp['objective']:.4f}")
        c3.metric("Gap", f"{cmp['gap']:.4f}", f"{cmp['gap_pct']:.2f}%", delta_color="off")

        st.dataframe(pd.DataFrame(cmp["rows"]), use_container_width=True, hide_index=True)
        st.info(cmp["note"])
        st.caption(f"Solve time — LP: {lp['time_s']:.3f}s | MILP: {milp['time_s']:.3f}s")
    else:
        mode = LP if choice.startswith("Continuous") else MILP
        res = solve_model(model, mode)
        if res["status"] != "Optimal":
            st.warning(f"Solver status: {res['status']}")
            return
        st.metric("Objective value", f"{res['objective']:.4f}")
        st.dataframe(
            pd.DataFrame(
                [{"Variable": k, "Value": round(v, 4)} for k, v in res["values"].items()]
            ),
            use_container_width=True, hide_index=True,
        )
        if mode == LP and res["shadow_prices"]:
            st.markdown("**Shadow prices (LP only)**")
            st.dataframe(
                pd.DataFrame(
                    [{"Constraint": k, "Shadow price": round(v, 4)}
                     for k, v in res["shadow_prices"].items()]
                ),
                use_container_width=True, hide_index=True,
            )
