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
# Adaptador: converte o modelo do app.py (chaves em português) para o formato acima
# ----------------------------------------------------------------------------
def converter_modelo(modelo: dict, forcar_inteiras: bool = True) -> dict:
    """Converte o dict do app (variaveis/objetivo/restricoes) para o formato do solver.

    forcar_inteiras=True  -> no modo MILP, TODAS as variáveis viram inteiras.
    forcar_inteiras=False -> só as variáveis com tipo 'inteira' ou 'binaria' viram inteiras.
    """
    variaveis = []
    for v in modelo["variaveis"]:
        tipo = str(v.get("tipo", "")).lower()
        binaria = tipo.startswith("bin")
        inteira = binaria or tipo.startswith("int")
        variaveis.append({
            "name": v["nome"],
            "lb": 0,
            "ub": 1 if binaria else None,
            "integer": True if forcar_inteiras else inteira,
        })

    obj = modelo["objetivo"]
    sentido = "max" if str(obj["sentido"]).lower().startswith("max") else "min"

    constraints = []
    for i, r in enumerate(modelo["restricoes"], 1):
        sent = r["sentido"]
        constraints.append({
            "name": f"({i}) {r.get('nome', '')}".strip(),
            "coeffs": _somar_termos(r["termos"]),
            "sense": "=" if sent in ("==", "=") else sent,
            "rhs": r["rhs"],
        })

    return {
        "name": "MathModel",
        "sense": sentido,
        "variables": variaveis,
        "objective": _somar_termos(obj["termos"]),
        "constraints": constraints,
    }


def _somar_termos(termos):
    """[{'coef': 2, 'var': 'x'}, ...] -> {'x': 2} (soma coeficientes repetidos)."""
    d = {}
    for t in termos:
        d[t["var"]] = d.get(t["var"], 0) + t["coef"]
    return d


# ----------------------------------------------------------------------------
# Seção Streamlit — chame render_solver_section(modelo) depois do bloco "Resolver"
# ----------------------------------------------------------------------------
def render_solver_section(modelo: dict):
    import json
    import pandas as pd
    import streamlit as st

    st.markdown("### 7. Comparar: contínuo (LP) x inteiro (MILP)")

    c1, c2 = st.columns([2, 3])
    escolha = c1.radio(
        "Modo de solução",
        ["Contínuo (LP)", "Inteiro (MILP)", "Comparar os dois"],
        key="cmp_modo",
    )
    forcar = c2.checkbox(
        "No modo inteiro, forçar todas as variáveis a serem inteiras",
        value=True,
        key="cmp_forcar",
        help="Desmarque para respeitar só o tipo (contínua/inteira/binária) que a IA definiu.",
    )

    assinatura = json.dumps([modelo, escolha, forcar], sort_keys=True, default=str)

    if st.button("Resolver neste modo", key="cmp_botao"):
        m = converter_modelo(modelo, forcar_inteiras=forcar)
        if escolha == "Comparar os dois":
            dados = compare_models(m)
        else:
            dados = solve_model(m, LP if escolha.startswith("Cont") else MILP)
        st.session_state["cmp_resultado"] = {"sig": assinatura, "dados": dados}

    guardado = st.session_state.get("cmp_resultado")
    if not guardado or guardado["sig"] != assinatura:
        return
    dados = guardado["dados"]

    if escolha == "Comparar os dois":
        lp, milp = dados["lp"], dados["milp"]
        if not dados["rows"]:
            st.warning(dados["note"])
            return
        a, b, c = st.columns(3)
        a.metric("Z no LP (contínuo)", f"{lp['objective']:.6g}")
        b.metric("Z no MILP (inteiro)", f"{milp['objective']:.6g}")
        c.metric("Diferença (gap)", f"{dados['gap']:.6g}", f"{dados['gap_pct']:.2f}%", delta_color="off")
        st.dataframe(
            pd.DataFrame(dados["rows"]).rename(columns={
                "Variable": "Variável", "LP (continuous)": "LP (contínuo)",
                "MILP (integer)": "MILP (inteiro)", "Difference": "Diferença"}),
            hide_index=True, use_container_width=True,
        )
        if dados["lp_already_integer"]:
            st.info("A solução do LP já é inteira: exigir números inteiros não custa nada neste problema.")
        else:
            mais = "menor" if dados["lp"]["objective"] >= dados["milp"]["objective"] else "maior"
            st.info(f"Exigir números inteiros deixa o valor de Z {mais} em {dados['gap']:.6g} "
                    f"({dados['gap_pct']:.2f}%). É o custo de não poder fracionar.")
        st.caption(f"Tempo — LP: {lp['time_s']:.3f}s | MILP: {milp['time_s']:.3f}s")
    else:
        if dados["status"] != "Optimal":
            st.error(f"Status: {dados['status']}")
            return
        st.metric("Valor ótimo de Z", f"{dados['objective']:.6g}")
        st.dataframe(
            pd.DataFrame([{"Variável": k, "Valor": round(v, 6)} for k, v in dados["values"].items()]),
            hide_index=True, use_container_width=True,
        )
        if dados["mode"] == LP and dados["shadow_prices"]:
            st.write("**Preços-sombra (só no modo contínuo)**")
            st.dataframe(
                pd.DataFrame([{"Restrição": k, "Preço-sombra": round(v, 6)}
                              for k, v in dados["shadow_prices"].items()]),
                hide_index=True, use_container_width=True,
            )
