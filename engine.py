"""Motor do MathModel AI: valida o modelo extraído pela IA, monta no PuLP e resolve com CBC."""
import math
import re
from fractions import Fraction

import pulp

TIPOS = {"continua": pulp.LpContinuous, "inteira": pulp.LpInteger, "binaria": pulp.LpBinary}
SENTIDOS = ("<=", ">=", "==")


class ModeloInvalido(Exception):
    pass


def _num(valor, onde):
    """Aceita número ou texto como '1/2' ou '0,5'. Rejeita qualquer outra coisa."""
    if isinstance(valor, bool):
        raise ModeloInvalido(f"Valor inválido em {onde}: {valor!r}")
    if isinstance(valor, (int, float)):
        x = float(valor)
    elif isinstance(valor, str):
        try:
            x = float(Fraction(valor.strip().replace(",", ".")))
        except (ValueError, ZeroDivisionError):
            raise ModeloInvalido(f"Valor não numérico em {onde}: {valor!r}")
    else:
        raise ModeloInvalido(f"Valor ausente ou inválido em {onde}: {valor!r}")
    if not math.isfinite(x):
        raise ModeloInvalido(f"Valor não finito em {onde}")
    return x


def validar(modelo: dict) -> dict:
    """Valida a estrutura e normaliza os números. Levanta ModeloInvalido se algo estiver errado."""
    variaveis = modelo.get("variaveis") or []
    if not variaveis:
        raise ModeloInvalido("Nenhuma variável de decisão foi identificada.")
    nomes = set()
    for v in variaveis:
        n = v.get("nome")
        if not n or n in nomes:
            raise ModeloInvalido(f"Nome de variável vazio ou repetido: {n!r}")
        nomes.add(n)
        v["tipo"] = v.get("tipo", "continua")
        if v["tipo"] not in TIPOS:
            raise ModeloInvalido(f"Tipo de variável inválido: {v['tipo']!r}")

    def termos(lista, onde):
        if not lista:
            raise ModeloInvalido(f"{onde} está sem termos.")
        saida = []
        for t in lista:
            if t.get("var") not in nomes:
                raise ModeloInvalido(f"{onde} usa a variável não declarada {t.get('var')!r}.")
            saida.append({"var": t["var"], "coef": _num(t.get("coef"), onde)})
        return saida

    obj = modelo.get("objetivo") or {}
    if obj.get("sentido") not in ("max", "min"):
        raise ModeloInvalido("O objetivo precisa ser 'max' ou 'min'.")
    obj["termos"] = termos(obj.get("termos"), "a função objetivo")

    restricoes = modelo.get("restricoes") or []
    if not restricoes:
        raise ModeloInvalido("Nenhuma restrição foi identificada.")
    for i, r in enumerate(restricoes, 1):
        onde = f"a restrição {i} ({r.get('nome', '?')})"
        if r.get("sentido") not in SENTIDOS:
            raise ModeloInvalido(f"Sentido inválido em {onde}: {r.get('sentido')!r}")
        r["termos"] = termos(r.get("termos"), onde)
        r["rhs"] = _num(r.get("rhs"), onde)
    return modelo


def _safe(nome):
    return re.sub(r"\W", "_", nome)


def compilar(modelo: dict):
    """Monta o problema no PuLP. Retorna (problema, variáveis, restrições)."""
    sentido = pulp.LpMaximize if modelo["objetivo"]["sentido"] == "max" else pulp.LpMinimize
    prob = pulp.LpProblem("MathModelAI", sentido)
    vars_ = {}
    for v in modelo["variaveis"]:
        kw = {"cat": TIPOS[v["tipo"]]}
        if v["tipo"] != "binaria":
            kw["lowBound"] = 0
        vars_[v["nome"]] = pulp.LpVariable(_safe(v["nome"]), **kw)

    def expr(termos):
        return pulp.lpSum(t["coef"] * vars_[t["var"]] for t in termos)

    prob += expr(modelo["objetivo"]["termos"]), "Z"
    restr = []
    for i, r in enumerate(modelo["restricoes"], 1):
        lhs, rhs = expr(r["termos"]), r["rhs"]
        c = lhs <= rhs if r["sentido"] == "<=" else lhs >= rhs if r["sentido"] == ">=" else lhs == rhs
        nome = f"R{i}_{_safe(r.get('nome', ''))}"[:60]
        prob += c, nome
        restr.append((r, nome))
    return prob, vars_, restr


def gerar_codigo(modelo: dict) -> str:
    """Gera o script PuLP reproduzível do modelo."""
    o = modelo["objetivo"]
    L = ["import pulp", "", f"prob = pulp.LpProblem('Modelo', pulp.Lp{'Maximize' if o['sentido'] == 'max' else 'Minimize'})", ""]
    for v in modelo["variaveis"]:
        n = _safe(v["nome"])
        if v["tipo"] == "binaria":
            L.append(f"{n} = pulp.LpVariable('{n}', cat='Binary')")
        else:
            cat = "Integer" if v["tipo"] == "inteira" else "Continuous"
            L.append(f"{n} = pulp.LpVariable('{n}', lowBound=0, cat='{cat}')")

    def e(termos):
        return " + ".join(f"{t['coef']:g}*{_safe(t['var'])}" for t in termos)

    L += ["", f"prob += {e(o['termos'])}, 'Z'"]
    for i, r in enumerate(modelo["restricoes"], 1):
        L.append(f"prob += {e(r['termos'])} {r['sentido']} {r['rhs']:g}, 'R{i}'")
    L += ["", "prob.solve(pulp.PULP_CBC_CMD(msg=False))",
          "print(pulp.LpStatus[prob.status], pulp.value(prob.objective))",
          "for v in prob.variables():", "    print(v.name, v.varValue)"]
    return "\n".join(L)


def resolver(modelo: dict) -> dict:
    """Resolve com CBC e devolve resultado + interpretação."""
    modelo = validar(modelo)
    prob, vars_, restr = compilar(modelo)
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]
    res = {"status": status, "z": None, "valores": {}, "restricoes": [], "codigo": gerar_codigo(modelo)}
    if status != "Optimal":
        res["interpretacao"] = {
            "Infeasible": "O problema é **inviável**: nenhuma combinação de valores atende todas as restrições ao mesmo tempo. Revise os dados ou as restrições.",
            "Unbounded": "O problema é **ilimitado**: o objetivo pode melhorar sem parar. Provavelmente falta uma restrição de limite.",
        }.get(status, f"O solver não encontrou solução (status: {status}).")
        return res

    res["z"] = pulp.value(prob.objective)
    res["valores"] = {n: (v.varValue or 0.0) for n, v in vars_.items()}
    so_lp = all(v["tipo"] == "continua" for v in modelo["variaveis"])
    for r, nome in restr:
        c = prob.constraints[nome]
        lhs = sum(t["coef"] * res["valores"][t["var"]] for t in r["termos"])
        folga = {"<=": r["rhs"] - lhs, ">=": lhs - r["rhs"], "==": 0.0}[r["sentido"]]
        pi = c.pi if so_lp and c.pi is not None else None
        res["restricoes"].append({"nome": r.get("nome", nome), "uso": lhs, "rhs": r["rhs"],
                                  "sentido": r["sentido"], "folga": abs(folga) if abs(folga) > 1e-9 else 0.0,
                                  "preco_sombra": None if pi is None else (pi if abs(pi) > 1e-9 else 0.0)})
    res["interpretacao"] = interpretar(modelo, res, so_lp)
    return res


def interpretar(modelo, res, so_lp):
    """Interpretação gerencial montada só a partir dos números (sem IA, sem invenção)."""
    obj = modelo["objetivo"]
    meta = "lucro máximo" if obj["sentido"] == "max" else "custo mínimo"
    linhas = [f"O valor ótimo é **Z = {res['z']:.4g}** ({meta})."]
    pos = {n: v for n, v in res["valores"].items() if v > 1e-9}
    zero = [n for n, v in res["valores"].items() if v <= 1e-9]
    if pos:
        linhas.append("Plano ótimo: " + ", ".join(f"{n} = {v:.4g}" for n, v in pos.items()) + ".")
    if zero:
        linhas.append("Ficam em zero: " + ", ".join(zero) + ".")
    for r in res["restricoes"]:
        if r["sentido"] != "==" and r["folga"] == 0:
            msg = f"A restrição **{r['nome']}** está no limite (folga 0): é ela que limita a solução."
            if r["preco_sombra"]:
                msg += f" Preço-sombra {r['preco_sombra']:.4g}: uma unidade a mais no lado direito altera Z em {r['preco_sombra']:.4g}."
            linhas.append(msg)
        elif r["sentido"] != "==":
            linhas.append(f"A restrição **{r['nome']}** tem folga de {r['folga']:.4g}: sobra recurso.")
    if not so_lp:
        linhas.append("Há variáveis inteiras/binárias, então preços-sombra não se aplicam.")
    return "\n\n".join(linhas)
