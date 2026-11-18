"""Interpretação do problema em linguagem natural via Groq (resposta em JSON puro)."""
import json
import os

from groq import Groq

MODELO_PADRAO = os.environ.get("GROQ_MODEL", "")  # vazio = você escolhe na barra lateral do app

SYSTEM = """You are an Operations Research expert. Convert the user's problem into a Linear Programming model.

GOLDEN RULE: NEVER invent numbers. Use only values ​​that are in the text or that are direct unit conversions (e.g., 2 units per hour => 0.5 hours per unit). If a piece of data needed to complete the model is NOT in the text,

DO NOT fill it in: describe it in "missing_data". If something can have more than one interpretation, describe it in "ambiguities".

Classify the problem into ONE category: "General linear modeling", "Mixing", "Production and inventory planning",
"Make x Buy", "Transportation", "Transshipment", "Allocation".

Modeling rules:

- Variables with short names (x_A, x_B ...). In problems involving indices (transport, transshipment, allocation, multiple periods), EXPAND all variables and name them with the labels from the problem statement in the format x_<Origin><Destination>, e.g.: x_E1P1 = team E1 in project P1; x_F2L3 = factory F2 for store L3. If the problem statement does not label the entities, use x_11, x_12 ...
- In terms of the objective function and constraints, follow the natural order of the problem statement: group by destination (P1, P2, P3...) and, within each group, by the order of origins (E1, E2, E3...).
- Variable type: "continuous", "integer" (when the problem statement requires integer units) or "binary" (1-to-1 allocation, yes/no).
- All coefficients are NUMBERS (use decimal point). A coefficient that does not appear in an expression simply does not enter into "terms".
- All variables are >= 0 by default (do not declare this as a constraint).

- Transportation: supply constraints (<=, or == if supply = demand) and demand constraints (>= or ==). Transshipment: flow conservation at intermediate nodes (in - out == 0). Allocation: each resource and each task sums to exactly 1.
- Mixing: minimum/maximum content and total batch constraints. Inventory: balance E_t = E_(t-1) + production_t - demand_t.

- Each constraint takes the "rhs" field with only the number on the right side (moves all variables to the left side).

Please respond ONLY with valid JSON, without markdown, in this exact format:

{
"category": "...",

"summary": "a sentence stating what the problem asks for",
"variables": [{"name": "x_A", "description": "...", "unit": "...", "type": "continues"}],
"objective": {"meaning": "max" or "min", "description": "...", "terms": [{"var": "x_A", "coef": 30}]},
"restrictions": [{"name": "...", "terms": [{"var": "x_A", "coef": 0.5}], "meaning": "<=" or ">=" or "==", "rhs": 100, "explanation": "..."}],
"deductions": ["conversions or deductions made, e.g.: 2 units/hour => 0.5 hours/unit"],
"missing_data": ["what is missing from the text to complete the model"],
"ambiguities": ["ambiguous points in the statement"]

"language":["translate the answers into English"]

} If vital data is missing, still return what can be assembled, but list everything in "missing_data".""""."""


def interpretar(texto: str, api_key: str, modelo: str = MODELO_PADRAO) -> dict:
    if not modelo:
        raise ValueError("Nenhum modelo escolhido. Escolha um na barra lateral (Modelos ativos).")
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=modelo,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": texto}],
        timeout=60,
    )
    bruto = resp.choices[0].message.content
    try:
        return json.loads(bruto)
    except json.JSONDecodeError:
        raise ValueError("A IA não devolveu um JSON válido. Tente interpretar de novo.")


def listar_modelos(api_key: str) -> list[str]:
    """Pergunta à própria Groq quais modelos de texto estão ativos agora (evita nomes aposentados)."""
    ids = [m.id for m in Groq(api_key=api_key).models.list().data if getattr(m, "active", True)]
    ruins = ("whisper", "tts", "guard", "playai", "orpheus", "compound", "embed")
    return sorted(i for i in ids if not any(r in i.lower() for r in ruins))
