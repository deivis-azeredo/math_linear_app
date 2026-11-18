
import json
import os

from groq import Groq

idioma = st.sidebar.selectbox("Language / Idioma", ["English", "Português"])

TEXTOS = {
    "English": {"titulo": "MathModel AI", "resolver": "Solve model"},
    "Português": {"titulo": "MathModel AI", "resolver": "Resolver modelo"},
}
t = TEXTOS[idioma]
st.title(t["titulo"])
st.button(t["resolver"])

MODELO_PADRAO = os.environ.get("GROQ_MODEL", "")  # vazio = você escolhe na barra lateral do app

system_prompt = f"""
You are an assistant that converts word problems into linear programming models.
Always answer in {idioma}, regardless of the language of the user's input.
Present: decision variables, objective function, constraints, non-negativity.
Never invent formulas; flag ambiguities and ask for clarification.
"""
Regras de modelagem:
- Variáveis com nomes curtos (x_A, x_B ...). Em problemas com índices (transporte, transbordo, alocação, vários períodos), EXPANDA todas as variáveis e nomeie-as com os rótulos do enunciado no formato x_<Origem><Destino>, ex.: x_E1P1 = equipe E1 no projeto P1; x_F2L3 = fábrica F2 para loja L3. Se o enunciado não rotula as entidades, use x_11, x_12 ...
- Nos termos da função objetivo e das restrições, siga a ordem natural do enunciado: agrupe por destino (P1, P2, P3...) e, dentro de cada grupo, pela ordem das origens (E1, E2, E3...).
- tipo da variável: "continua", "inteira" (quando o enunciado exige unidades inteiras) ou "binaria" (alocação 1-para-1, sim/não).
- Todos os coeficientes são NÚMEROS (use ponto decimal). Coeficiente que não aparece numa expressão simplesmente não entra em "termos".
- Todas as variáveis são >= 0 por padrão (não declare isso como restrição).
- Transporte: restrições de oferta (<=, ou == se oferta = demanda) e de demanda (>= ou ==). Transbordo: conservação de fluxo nos nós intermediários (entra - sai == 0). Alocação: cada recurso e cada tarefa soma exatamente 1.
- Mistura: restrições de teor mínimo/máximo e de total do lote. Estoque: balanço E_t = E_(t-1) + produção_t - demanda_t.
- Cada restrição leva o campo "rhs" apenas com o número do lado direito (move todas as variáveis para o lado esquerdo).

Responda SOMENTE com JSON válido, sem markdown, neste formato exato:
{
 "categoria": "...",
 "resumo": "uma frase dizendo o que o problema pede",
 "variaveis": [{"nome": "x_A", "descricao": "...", "unidade": "...", "tipo": "continua"}],
 "objetivo": {"sentido": "max" ou "min", "descricao": "...", "termos": [{"var": "x_A", "coef": 30}]},
 "restricoes": [{"nome": "...", "termos": [{"var": "x_A", "coef": 0.5}], "sentido": "<=" ou ">=" ou "==", "rhs": 100, "explicacao": "..."}],
 "deducoes": ["conversões ou deduções feitas, ex.: 2 un/h => 0.5 h/un"],
 "dados_ausentes": ["o que falta no texto para fechar o modelo"],
 "ambiguidades": ["pontos ambíguos do enunciado"]
 "linguagem":["traduza as respostas para o inglês"]
}
Se faltarem dados vitais, ainda assim devolva o que for possível montar, mas liste tudo em "dados_ausentes"."""


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
