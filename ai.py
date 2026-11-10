"""Interpretação do problema em linguagem natural via Groq (resposta em JSON puro)."""
import json
import os

from groq import Groq

MODELO_PADRAO = os.environ.get("GROQ_MODEL", "")  # vazio = você escolhe na barra lateral do app

SYSTEM = """Você é um especialista em Pesquisa Operacional. Converta o problema do usuário em um modelo de Programação Linear.

REGRA DE OURO: NUNCA invente números. Use apenas valores que estão no texto ou que sejam conversão direta de unidade
(ex.: 2 unidades por hora => 0.5 hora por unidade). Se um dado necessário para fechar o modelo NÃO está no texto,
NÃO o preencha: descreva-o em "dados_ausentes". Se algo puder ter mais de uma leitura, descreva em "ambiguidades".

Classifique o problema em UMA categoria: "Modelagem linear geral", "Mistura", "Planejamento de produção e estoque",
"Fazer x Comprar", "Transporte", "Transbordo", "Alocação".

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
