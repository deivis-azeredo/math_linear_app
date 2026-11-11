import os
import re

import pandas as pd
import streamlit as st

import ai
import engine
#==============================================COMEÇO
import base64
import streamlit as st

st.set_page_config(page_title="MathModel AI", page_icon="📐", layout="wide")

def aplicar_estilo(css_path: str, imagem_path: str):
    with open(imagem_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    with open(css_path, encoding="utf-8") as f:
        css = f.read()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                linear-gradient(rgba(15,23,42,0.55), rgba(15,23,42,0.55)),
                url("data:image/png;base64,{img_b64}") center / cover fixed no-repeat;
        }}
        {css}
        </style>
        """,
        unsafe_allow_html=True,
    )

aplicar_estilo("style.css", "math.png")
#=====================================================fim

st.set_page_config(page_title="MathModel AI", page_icon="📐", layout="wide")

EXEMPLOS = {
    "Fábrica A e B": "Uma fábrica produz dois produtos. O produto A gera R$ 30 de lucro e o B gera R$ 50. "
                     "A fábrica dispõe de 100 horas de máquina. A produz 2 unidades por hora e B produz 5 unidades por hora. "
                     "O objetivo é maximizar o lucro.",
    "Mistura (ração)": "Uma fazenda quer misturar milho e soja para fazer 100 kg de ração ao menor custo. "
                       "O milho custa R$ 2 por kg e a soja R$ 5 por kg. O milho tem 8% de proteína e a soja 40%. "
                       "A ração deve ter no mínimo 20% de proteína.",
    "Transporte": "Duas fábricas F1 e F2 produzem 100 e 150 unidades. Três lojas L1, L2 e L3 demandam 80, 90 e 80 unidades. "
                  "Custos de frete por unidade: F1->L1 4, F1->L2 6, F1->L3 8, F2->L1 5, F2->L2 3, F2->L3 7. "
                  "Minimize o custo total de transporte.",
    "Com dado faltando": "Uma padaria faz pães e bolos. O pão dá R$ 0,50 de lucro por unidade e o bolo dá R$ 8. "
                         "A padaria tem 40 horas de forno por semana. Maximize o lucro.",
}


def latex_nome(n):
    """x_E1P1 -> x_{E_1 P_1}; x_12 -> x_{12}; x_A -> x_{A}."""
    if "_" not in n:
        return n
    a, b = n.split("_", 1)
    b = b.replace("_", ",")
    if re.search(r"[A-Za-z]\d", b):
        b = re.sub(r"([A-Za-z]+)(\d+)", r"\1_{\2}\\,", b).rstrip("\\,")
    return f"{a}_{{{b}}}"


def latex_expr(termos):
    partes = []
    for t in termos:
        c = t["coef"]
        v = latex_nome(t["var"])
        mod = abs(c)
        corpo = v if mod == 1 else f"{mod:g}\\,{v}"
        if not partes:
            partes.append(("-" if c < 0 else "") + corpo)
        else:
            partes.append(("- " if c < 0 else "+ ") + corpo)
    return " ".join(partes) or "0"


# ---------------- barra lateral ----------------
with st.sidebar:
    st.header("Configuração")
    try:
        chave_padrao = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        chave_padrao = ""
    chave = st.text_input("Chave da API Groq", type="password", value=chave_padrao)
    try:
        modelo_padrao = ai.MODELO_PADRAO or st.secrets.get("GROQ_MODEL", "")
    except Exception:
        modelo_padrao = ai.MODELO_PADRAO
    modelo_ia = st.text_input("Modelo da IA (opcional se escolher na lista abaixo)", value=modelo_padrao)
    if chave and "lista_modelos" not in st.session_state:
        try:
            st.session_state.lista_modelos = ai.listar_modelos(chave)
        except Exception as e:
            st.session_state.lista_modelos = []
            st.warning(f"Não consegui listar os modelos: {e}")
    if st.session_state.get("lista_modelos"):
        modelo_ia = st.selectbox("Modelos ativos (escolha um)", st.session_state.lista_modelos,
                                 index=None, placeholder="Escolha um modelo") or modelo_ia
    st.caption("A IA só monta o modelo. A solução vem do PuLP + CBC, não da IA.")

st.title("📐 MathModel AI")
st.caption("Do problema à solução matemática: cole o enunciado e a plataforma monta o modelo de programação linear.")

# ---------------- entrada ----------------
st.session_state.setdefault("texto", "")
cols = st.columns(len(EXEMPLOS))
for col, (nome, txt) in zip(cols, EXEMPLOS.items()):
    if col.button(nome, width='stretch'):
        st.session_state.texto = txt
        st.session_state.pop("modelo", None)
        st.session_state.pop("resultado", None)

texto = st.text_area("Problema", key="texto", height=170,
                     placeholder="Cole aqui o enunciado do problema de programação linear...")

if st.button("Interpretar problema", type="primary"):
    if not chave:
        st.error("Informe a chave da API Groq na barra lateral.")
    elif not texto.strip():
        st.error("Escreva ou cole um problema.")
    else:
        with st.spinner("Interpretando o problema..."):
            try:
                st.session_state.modelo = ai.interpretar(texto, chave, modelo_ia)
                st.session_state.pop("resultado", None)
            except Exception as e:
                st.error(f"Falha ao interpretar: {e}")

modelo = st.session_state.get("modelo")
if not modelo:
    st.stop()

# ---------------- dados ausentes / ambiguidades ----------------
ausentes = [a for a in modelo.get("dados_ausentes", []) if a]
ambiguidades = [a for a in modelo.get("ambiguidades", []) if a]

if ausentes:
    st.warning("**Faltam dados para fechar o modelo.** Por isso o solver está bloqueado, e nenhum número foi inventado:\n\n"
               + "\n".join(f"- {a}" for a in ausentes))
    comp = st.text_input("Complete a informação que falta (ex.: 'cada pão usa 0,1 hora de forno e cada bolo 0,5')")
    if st.button("Reinterpretar com o complemento") and comp.strip():
        with st.spinner("Reinterpretando..."):
            try:
                st.session_state.modelo = ai.interpretar(texto + "\n\nInformação adicional: " + comp, chave, modelo_ia)
                st.session_state.pop("resultado", None)
                st.rerun()
            except Exception as e:
                st.error(f"Falha ao interpretar: {e}")
if ambiguidades:
    st.info("**Pontos ambíguos no enunciado:**\n\n" + "\n".join(f"- {a}" for a in ambiguidades))

# ---------------- validação ----------------
try:
    engine.validar(modelo)
    valido = True
except engine.ModeloInvalido as e:
    valido = False
    st.error(f"O modelo extraído está incompleto ou inconsistente: {e}")

# ---------------- modelo organizado ----------------
st.subheader(f"Tipo de problema: {modelo.get('categoria', '—')}")
if modelo.get("resumo"):
    st.write(modelo["resumo"])

if valido:
    st.markdown("### 1. Variáveis de decisão")
    st.dataframe(pd.DataFrame([{"Símbolo": v["nome"], "Significado": v.get("descricao", ""),
                                "Unidade": v.get("unidade", ""), "Tipo": v["tipo"].capitalize()}
                               for v in modelo["variaveis"]]), hide_index=True, width='stretch')

    obj = modelo["objetivo"]
    st.markdown("### 2. Função objetivo")
    st.latex(f"\\{obj['sentido']}\\; Z = {latex_expr(obj['termos'])}")
    if obj.get("descricao"):
        st.caption(obj["descricao"])

    st.markdown("### 3. Restrições")
    simb = {"<=": "\\le", ">=": "\\ge", "==": "="}
    for i, r in enumerate(modelo["restricoes"], 1):
        st.latex(f"({i})\\quad {latex_expr(r['termos'])} {simb[r['sentido']]} {r['rhs']:g}")
        st.caption(f"{r.get('nome', '')}: {r.get('explicacao', '')}")

    st.markdown("### 4. Condições de não negatividade")
    nomes = ", ".join(latex_nome(v["nome"]) for v in modelo["variaveis"] if v["tipo"] != "binaria")
    if nomes:
        st.latex(f"{nomes} \\ge 0")
    binarias = [latex_nome(v["nome"]) for v in modelo["variaveis"] if v["tipo"] == "binaria"]
    if binarias:
        st.latex(", ".join(binarias) + " \\in \\{0,1\\}")

    st.markdown("### 5. Modelo matemático organizado")
    st.latex(f"\\begin{{aligned}} \\{obj['sentido']}\\; Z &= {latex_expr(obj['termos'])} \\\\ "
             "\\text{sujeito a:} & \\\\ "
             + " \\\\ ".join(f"& {latex_expr(r['termos'])} {simb[r['sentido']]} {r['rhs']:g}" for r in modelo["restricoes"])
             + " \\end{aligned}")
    if modelo.get("deducoes"):
        with st.expander("Deduções e conversões feitas pela IA (confira!)"):
            for d in modelo["deducoes"]:
                st.write("- " + d)

    # ---------------- resolver ----------------
    st.markdown("### 6. Resolver")
    if ausentes:
        st.button("Resolver modelo", disabled=True, help="Complete os dados ausentes primeiro.")
    elif st.button("Resolver modelo com PuLP + CBC", type="primary"):
        st.session_state.resultado = engine.resolver(modelo)

    res = st.session_state.get("resultado")
    if res:
        ok = res["status"] == "Optimal"
        (st.success if ok else st.error)(f"Status: {res['status']}")
        if ok:
            st.metric("Valor ótimo de Z", f"{res['z']:.6g}")
            st.write("**Valores ótimos das variáveis**")
            st.dataframe(pd.DataFrame([{"Variável": k, "Valor": round(v, 6)} for k, v in res["valores"].items()]),
                         hide_index=True, width='stretch')
            st.write("**Folgas e preços-sombra**")
            st.dataframe(pd.DataFrame([{"Restrição": r["nome"], "Uso": round(r["uso"], 6), "Limite": r["rhs"],
                                        "Folga": round(r["folga"], 6),
                                        "Preço-sombra": "—" if r["preco_sombra"] is None else round(r["preco_sombra"], 6)}
                                       for r in res["restricoes"]]), hide_index=True, width='stretch')
        st.markdown("**Interpretação**")
        st.markdown(res["interpretacao"])
        with st.expander("Código Python (PuLP) deste modelo"):
            st.code(res["codigo"], language="python")
