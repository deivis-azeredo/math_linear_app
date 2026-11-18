# MathModel AI

> Nome provisório. **MVP 1.0**

Plataforma que transforma problemas escritos em linguagem natural em modelos de **Programação Linear (PL)** e oferece resolvê-los.

## Ideia

O projeto nasce de uma matéria da faculdade cujos problemas são do **Tema 3 — Aplicações da Programação Linear**. O usuário escreve o problema com suas próprias palavras e o sistema devolve o modelo matemático organizado, pronto para ser resolvido.

## O que o sistema faz

1. Recebe um problema em texto (linguagem natural).
2. Interpreta o enunciado com IA.
3. Apresenta o modelo de PL:
   - variáveis de decisão
   - função objetivo
   - restrições
   - não negatividade
4. Oferece a opção de **resolver** o modelo.
5. Mostra o resultado de forma explicada.

## Tipos de problema abordados

- Modelagem linear
- Mistura
- Planejamento de produção e estoque
- Fazer × comprar
- Transporte
- Transbordo
- Alocação

## Regras de projeto

- **A IA não pode inventar fórmula.** Ela deve apresentar variáveis, coeficientes e restrições de forma explícita.
- Quando o enunciado for ambíguo, a IA deve **sinalizar a ambiguidade e pedir esclarecimentos**.
- O sistema deve **detectar dados ausentes** antes de montar o modelo.
- O resultado deve vir **explicado**, não apenas como números.

## Escopo do MVP 1.0

**Inclui**
- Entrada de problema em texto
- Geração do modelo de PL
- Detecção de dados ausentes
- Resolução e resultado explicado

**Fica para depois**
- Histórico de problemas
- Cadastro e assinatura

## Stack

- **Python**
- **Streamlit** (interface e execução, somente Python — escolhido no lugar de rodar em localhost, por causa das limitações do localhost)
- **PuLP** com solver **CBC**
- **IA** para interpretar a linguagem natural

## Plano de desenvolvimento (Google Antigravity)

| Etapa | Descrição |
|-------|-----------|
| 1 | Estrutura do projeto |
| 2 | Motor matemático |
| 3 | Testes com problemas conhecidos |
| 4 | Interpretação por IA |
| 5 | Interface |
| 6 | Integração e casos reais |

## Como rodar

```bash
pip install streamlit pulp
streamlit run app.py
```

> Ajuste o nome do arquivo (`app.py`) e as dependências conforme a estrutura final do projeto.

## Status

Em desenvolvimento — MVP 1.0.
