# MathModel AI

> Working title. **MVP 1.0**

A platform that turns problems written in natural language into **Linear Programming (LP)** models and offers to solve them.

## Idea

The project grew out of a university course whose problems come from **Topic 3 — Applications of Linear Programming**. The user describes the problem in their own words, and the system returns a well-organized mathematical model, ready to be solved.

## What the system does

1. Receives a problem as text (natural language).
2. Interprets the statement using AI.
3. Presents the LP model:
   - decision variables
   - objective function
   - constraints
   - non-negativity
4. Offers the option to **solve** the model.
5. Shows the result with an explanation.

## Problem types covered

- Linear modeling
- Blending / mixing
- Production and inventory planning
- Make vs. buy
- Transportation
- Transshipment
- Assignment / allocation

## Project rules

- **The AI must not invent formulas.** It must explicitly present variables, coefficients, and constraints.
- When a statement is ambiguous, the AI must **flag the ambiguity and ask for clarification**.
- The system must **detect missing data** before building the model.
- The result must be **explained**, not just a set of numbers.

## MVP 1.0 scope

**Included**
- Text input of the problem
- LP model generation
- Missing-data detection
- Solving, with an explained result

**Left for later**
- Problem history
- Sign-up and subscription

## Stack

- **Python**
- **Streamlit** (interface and execution, Python only — chosen over running on localhost because of localhost limitations)
- **PuLP** with the **CBC** solver
- **AI** to interpret natural language

## Development plan (Google Antigravity)

| Step | Description |
|------|-------------|
| 1 | Project structure |
| 2 | Math engine |
| 3 | Tests with well-known problems |
| 4 | AI interpretation |
| 5 | Interface |
| 6 | Integration and real cases |

## How to run

```bash
pip install streamlit pulp
streamlit run app.py
```

> Adjust the file name (`app.py`) and dependencies to match the final project structure.

## Status

Under development — MVP 1.0.
