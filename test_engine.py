"""Testes do motor com problemas clássicos (sem IA)."""
import pytest
import engine

def T(**kw): return [{"var": k, "coef": v} for k, v in kw.items()]

def V(*n, tipo="continua"): return [{"nome": x, "tipo": tipo} for x in n]

def test_fabrica():
    m = {"variaveis": V("x_A","x_B"), "objetivo": {"sentido":"max","termos":T(x_A=30,x_B=50)},
         "restricoes":[{"nome":"horas","termos":T(x_A=0.5,x_B=0.2),"sentido":"<=","rhs":100}]}
    r = engine.resolver(m)
    assert r["status"]=="Optimal" and r["z"]==pytest.approx(25000)
    assert r["valores"]["x_B"]==pytest.approx(500) and r["valores"]["x_A"]==pytest.approx(0)
    assert r["restricoes"][0]["preco_sombra"]==pytest.approx(250)

def test_fracao_em_texto():
    m = {"variaveis": V("x"), "objetivo": {"sentido":"max","termos":[{"var":"x","coef":"2"}]},
         "restricoes":[{"nome":"a","termos":[{"var":"x","coef":"1/2"}],"sentido":"<=","rhs":"10"}]}
    assert engine.resolver(m)["z"]==pytest.approx(40)

def test_mistura():
    m = {"variaveis": V("milho","soja"), "objetivo":{"sentido":"min","termos":T(milho=2,soja=5)},
         "restricoes":[{"nome":"lote","termos":T(milho=1,soja=1),"sentido":"==","rhs":100},
                       {"nome":"prot","termos":T(milho=0.08,soja=0.40),"sentido":">=","rhs":20}]}
    r = engine.resolver(m)
    assert r["status"]=="Optimal"
    assert r["valores"]["milho"]+r["valores"]["soja"]==pytest.approx(100)

def test_transporte():
    cost={"11":4,"12":6,"13":8,"21":5,"22":3,"23":7}
    m={"variaveis":V(*[f"x_{k}" for k in cost]),"objetivo":{"sentido":"min","termos":[{"var":f"x_{k}","coef":c} for k,c in cost.items()]},
       "restricoes":[{"nome":"F1","termos":T(x_11=1,x_12=1,x_13=1),"sentido":"<=","rhs":100},
                     {"nome":"F2","termos":T(x_21=1,x_22=1,x_23=1),"sentido":"<=","rhs":150},
                     {"nome":"L1","termos":T(x_11=1,x_21=1),"sentido":">=","rhs":80},
                     {"nome":"L2","termos":T(x_12=1,x_22=1),"sentido":">=","rhs":90},
                     {"nome":"L3","termos":T(x_13=1,x_23=1),"sentido":">=","rhs":80}]}
    r=engine.resolver(m); assert r["status"]=="Optimal"
    assert sum(r["valores"].values())==pytest.approx(250)

def test_alocacao_binaria():
    c={"11":9,"12":2,"21":3,"22":7}
    m={"variaveis":V(*[f"y_{k}" for k in c],tipo="binaria"),"objetivo":{"sentido":"min","termos":[{"var":f"y_{k}","coef":v} for k,v in c.items()]},
       "restricoes":[{"nome":"r1","termos":T(y_11=1,y_12=1),"sentido":"==","rhs":1},{"nome":"r2","termos":T(y_21=1,y_22=1),"sentido":"==","rhs":1},
                     {"nome":"t1","termos":T(y_11=1,y_21=1),"sentido":"==","rhs":1},{"nome":"t2","termos":T(y_12=1,y_22=1),"sentido":"==","rhs":1}]}
    r=engine.resolver(m); assert r["z"]==pytest.approx(5)

def test_inviavel():
    m={"variaveis":V("x"),"objetivo":{"sentido":"max","termos":T(x=1)},
       "restricoes":[{"nome":"a","termos":T(x=1),"sentido":"<=","rhs":10},{"nome":"b","termos":T(x=1),"sentido":">=","rhs":20}]}
    assert engine.resolver(m)["status"]=="Infeasible"

def test_ilimitado():
    m={"variaveis":V("x"),"objetivo":{"sentido":"max","termos":T(x=1)},
       "restricoes":[{"nome":"a","termos":T(x=1),"sentido":">=","rhs":1}]}
    assert engine.resolver(m)["status"]=="Unbounded"

def test_variavel_nao_declarada():
    m={"variaveis":V("x"),"objetivo":{"sentido":"max","termos":T(y=1)},"restricoes":[{"nome":"a","termos":T(x=1),"sentido":"<=","rhs":1}]}
    with pytest.raises(engine.ModeloInvalido): engine.validar(m)

def test_rhs_ausente():
    m={"variaveis":V("x"),"objetivo":{"sentido":"max","termos":T(x=1)},"restricoes":[{"nome":"a","termos":T(x=1),"sentido":"<=","rhs":None}]}
    with pytest.raises(engine.ModeloInvalido): engine.validar(m)
