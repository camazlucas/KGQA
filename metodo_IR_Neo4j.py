from Processamento.avaliacao import Evaluator
from Processamento.extrair import extract_entity
from Dataset.dataset import load_qa
from neo4j import GraphDatabase
from neo4jIR import ir_query
import armazenar_database as ad

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "senha123")
)

# 🔥 verificação automática
if not ad.grafo_esta_carregado(driver):
    print("Grafo não encontrado. Carregando...")
    ad.carregar_grafo_neo4j("Benchmark/kb.txt")
else:
    print("Grafo carregado.")


# -----------------------
# 2. LOOP DE PERGUNTAS
# -----------------------

qa_data = load_qa("Benchmark/1-hop/vanilla/qa_test.txt")
evaluator = Evaluator()

k = 5
hops = 1  # depois você testa 2 e 3

with driver.session() as session:

    for q, a in qa_data:
        
        # 1 - Extrair entidade
        entity = extract_entity(q)
        
        # 2 - IR com Neo4j
        candidates = ir_query(session, entity, hops=hops)
        
        # 3 - top-k
        prediction = candidates[:k] if candidates else []
        
        # 4 - avaliação
        evaluator.update(prediction, a)

    evaluator.report()

driver.close()