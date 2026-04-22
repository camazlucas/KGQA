from Processamento.avaliacao import Evaluator
from Processamento.extrair import extract_entity
from Dataset.dataset import load_qa
import Semantic_Parsing.neo4SPretrival as neoSP
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "senha123")
)
LOAD_DB = True  # 🔥 controle aqui

# -----------------------
# 1. CARREGAR O GRAFO (uma vez só)
# -----------------------
if LOAD_DB:
    with driver.session() as session:
        with open("Benchmark/kb.txt", encoding="utf-8") as f:
            for line in f:
                h, r, t = line.strip().split("|")
                
                r = r.replace(" ", "_").replace("-", "_")
                
                query = f"""
                MERGE (a:Entity {{name: $h}})
                MERGE (b:Entity {{name: $t}})
                MERGE (a)-[:{r}]->(b)
                """
                
                session.run(query, h=h, t=t)

    print("Grafo carregado!")

# -----------------------
# 2. LOOP DE PERGUNTAS
# -----------------------

qa_data = load_qa("Benchmark/1-hop/vanilla/qa_test.txt")
evaluator = Evaluator()
k = 5

with driver.session() as session:
    
    for q, a in qa_data:
        
        entity = extract_entity(q)
        relation = neoSP.parse_question(q)
    
        if relation is None:
            evaluator.update([], a)
            continue

        # 🔥 agora sim usando Neo4j
        candidates = neoSP.sp_query(session, entity, relation)
    
        prediction = candidates[:k] if candidates else []
    
        evaluator.update(prediction, a)

    evaluator.report()
driver.close()

