from neo4j import GraphDatabase


def carregar_grafo_neo4j(path, uri="bolt://localhost:7687", user="neo4j", password="senha123"):
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    with driver.session() as session:
        with open(path, encoding="utf-8") as f:
            
            for i, line in enumerate(f):
                parts = line.strip().split("|")
                
                if len(parts) != 3:
                    continue
                
                h, r, t = parts
                
                # sanitizar relação
                r = r.replace(" ", "_").replace("-", "_")
                
                query = f"""
                MERGE (a:Entity {{name: $h}})
                MERGE (b:Entity {{name: $t}})
                MERGE (a)-[:{r}]->(b)
                """
                
                # 🔥 consumir resultado (evita bug de buffer)
                list(session.run(query, h=h, t=t))
                
                if i % 1000 == 0:
                    print(f"{i} triplas inseridas...")

    driver.close()
    print("Grafo carregado com sucesso!")

def grafo_esta_carregado(driver):
    with driver.session() as session:
        nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        
        return nodes > 0 and rels > 0