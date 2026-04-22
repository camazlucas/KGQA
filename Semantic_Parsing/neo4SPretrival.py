from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "senha123"))

def insert_triplet(tx, h, r, t):
    query = f"""
    MERGE (a:Entity {{name: $h}})
    MERGE (b:Entity {{name: $t}})
    MERGE (a)-[:{r}]->(b)
    """
    tx.run(query, h=h, t=t)

def query_graph(entity, relation):
    query = f"""
    MATCH (a)-[:{relation}]->(b)
    WHERE a.name = $entity
    RETURN b.name
    """
    
    with driver.session() as session:
        result = session.run(query, entity=entity)
        return [record["b.name"] for record in result]
    

def parse_question(q):
    q = q.lower()
    
    if "direct" in q:
        return "directed_by"
    
    if "actor" in q or "star" in q:
        return "starred_actors"
    
    if "writer" in q:
        return "written_by"
    
    if "genre" in q:
        return "has_genre"
    
    if "language" in q:
        return "in_language"
    
    if "release" in q or "year" in q:
        return "release_year"
    
    return None


def sp_query(session, entity, relation):
    query = f"""
    MATCH (a {{name: $entity}})-[:{relation}]->(b)
    RETURN b.name AS name
    """
    
    result = session.run(query, entity=entity)
    
    data = [record["name"] for record in result]
    
    result.consume()  # evita bug de buffer
    
    return data