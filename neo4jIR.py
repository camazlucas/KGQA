def ir_query(session, entity, hops=1):
    query = f"""
    MATCH (a {{name: $entity}})-[*1..{hops}]->(b)
    RETURN DISTINCT b.name AS name
    """
    
    result = session.run(query, entity=entity)
    
    data = [record["name"] for record in result]
    
    result.consume()
    
    return data