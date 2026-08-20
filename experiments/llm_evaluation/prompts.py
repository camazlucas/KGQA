TRIPLES_PROMPT = """Use the following structured knowledge graph triples as context to answer the question.

Knowledge Graph triples:

{kg_triples}

Each triple is represented as:

Head | Relation | Tail

The relation describes the relationship from Head to Tail.

Use the triples to determine which entity, Head or Tail, answers the question.

Question:

{question}

Return only the answer.
Do not explain your reasoning.
"""


def build_triples_prompt(question, kg_triples):
    return TRIPLES_PROMPT.format(
        kg_triples=kg_triples,
        question=question
    )