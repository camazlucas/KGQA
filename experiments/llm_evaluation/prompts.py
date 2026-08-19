TRIPLES_PROMPT = """Use the following structured knowledge graph triples as context to answer the question.

Knowledge Graph triples:
{kg_triples}

Question:
{question}

Answer only with the answer to the question. Do not provide explanations.
"""


def build_triples_prompt(question, kg_triples):
    return TRIPLES_PROMPT.format(
        kg_triples=kg_triples,
        question=question
    )