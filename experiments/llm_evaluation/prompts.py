TRIPLES_PROMPT = """Use the following structured knowledge graph triples as context to answer the question.

Knowledge Graph triples:
{kg_triples}

Question:
{question}

Answer the question using only the information provided in the knowledge graph triples.

Return ONLY the answer.
Do not explain.
Do not describe the reasoning.
Do not repeat the question.
Do not add punctuation or additional text.
"""


def build_triples_prompt(question, kg_triples):
    return TRIPLES_PROMPT.format(
        kg_triples=kg_triples,
        question=question
    )