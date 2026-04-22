from Classicos.bfs import bfs_answer
from Dataset.dataset import load_graph
from Processamento.avaliacao import Evaluator
from Processamento.extrair import extract_entity
from Dataset.dataset import load_qa

# carregar grafo
G = load_graph("Benchmark/kb.txt")

# carregar perguntas
qa_data = load_qa("Benchmark/1-hop/vanilla/qa_train.txt")

# inicializar avaliação
evaluator = Evaluator()
k = 5


# loop principal
for q, a in qa_data:
    
    #1 - Extrair entidade da pergunta
    entity = extract_entity(q)
    
    #2 - Calcula candidatos a resposta
    candidates = bfs_answer(G, entity)
    
    #3 - Armazena resposta final
    prediction = candidates[:k] if candidates else []
    
    #4 - Compara resposta encontrada com a resposta certa 
    evaluator.update(prediction, a)

# resultado final
evaluator.report()