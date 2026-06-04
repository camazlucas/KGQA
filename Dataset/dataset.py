import networkx as nx

def load_graph(path):
    G = nx.DiGraph()
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            h, r, t = line.strip().split("|")
            G.add_edge(h, t, relation=r)
    
    return G

def load_qa(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            q, a = line.strip().split("\t")
            data.append((q, a))
    return data