def load_qa(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            q, a = line.strip().split("\t")
            data.append((q, a))
    return data