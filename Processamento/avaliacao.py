class Evaluator:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, predictions, gold):
        # tratar múltiplas respostas
        gold_answers = gold.split("|")
        
        # verifica se alguma previsão está correta
        if any(p in gold_answers for p in predictions):
            self.correct += 1
        
        self.total += 1

    def report(self):
        print(f"Hits@k: {self.correct / self.total:.4f}")