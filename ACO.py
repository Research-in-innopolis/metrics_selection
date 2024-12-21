import torch
from tqdm import tqdm

from utils import sammon_error
from tqdm.auto import trange

class ACO:
    def __init__(self, n_ants, n_features, n_selected_features, alpha=1.0, beta=2.0, rho=0.5, q=1.0, max_iter=100,
                 device='cuda'):
        self.n_ants = n_ants
        self.n_features = n_features
        self.n_selected_features = n_selected_features
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.max_iter = max_iter
        self.device = device
        self.pheromone = torch.ones(n_features, device=device)
        self.best_solution = None
        self.best_score = float('inf')

    def run(self, data, batch_size=None, use_tqdm=True):
        data = data.to(self.device)
        iterator = range(self.max_iter)
        if use_tqdm:
            iterator = trange(self.max_iter, leave=False)

        for iteration in iterator:
            solutions = []
            scores = []
            for ant in range(self.n_ants):
                solution = self.construct_solution()
                score = self.evaluate_solution(solution, data, batch_size)
                solutions.append(solution)
                scores.append(score)
                if score < self.best_score:
                    self.best_score = score
                    self.best_solution = solution
            self.update_pheromone(solutions, scores)
        return self.best_solution, self.best_score

    def construct_solution(self):
        probabilities = (self.pheromone ** self.alpha) * ((1.0 / self.pheromone) ** self.beta)

        probabilities[torch.isnan(probabilities)] = 0.001
        probabilities[torch.isinf(probabilities)] = 0.001
        probabilities[probabilities < 0] = 0.001

        probabilities = probabilities / probabilities.sum()

        solution = torch.multinomial(probabilities, self.n_selected_features, replacement=False)
        return solution

    def evaluate_solution(self, solution, data, batch_size):
        data_selected = data[:, solution]
        error = sammon_error(data_selected, data, batched_input=batch_size is not None, batch_size=batch_size)
        return error

    def update_pheromone(self, solutions, scores):
        self.pheromone *= (1 - self.rho)
        for solution, score in zip(solutions, scores):
            for feature in solution:
                self.pheromone[feature] += self.q / score
