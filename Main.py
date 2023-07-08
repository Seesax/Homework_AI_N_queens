import numpy as np
import random
import heapq


class State:
    def __init__(self, heuristic, state):
        self.heuristic = heuristic
        self.state = state

    def __lt__(self, other):
        return self.heuristic < other.heuristic


class NQueensProblem:
    def __init__(self, n):
        self.n = n

    def is_valid(self, board, row, col):
        for i in range(row):
            if board[i] == col or board[i] - i == col - row or board[i] + i == col + row:
                return False
        return True

    def backtrack(self, board, row):
        if row == self.n:
            return True

        for col in range(self.n):
            if self.is_valid(board, row, col):
                board[row] = col
                if self.backtrack(board, row + 1):
                    return True
        return False

    def solve_n_queens(self):
        board = np.zeros(self.n, dtype=int)
        if not self.backtrack(board, 0):
            return None
        return board

    def count_attacking_pairs(self, board):
        count = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if board[i] == board[j] or board[i] - i == board[j] - j or board[i] + i == board[j] + j:
                    count += 1
        return count

    def generate_successors(self, board):
        successors = []
        for col in range(self.n):
            for row in range(self.n):
                if board[row] != col:
                    successor = np.copy(board)
                    successor[row] = col
                    successors.append(successor)
        return successors

    def graph_search_astar(self):
        initial_state = np.random.permutation(self.n)
        queue = [State(self.count_attacking_pairs(initial_state), initial_state)]
        heapq.heapify(queue)
        while queue:
            state = heapq.heappop(queue)
            if self.count_attacking_pairs(state.state) == 0:
                return state.state
            successors = self.generate_successors(state.state)
            for successor in successors:
                heuristic = self.count_attacking_pairs(successor)
                heapq.heappush(queue, State(heuristic, successor))

    def genetic_algorithm(self, population_size=100, max_generations=1000):
        population = []
        for _ in range(population_size):
            state = np.random.permutation(self.n)
            population.append(state)

        for _ in range(max_generations):
            fitness_scores = [self.count_attacking_pairs(state) for state in population]
            if 0 in fitness_scores:
                return population[fitness_scores.index(0)]

            probabilities = [1 / (score + 1) for score in fitness_scores]
            probabilities_sum = sum(probabilities)
            probabilities = [p / probabilities_sum for p in probabilities]

            new_population = []
            for _ in range(population_size):
                parent1 = random.choices(population, probabilities)[0]
                parent2 = random.choices(population, probabilities)[0]
                offspring = self.crossover(parent1, parent2)
                offspring = self.mutate(offspring)
                new_population.append(offspring)

            population = new_population

        best_state = min(population, key=lambda x: self.count_attacking_pairs(x))
        return best_state

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, self.n - 1)
        offspring = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        return offspring

    def mutate(self, state):
        mutated_state = np.copy(state)
        index = random.randint(0, self.n - 1)
        value = random.randint(0, self.n - 1)
        mutated_state[index] = value
        return mutated_state


def display_solution(board):
    n = len(board)
    for i in range(n):
        row = ["Q" if j == board[i] else "*" for j in range(n)]
        print(" ".join(row))


def main():
    print("N-Queens Problem Solver")
    print("-----------------------")
    n = int(input("Enter the number of queens (N): "))

    problem = NQueensProblem(n)

    while True:
        print("\nSelect an algorithm:")
        print("1. Uniform-cost search")
        print("2. Graph-search A* with MIN-CONFLICT heuristic")
        print("3. Genetic algorithm")
        print("0. Exit")
        choice = int(input("Enter your choice: "))

        if choice == 0:
            print("Exiting...")
            break

        solution = None
        if choice == 1:
            solution = problem.solve_n_queens()
        elif choice == 2:
            solution = problem.graph_search_astar()
        elif choice == 3:
            solution = problem.genetic_algorithm()
        else:
            print("Invalid choice. Please try again.")
            continue

        if solution is None:
            print("No solution found.")
        else:
            print("\nSolution:")
            display_solution(solution)


if __name__ == "__main__":
    main()
