import numpy as np
import random
import heapq


class State:
    def __init__(self, board):
        self.board = board  # Trạng thái của bàn cờ
        self.attacking_pairs = None  # Số cặp quân hậu tấn công nhau

    def __lt__(self, other):
        return self.attacking_pairs < other.attacking_pairs


class NQueensProblem:
    def __init__(self, n):
        self.n = n  # Số lượng quân hậu và kích thước bàn cờ

    def is_valid(self, board, row, col):
        # Kiểm tra xem việc đặt một quân hậu tại (row, col) có hợp lệ không
        # bằng cách kiểm tra cột, đường chéo và đường chéo ngược

        # Kiểm tra cột
        for i in range(row):
            if board[i][col] == 1:
                return False

        # Kiểm tra đường chéo phía trên và bên trái
        for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
            if board[i][j] == 1:
                return False

        # Kiểm tra đường chéo phía trên và bên phải
        for i, j in zip(range(row, -1, -1), range(col, self.n)):
            if board[i][j] == 1:
                return False

        return True

    def backtrack(self, board, row):
        # Thuật toán quay lui để đặt quân hậu lần lượt trên từng hàng

        # Trường hợp cơ bản: tất cả các quân hậu đã được đặt
        if row == self.n:
            return True

        for col in range(self.n):
            if self.is_valid(board, row, col):
                board[row][col] = 1  # Đặt quân hậu vào vị trí (row, col)
                if self.backtrack(board, row + 1):
                    return True
                board[row][col] = 0  # Quay lui: loại bỏ quân hậu khỏi vị trí (row, col)

        return False

    def solve_n_queens(self):
        # Hàm chính để giải bài toán N-Queens
        board = np.zeros((self.n, self.n), dtype=int)  # Khởi tạo bàn cờ
        if not self.backtrack(board, 0):  # Gọi hàm backtrack để tìm kiếm lời giải
            return None
        return board

    def count_attacking_pairs(self, board):
        # Đếm số cặp quân hậu tấn công nhau trên bàn cờ
        count = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.is_attacking(board, i, j):
                    count += 1
        return count

    def is_attacking(self, board, i, j):
        # Kiểm tra xem quân hậu tại hàng i và j có tấn công nhau không
        return board[i][1] == board[j][1] or abs(board[i][1] - board[j][1]) == abs(i - j)

    def generate_successors(self, board):
        # Tạo ra tất cả các trạng thái kế tiếp từ trạng thái hiện tại bằng cách di chuyển một quân hậu
        # sang một ô trống trong cùng một cột

        successors = []
        for i in range(self.n):
            for j in range(self.n):
                if board[i][1] != j:  # Chỉ di chuyển nếu ô đó không chứa quân hậu
                    successor = np.copy(board)
                    successor[i][1] = j  # Di chuyển quân hậu sang ô mới
                    successors.append(successor)
        return successors

    def graph_search_astar(self):
        # Tìm kiếm đồ thị A* với heuristics MIN-CONFLICT

        initial_state = State(np.zeros((self.n, 2), dtype=int))  # Trạng thái ban đầu
        initial_state.attacking_pairs = self.count_attacking_pairs(initial_state.board)  # Heuristics

        priority_queue = [initial_state]  # Hàng đợi ưu tiên với trạng thái ban đầu
        heapq.heapify(priority_queue)

        while priority_queue:
            state = heapq.heappop(priority_queue)

            if state.attacking_pairs == 0:  # Tìm thấy lời giải
                return state.board

            successors = self.generate_successors(state.board)
            for successor in successors:
                new_state = State(successor)
                new_state.attacking_pairs = self.count_attacking_pairs(successor)
                heapq.heappush(priority_queue, new_state)

    def genetic_algorithm(self, population_size=100, max_generations=1000):
        # Giải thuật di truyền

        population = []
        for _ in range(population_size):
            state = np.zeros((self.n, 2), dtype=int)
            state[:, 1] = random.sample(range(self.n), self.n)  # Tạo ra một cá thể ngẫu nhiên
            population.append(state)

        for generation in range(max_generations):
            new_population = []
            for _ in range(population_size):
                parent1 = random.choice(population)  # Lựa chọn cha mẹ
                parent2 = random.choice(population)
                offspring = self.crossover(parent1, parent2)  # Lai ghép
                offspring = self.mutate(offspring)  # Đột biến
                new_population.append(offspring)

            population = new_population

        best_state = min(population, key=lambda x: self.count_attacking_pairs(x))  # Tìm kiếm cá thể tốt nhất
        return best_state

    def crossover(self, parent1, parent2):
        # Lai ghép bằng cách chia tỉ lệ ngẫu nhiên giữa hai cha mẹ

        crossover_point = random.randint(0, self.n - 1)
        offspring = np.vstack((parent1[:crossover_point], parent2[crossover_point:]))
        return offspring

    def mutate(self, state):
        # Đột biến bằng cách thay đổi một giá trị ngẫu nhiên trong cá thể

        mutated_state = np.copy(state)
        index = random.randint(0, self.n - 1)
        value = random.randint(0, self.n - 1)
        mutated_state[index][1] = value
        return mutated_state


def display_solution(board):
    # Hiển thị lời giải trên bàn cờ

    n = len(board)
    for i in range(n):
        for j in range(n):
            if board[i][1] == j:
                print("Q ", end="")
            else:
                print("* ", end="")
        print()


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
