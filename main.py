import random

class SOPProblem:
    def __init__(self, costs, precedences):
        self.costs = costs
        self.n = len(costs)
        self.precedences = precedences  # danh sách set()

    def is_feasible(self, path, next_node):
        """Kiểm tra ràng buộc precedence: next_node chỉ được chọn nếu tất cả nút phải-đứng-trước đã có trong path."""
        required = self.precedences[next_node]
        return required.issubset(path)


class HASSOP:
    def __init__(self, problem, m=10, s=10, rho=0.1, phi=0.1, max_iter=100):
        self.p = problem
        self.m = m
        self.s = s
        self.rho = rho
        self.phi = phi
        self.max_iter = max_iter

        n = problem.n
        self.pheromone = [[1.0 for _ in range(n)] for _ in range(n)]  # init pheromone
        self.eta = [[0 for _ in range(n)] for _ in range(n)]          # heuristic = 1/cost

        # Tạo heuristic
        for i in range(n):
            for j in range(n):
                if problem.costs[i][j] < 1e9:
                    self.eta[i][j] = 1.0 / problem.costs[i][j] if problem.costs[i][j] > 0 else 1
                else:
                    self.eta[i][j] = 0

    def construct_solution(self):
        """Dàn kiến xây một đường đi hợp lệ từ 0 đến n-1."""
        n = self.p.n
        path = [0]
        current = 0

        while len(path) < n:
            candidates = []
            for j in range(n):
                if j not in path and self.eta[current][j] > 0:
                    if self.p.is_feasible(path, j):
                        candidates.append(j)

            if not candidates:
                return None  # dead end → invalid solution

            # Chọn theo xác suất: τ^s * η^s
            probs = []
            for j in candidates:
                tau = self.pheromone[current][j]
                eta = self.eta[current][j]
                probs.append((tau ** self.s) * (eta ** self.s))

            total = sum(probs)
            probs = [x / total for x in probs]

            # Roulette wheel
            r = random.random()
            cumulative = 0
            for c, p in zip(candidates, probs):
                cumulative += p
                if r <= cumulative:
                    next_node = c
                    break

            path.append(next_node)
            current = next_node

        return path

    def path_cost(self, path):
        total = 0
        for i in range(len(path) - 1):
            c = self.p.costs[path[i]][path[i+1]]
            if c >= 1e9:
                return 1e18
            total += c
        return total

    def update_pheromone(self, best_path, best_cost):
        n = self.p.n

        # 1. Evaporation
        for i in range(n):
            for j in range(n):
                self.pheromone[i][j] *= (1 - self.rho)

        # 2. Deposit on best path
        deposit = 1.0 / best_cost
        for i in range(len(best_path) - 1):
            a = best_path[i]
            b = best_path[i+1]
            self.pheromone[a][b] += deposit

        # 3. Local update (phi)
        for i in range(n):
            for j in range(n):
                self.pheromone[i][j] = (1 - self.phi) * self.pheromone[i][j] + self.phi * 1.0

    def solve(self):
        best_path = None
        best_cost = 1e18

        for _ in range(self.max_iter):
            for _ in range(self.m):
                path = self.construct_solution()
                if path is None:
                    continue

                cost = self.path_cost(path)

                if cost < best_cost:
                    best_cost = cost
                    best_path = path

            if best_path is not None:
                self.update_pheromone(best_path, best_cost)

        return best_path, best_cost



# ----------------------
#        MAIN
# ----------------------
if __name__ == "__main__":
    # 1. Định nghĩa bài toán SOP
    n = 5
    INF = 1e9

    costs = [
        [0,   1,   10,  10,  INF],
        [INF, 0,   1,   5,   INF],
        [INF, INF, 0,   1,   10],
        [INF, INF, 2,   0,   1],
        [INF, INF, INF, INF, 0]
    ]

    precedences = [
        set(),
        set(),
        set(),
        {1},   # 3 is allowed only after 1
        set()
    ]

    print("--- Khởi tạo Bài toán SOP ---")
    problem = SOPProblem(costs, precedences)

    # 2. HAS-SOP Solver
    solver = HASSOP(problem, m=10, s=10, rho=0.1, phi=0.1, max_iter=100)

    # 3. Chạy
    best_path, best_cost = solver.solve()

    print("\nBest path:", best_path)
    print("Best cost:", best_cost)

    if best_path == [0, 1, 2, 3, 4] and best_cost == 4:
        print("\nKết quả: Đã tìm thấy đường đi tối ưu chính xác!")
    else:
        print("\nKết quả: Tìm thấy một giải pháp (có thể không tối ưu).")
