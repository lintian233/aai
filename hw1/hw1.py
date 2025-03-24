import heapq
import time
from typing import List, Tuple, Optional

State = Tuple[int, ...]
Move = Tuple[str, State]


class Node:
    def __init__(
        self,
        state: State,
        parent: Optional["Node"],
        action: Optional[str],
        g: int,
        h: int,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.g = g  # 实际步数
        self.h = h  # 启发式估值
        self.f = g + h  # 综合评估值

    def __lt__(self, other: "Node") -> bool:
        return self.f < other.f


def manhattan_distance(current: State, goal: State) -> int:
    """计算曼哈顿距离作为启发函数"""
    distance = 0
    for i in range(9):
        if current[i] == 0:
            continue
        x1, y1 = i % 3, i // 3
        for j in range(9):
            if goal[j] == current[i]:
                x2, y2 = j % 3, j // 3
                distance += abs(x1 - x2) + abs(y1 - y2)
                break
    return distance


def misplaced_tiles(current: State, goal: State) -> int:
    """计算错位方块数启发函数（0除外）"""
    count = 0
    for i in range(9):
        if current[i] != 0 and current[i] != goal[i]:
            count += 1
    return count


def get_neighbors(state: State) -> List[Move]:
    """生成所有可能的合法移动"""
    moves = []
    empty_index = state.index(0)
    ex, ey = empty_index % 3, empty_index // 3

    directions = [("上", (0, 1)), ("下", (0, -1)), ("左", (1, 0)), ("右", (-1, 0))]

    for action, (dx, dy) in directions:
        nx, ny = ex + dx, ey + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_index = ny * 3 + nx
            new_state = list(state)
            new_state[empty_index], new_state[new_index] = (
                new_state[new_index],
                new_state[empty_index],
            )
            moves.append((action, tuple(new_state)))
    return moves


def a_star_search(initial: State, goal: State, h_func) -> tuple:
    """执行A*搜索算法，返回(路径, 扩展节点数)"""
    open_heap = []
    visited = set()
    nodes_expanded = 0

    initial_node = Node(
        state=initial, parent=None, action=None, g=0, h=h_func(initial, goal)
    )
    heapq.heappush(open_heap, (initial_node.f, initial_node))

    while open_heap:
        current = heapq.heappop(open_heap)[1]
        nodes_expanded += 1

        if current.state == goal:
            path = []
            while current.parent is not None:
                path.append((current.action, current.state))
                current = current.parent
            path.reverse()
            return (path, nodes_expanded)

        if current.state in visited:
            continue
        visited.add(current.state)

        for action, neighbor_state in get_neighbors(current.state):
            if neighbor_state in visited:
                continue

            neighbor_node = Node(
                state=neighbor_state,
                parent=current,
                action=action,
                g=current.g + 1,
                h=h_func(neighbor_state, goal),
            )
            heapq.heappush(open_heap, (neighbor_node.f, neighbor_node))

    return (None, nodes_expanded)


def format_state(state: State) -> str:
    """将状态转换为3x3网格的可视化字符串"""
    return "\n".join(
        " ".join(str(num) if num != 0 else " " for num in state[i * 3 : (i + 1) * 3])
        for i in range(3)
    )


def display_solution(initial: State, path: List[Move]) -> None:
    """可视化显示解决方案步骤"""
    print("初始状态:")
    print(format_state(initial))
    print("-" * 20)

    for step, (action, state) in enumerate(path, 1):
        print(f"步骤 {step}: 执行移动 [{action}]")
        print(format_state(state))
        print("-" * 20)


#
if __name__ == "__main__":
    initial_state = (2, 8, 3, 1, 6, 4, 7, 0, 5)
    goal_state = (1, 2, 3, 8, 0, 4, 7, 6, 5)

    print("使用曼哈顿距离启发函数:")
    start_time = time.time()
    solution1, expanded1 = a_star_search(initial_state, goal_state, manhattan_distance)
    time1 = time.time() - start_time

    print("\n使用错位方块数启发函数:")
    start_time = time.time()
    solution2, expanded2 = a_star_search(initial_state, goal_state, misplaced_tiles)
    time2 = time.time() - start_time

    print("\n=== 曼哈顿距离结果 ===")
    if solution1:
        print(f"求解时间: {time1:.4f}秒")
        print(f"扩展节点数: {expanded1}")
        print(f"移动步数: {len(solution1)}")
        display_solution(initial_state, solution1)
    else:
        print("无解")

    print("\n=== 错位方块数结果 ===")
    if solution2:
        print(f"求解时间: {time2:.4f}秒")
        print(f"扩展节点数: {expanded2}")
        print(f"移动步数: {len(solution2)}")
        display_solution(initial_state, solution2)
    else:
        print("无解")
