"""A* grid search for generating training data (2D and 3D)."""
import torch
import heapq


def astar_2d(voxels, start, goal, voxel_res=32):
    """2D A* on voxel grid. Returns path as (N, 2) tensor of normalized coords.

    Args:
        voxels: (H, W) tensor, 1=obstacle, 0=free
        start:  (2,) tensor, normalized [0,1] coords
        goal:   (2,) tensor, normalized [0,1] coords
    Returns:
        path: (N, 2) tensor or None if no path found
    """
    H, W = voxels.shape
    si, sj = int(start[1] * H), int(start[0] * W)
    gi, gj = int(goal[1] * H), int(goal[0] * W)
    si, sj = max(0, min(H-1, si)), max(0, min(W-1, sj))
    gi, gj = max(0, min(H-1, gi)), max(0, min(W-1, gj))
    if voxels[si, sj] > 0 or voxels[gi, gj] > 0:
        return None

    def heuristic(i, j):
        return abs(i - gi) + abs(j - gj)

    open_set = [(heuristic(si, sj), 0, si, sj)]
    came_from = {}
    g_score = {(si, sj): 0}
    visited = set()
    # 8-connected neighbors
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                  (0, 1), (1, -1), (1, 0), (1, 1)]

    while open_set:
        _, g, i, j = heapq.heappop(open_set)
        if (i, j) in visited:
            continue
        visited.add((i, j))
        if (i, j) == (gi, gj):
            # Reconstruct path
            path = [(i, j)]
            while (i, j) in came_from:
                i, j = came_from[(i, j)]
                path.append((i, j))
            path.reverse()
            pts = torch.tensor([[j / W, i / H] for i, j in path], dtype=torch.float32)
            return pts
        for di, dj in neighbors:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and voxels[ni, nj] == 0:
                cost = 1.414 if di != 0 and dj != 0 else 1.0
                ng = g + cost
                if ng < g_score.get((ni, nj), float('inf')):
                    g_score[(ni, nj)] = ng
                    came_from[(ni, nj)] = (i, j)
                    heapq.heappush(open_set, (ng + heuristic(ni, nj), ng, ni, nj))
    return None


def astar_3d(voxels, start, goal, voxel_res=32):
    """3D A* on voxel grid.

    Args:
        voxels: (D, H, W) tensor
        start:  (3,) tensor, normalized
        goal:   (3,) tensor, normalized
    Returns:
        path: (N, 3) tensor or None
    """
    D, H, W = voxels.shape
    si, sj, sk = int(start[1]*H), int(start[0]*W), int(start[2]*D)
    gi, gj, gk = int(goal[1]*H), int(goal[0]*W), int(goal[2]*D)
    si, sj, sk = max(0,min(H-1,si)), max(0,min(W-1,sj)), max(0,min(D-1,sk))
    gi, gj, gk = max(0,min(H-1,gi)), max(0,min(W-1,gj)), max(0,min(D-1,gk))
    if voxels[sk, si, sj] > 0 or voxels[gk, gi, gj] > 0:
        return None

    def heuristic(i, j, k):
        return abs(i-gi) + abs(j-gj) + abs(k-gk)

    open_set = [(heuristic(si, sj, sk), 0, si, sj, sk)]
    came_from = {}
    g_score = {(si, sj, sk): 0}
    visited = set()
    # 26-connected neighbors
    neighbors = [(di, dj, dk) for di in (-1,0,1) for dj in (-1,0,1) for dk in (-1,0,1)
                 if not (di==0 and dj==0 and dk==0)]

    while open_set:
        _, g, i, j, k = heapq.heappop(open_set)
        if (i, j, k) in visited:
            continue
        visited.add((i, j, k))
        if (i, j, k) == (gi, gj, gk):
            path = [(i, j, k)]
            while (i, j, k) in came_from:
                i, j, k = came_from[(i, j, k)]
                path.append((i, j, k))
            path.reverse()
            pts = torch.tensor([[j/W, i/H, k/D] for i, j, k in path], dtype=torch.float32)
            return pts
        for di, dj, dk in neighbors:
            ni, nj, nk = i+di, j+dj, k+dk
            if 0<=ni<H and 0<=nj<W and 0<=nk<D and voxels[nk, ni, nj]==0:
                cost = (di**2 + dj**2 + dk**2) ** 0.5
                ng = g + cost
                if ng < g_score.get((ni, nj, nk), float('inf')):
                    g_score[(ni, nj, nk)] = ng
                    came_from[(ni, nj, nk)] = (i, j, k)
                    heapq.heappush(open_set, (ng+heuristic(ni,nj,nk), ng, ni, nj, nk))
    return None
