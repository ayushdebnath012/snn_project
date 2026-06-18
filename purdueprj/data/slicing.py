import math
import torch


# ---------------------------------------------------------------------------
# Original slicing strategies (unchanged)
# ---------------------------------------------------------------------------

def slice_random(points, T=16):
    N = points.shape[0]
    perm = torch.randperm(N)
    return torch.chunk(perm, T)


def slice_radial(points, T=16):
    center = points.mean(dim=0)
    dist = torch.norm(points - center, dim=1)
    perm = torch.argsort(dist)   # inner -> outer
    return torch.chunk(perm, T)


def slice_radial_batch(points, T=16):
    """
    Vectorized radial slicing for a batch of point clouds.
    points: [B, N, 3]
    return: indices [B, N] sorted by distance from center
    """
    B, N, _ = points.shape
    center = points.mean(dim=1, keepdim=True)  # [B, 1, 3]
    dist = torch.norm(points - center, dim=2)  # [B, N]
    perm = torch.argsort(dist, dim=1)          # [B, N]
    return perm


def slice_pca(points, T=16):
    X = points - points.mean(dim=0)
    U, S, V = torch.pca_lowrank(X)
    pc1 = V[:, 0]
    proj = X @ pc1
    perm = torch.argsort(proj)
    return torch.chunk(perm, T)


# ---------------------------------------------------------------------------
# Novel: FPS Hierarchical Slicing  (inspired by SPM's HDE)
# ---------------------------------------------------------------------------

def farthest_point_sample(pts, n_samples):
    """
    Iterative Farthest Point Sampling (FPS) for a single point cloud.
    pts       : [N, 3]
    n_samples : int
    Returns   : indices [n_samples]

    FPS guarantees maximum spatial coverage — SPM's HDE uses FPS to
    produce diverse temporal encodings across early/middle/late stages.
    """
    N = pts.shape[0]
    n_samples = min(n_samples, N)
    device = pts.device

    selected = torch.zeros(n_samples, dtype=torch.long, device=device)
    distances = torch.full((N,), float('inf'), device=device)
    farthest = torch.randint(0, N, (1,), device=device).item()

    for i in range(n_samples):
        selected[i] = farthest
        centroid = pts[farthest]
        dist = ((pts - centroid) ** 2).sum(-1)
        distances = torch.minimum(distances, dist)
        farthest = distances.argmax().item()

    return selected


def slice_fps_hierarchical_batch(points, T=16):
    """
    Novel: FPS-based Hierarchical Slicing inspired by SPM's HDE.

    SPM divides FPS into three temporal stages:
      Early  (unstable random init) -> Finite Forward Sliding
      Middle (stable skeleton)      -> fixed window, preserves structure
      Late   (redundant / noisy)    -> Infinite Backward Extension

    Our implementation:
      1. Run FPS to get T representative 'centre' points — spatially spread.
      2. Assign every original point to its nearest FPS centre.
      3. Each slice contains points assigned to one FPS centre.
         The FPS ordering (early = diverse spread, later = infill)
         approximates the HDE temporal hierarchy.

    Advantage over radial slicing:
      - Points are grouped by spatial locality (cluster-like), not just
        distance from one global centre.
      - The temporal sequence is more meaningful for SNNs: early slices
        see scattered structural anchors; later slices see detail.

    points : [B, N, 3]
    T      : number of time steps
    Returns: [B, T, N//T, 3]
    """
    B, N, C = points.shape
    points_per_slice = N // T
    device = points.device

    all_slices = []

    for b in range(B):
        pts_b = points[b]   # [N, 3]

        # Step 1: FPS -> T representative centres
        fps_idx = farthest_point_sample(pts_b, T)    # [T]
        centres = pts_b[fps_idx]                      # [T, 3]

        # Step 2: Assign every point to nearest centre
        diff = pts_b.unsqueeze(0) - centres.unsqueeze(1)   # [T, N, 3]
        dist2 = (diff ** 2).sum(-1)                         # [T, N]
        assign = dist2.argmin(dim=0)                        # [N]

        # Step 3: Build fixed-size slices
        slices_b = []
        for t in range(T):
            mask = (assign == t).nonzero(as_tuple=True)[0]
            if mask.numel() == 0:
                mask = torch.randperm(N, device=device)[:points_per_slice]
            elif mask.numel() < points_per_slice:
                # Tile mask until we have enough, then truncate
                reps = math.ceil(points_per_slice / mask.numel())
                mask = mask.repeat(reps)[:points_per_slice]
            else:
                mask = mask[:points_per_slice]
            slices_b.append(pts_b[mask])   # [points_per_slice, 3]

        all_slices.append(torch.stack(slices_b, dim=0))   # [T, pps, 3]

    return torch.stack(all_slices, dim=0)   # [B, T, points_per_slice, 3]
