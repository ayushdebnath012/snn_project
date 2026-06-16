#!/usr/bin/env python3
"""
run_mn40_colab.py
=================
Single-file ASP initial MN40 training script for Google Colab.

Usage (in a Colab code cell):
    !python run_mn40_colab.py
    # or with custom args:
    !python run_mn40_colab.py --epochs 50 --batch_size 32

Results saved to /content/asp_results_mn40/
  - best_model_mn40.pth   (best checkpoint)
  - history_mn40.json     (per-epoch metrics)
"""

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 1 — Install dependencies                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
import subprocess, sys

for pkg in ['trimesh', 'kagglehub']:
    subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'], check=True)

import os
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('WARNING: No GPU — go to Runtime > Change runtime type > T4 GPU')

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 2 — Write all source modules to /content/asp/                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
import os, sys, textwrap

ROOT = '/content/asp'
os.makedirs(ROOT, exist_ok=True)
sys.path.insert(0, ROOT)

FILES = {}
FILES["models/__init__.py"] = ""
FILES["models/snn_layers.py"] = "import torch\nimport torch.nn as nn\n\n\nclass SurrogateSpike(torch.autograd.Function):\n    @staticmethod\n    def forward(ctx, x):\n        out = (x > 0).float()\n        ctx.save_for_backward(x)\n        return out\n\n    @staticmethod\n    def backward(ctx, grad_output):\n        (x,) = ctx.saved_tensors\n        grad = 1.0 / (1 + torch.abs(x))**2\n        return grad_output * grad\n\n\nspike_fn = SurrogateSpike.apply\n\n\nclass LIFLayer(nn.Module):\n    def __init__(self, in_features, out_features, tau=0.9):\n        super().__init__()\n        self.fc = nn.Linear(in_features, out_features)\n        self.tau = tau\n        self.register_buffer(\"mem\", None)\n\n    def reset_state(self, batch_size, device=None):\n        dev = device if device else next(self.fc.parameters()).device\n        self.mem = torch.zeros(batch_size, self.fc.out_features, device=dev)\n        self.spike_count = torch.tensor(0.0, device=dev)\n        self.step_count  = torch.tensor(0,   device=dev)\n        self.batch_size  = batch_size\n\n    def firing_rate(self):\n        if self.step_count == 0:\n            return 0.0\n        return (self.spike_count / (self.fc.out_features * self.step_count *\n                getattr(self, \"batch_size\", 1))).item()\n\n    def forward(self, x):\n        cur = self.fc(x)\n        self.mem = self.tau * self.mem + cur\n        spk = spike_fn(self.mem - 1.0)\n        self.mem = self.mem * (1 - spk)\n        if not hasattr(self, \"spike_count\"):\n            self.spike_count = torch.tensor(0.0, device=cur.device)\n            self.step_count  = torch.tensor(0,   device=cur.device)\n            self.batch_size  = x.shape[0]\n        self.spike_count = self.spike_count + spk.detach().sum()\n        self.step_count  = self.step_count + 1\n        return spk, self.mem\n\n\nclass LearnableLIFLayer(nn.Module):\n    def __init__(self, in_features, out_features, tau_init=0.9, vth_init=1.0):\n        super().__init__()\n        self.fc = nn.Linear(in_features, out_features)\n        self.out_features = out_features\n        import math\n        tau_raw_init = math.log(tau_init / (1 - tau_init))\n        self.tau_raw = nn.Parameter(torch.full((out_features,), tau_raw_init))\n        self.vth_raw = nn.Parameter(torch.full((out_features,), float(vth_init)))\n        self.register_buffer(\"mem\", None)\n        self.register_buffer(\"spike_count\", torch.tensor(0.0))\n        self.register_buffer(\"step_count\",  torch.tensor(0))\n\n    @property\n    def tau(self):\n        return torch.sigmoid(self.tau_raw)\n\n    @property\n    def vth(self):\n        return torch.nn.functional.softplus(self.vth_raw)\n\n    def reset_state(self, batch_size, device=None):\n        dev = device if device else next(self.fc.parameters()).device\n        self.mem         = torch.zeros(batch_size, self.out_features, device=dev)\n        self.spike_count = torch.tensor(0.0, device=dev)\n        self.step_count  = torch.tensor(0,   device=dev)\n        self.batch_size  = batch_size\n\n    def firing_rate(self):\n        if self.step_count == 0:\n            return 0.0\n        return (self.spike_count / (self.out_features * self.step_count *\n                getattr(self, \"batch_size\", 1))).item()\n\n    def forward(self, x):\n        cur = self.fc(x)\n        self.mem = self.tau * self.mem + cur\n        spk = spike_fn(self.mem - self.vth)\n        self.mem = self.mem * (1 - spk)\n        self.spike_count = self.spike_count + spk.detach().sum()\n        self.step_count  = self.step_count + 1\n        return spk, self.mem\n"
FILES["models/pointnet_backbone.py"] = "import torch\nimport torch.nn as nn\nfrom models.snn_layers import LIFLayer, LearnableLIFLayer\n\n\ndef knn_graph(pts, k):\n    B, N, _ = pts.shape\n    diff  = pts.unsqueeze(2) - pts.unsqueeze(1)\n    dist2 = (diff ** 2).sum(-1)\n    eye   = torch.eye(N, device=pts.device, dtype=torch.bool).unsqueeze(0)\n    dist2 = dist2.masked_fill(eye, float('inf'))\n    idx   = dist2.topk(k, dim=-1, largest=False).indices\n    idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, 3)\n    neighbours = torch.gather(\n        pts.unsqueeze(1).expand(-1, N, -1, -1), 2, idx_exp)\n    return neighbours\n\n\nclass LocalKNNBackbone(nn.Module):\n    def __init__(self, hidden_dims=[128, 256, 512], k=16, learnable_lif=True):\n        super().__init__()\n        self.k = k\n        LayerCls = LearnableLIFLayer if learnable_lif else LIFLayer\n        in_dim = 3 + 3 * k\n        self.layers = nn.ModuleList()\n        for h in hidden_dims:\n            self.layers.append(LayerCls(in_dim, h))\n            in_dim = h\n        self.out_dim = hidden_dims[-1]\n\n    def reset_state(self, batch_size, device=None):\n        for layer in self.layers:\n            layer.reset_state(batch_size, device)\n\n    def forward(self, pts):\n        B, N, _ = pts.shape\n        neighbours = knn_graph(pts, self.k)\n        rel        = neighbours - pts.unsqueeze(2)\n        rel_flat   = rel.reshape(B, N, self.k * 3)\n        x = torch.cat([pts, rel_flat], dim=-1)\n        self.reset_state(B * N, pts.device)\n        x = x.reshape(B * N, -1)\n        for layer in self.layers:\n            spk, mem = layer(x)\n            x = mem\n        return mem.reshape(B, N, -1)\n\n    def firing_rates(self):\n        rates = {}\n        for i, layer in enumerate(self.layers):\n            if hasattr(layer, 'firing_rate'):\n                rates[f\"knn_layer_{i}\"] = layer.firing_rate()\n        return rates\n"
FILES["models/temporal_snn.py"] = "import torch\nimport torch.nn as nn\nfrom models.snn_layers import LIFLayer, LearnableLIFLayer\n\n\nclass TemporalSNN(nn.Module):\n    def __init__(self, dim=512, num_classes=10, learnable_lif=True):\n        super().__init__()\n        LayerCls = LearnableLIFLayer if learnable_lif else LIFLayer\n        self.lif1 = LayerCls(dim, dim)\n        self.lif2 = LayerCls(dim, dim)\n        self.fc   = nn.Linear(dim, num_classes)\n\n    def reset_state(self, batch_size, device=None):\n        self.lif1.reset_state(batch_size, device)\n        self.lif2.reset_state(batch_size, device)\n\n    def forward(self, x):\n        spk1, mem1 = self.lif1(x)\n        spk2, mem2 = self.lif2(mem1)\n        return self.fc(mem2)\n\n    def firing_rates(self):\n        rates = {}\n        for name, layer in [(\"temporal_lif1\", self.lif1), (\"temporal_lif2\", self.lif2)]:\n            if hasattr(layer, 'firing_rate'):\n                rates[name] = layer.firing_rate()\n        return rates\n"
FILES["models/slice_selection_policy.py"] = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport math\n\nGEO_DIM = 6\n\n\ndef compute_geometry_descriptors(pts, fps_anchors, anchor_assignments):\n    B, N, _ = pts.shape\n    M = fps_anchors.size(1)\n    cloud_centroid = pts.mean(dim=1, keepdim=True)\n    anchor_dist = (fps_anchors - cloud_centroid).norm(dim=-1)\n    mean_intra  = torch.zeros(B, M, device=pts.device)\n    point_count = torch.zeros(B, M, device=pts.device)\n    for m in range(M):\n        mask  = (anchor_assignments == m)\n        count = mask.float().sum(dim=1)\n        point_count[:, m] = count\n        for b in range(B):\n            cluster_pts = pts[b][mask[b]]\n            if cluster_pts.size(0) > 1:\n                centroid = cluster_pts.mean(dim=0)\n                mean_intra[b, m] = (cluster_pts - centroid).norm(dim=-1).mean()\n    avg_count  = N / M\n    norm_count = point_count / (avg_count + 1e-6)\n    G = torch.cat([\n        fps_anchors,\n        anchor_dist.unsqueeze(-1),\n        mean_intra.unsqueeze(-1),\n        norm_count.unsqueeze(-1),\n    ], dim=-1)\n    return G\n\n\nclass SliceSelectionPolicy(nn.Module):\n    def __init__(self, mem_dim, geo_dim=GEO_DIM, d_ssp=64):\n        super().__init__()\n        self.d_ssp = d_ssp\n        self.scale = math.sqrt(d_ssp)\n        self.W_k   = nn.Linear(mem_dim, d_ssp, bias=False)\n        self.W_q   = nn.Linear(geo_dim, d_ssp, bias=False)\n        nn.init.xavier_uniform_(self.W_k.weight)\n        nn.init.xavier_uniform_(self.W_q.weight)\n\n    def forward(self, mem, geo, visited_mask=None):\n        key   = self.W_k(mem)\n        query = self.W_q(geo)\n        scores = torch.bmm(query, key.unsqueeze(-1)).squeeze(-1) / self.scale\n        if visited_mask is not None:\n            scores = scores.masked_fill(visited_mask, float(\"-inf\"))\n        return scores\n\n    def select_gumbel(self, scores, tau=1.0):\n        return F.gumbel_softmax(scores, tau=tau, hard=True, dim=-1)\n\n    def select_greedy(self, scores):\n        idx = scores.argmax(dim=-1)\n        return F.one_hot(idx, num_classes=scores.size(-1)).float()\n"
FILES["models/active_snn.py"] = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom models.pointnet_backbone import LocalKNNBackbone\nfrom models.temporal_snn import TemporalSNN\nfrom models.slice_selection_policy import SliceSelectionPolicy, compute_geometry_descriptors\n\n\nclass ActiveSNN(nn.Module):\n    def __init__(self, point_dims=[128, 256, 512], temporal_dim=512,\n                 num_classes=10, knn_k=16, d_ssp=64):\n        super().__init__()\n        self.temporal_dim = temporal_dim\n        self.num_classes  = num_classes\n        self.backbone = LocalKNNBackbone(hidden_dims=point_dims, k=knn_k, learnable_lif=True)\n        self.temporal = TemporalSNN(dim=temporal_dim, num_classes=num_classes, learnable_lif=True)\n        self.ssp      = SliceSelectionPolicy(mem_dim=temporal_dim, d_ssp=d_ssp)\n        self.register_buffer(\"gumbel_tau\", torch.tensor(1.0))\n\n    def reset_state(self, batch_size, device=None):\n        self.backbone.reset_state(batch_size, device)\n        self.temporal.reset_state(batch_size, device)\n\n    def _get_membrane(self):\n        lif = self.temporal.lif2\n        if lif.mem is None:\n            return None\n        return lif.mem.detach()\n\n    def forward_active_train(self, pts_slices, geo_descriptors):\n        B, T, n_pts, _ = pts_slices.shape\n        device = pts_slices.device\n        self.reset_state(B, device)\n        pts_flat = pts_slices.reshape(B * T, n_pts, 3)\n        self.backbone.reset_state(B * T, device)\n        feat_per_point = self.backbone.forward(pts_flat)\n        all_feats = feat_per_point.mean(dim=1).reshape(B, T, -1)\n        self.backbone.reset_state(B, device)\n        self.temporal.reset_state(B, device)\n        visited_mask = torch.zeros(B, T, dtype=torch.bool, device=device)\n        logits_all   = []\n        mem_state    = torch.zeros(B, self.temporal_dim, device=device)\n        for t in range(T):\n            scores = self.ssp(mem_state, geo_descriptors, visited_mask)\n            tau = self.gumbel_tau.item()\n            w   = self.ssp.select_gumbel(scores, tau=tau)\n            selected_idx = scores.masked_fill(visited_mask, float(\"-inf\")).argmax(dim=-1)\n            for b in range(B):\n                visited_mask[b, selected_idx[b]] = True\n            e_t = (w.unsqueeze(-1) * all_feats).sum(dim=1)\n            logits_t = self.temporal(e_t)\n            logits_all.append(logits_t)\n            mem_state = self._get_membrane()\n            if mem_state is None:\n                mem_state = torch.zeros(B, self.temporal_dim, device=device)\n        return logits_all[-1], logits_all\n\n    def forward_active_infer(self, pts_slices, geo_descriptors, threshold=0.7):\n        B, T, n_pts, _ = pts_slices.shape\n        device = pts_slices.device\n        self.reset_state(B, device)\n        visited_mask = torch.zeros(B, T, dtype=torch.bool, device=device)\n        mem_state    = torch.zeros(B, self.temporal_dim, device=device)\n        slice_order  = []\n        last_logits  = None\n        for t in range(T):\n            with torch.no_grad():\n                scores = self.ssp(mem_state, geo_descriptors, visited_mask)\n                w      = self.ssp.select_greedy(scores)\n            selected_idx = w.argmax(dim=-1)\n            chosen = selected_idx[0].item()\n            slice_order.append(chosen)\n            for b in range(B):\n                visited_mask[b, selected_idx[b]] = True\n            slice_b = pts_slices[:, chosen, :, :]\n            with torch.no_grad():\n                self.backbone.reset_state(B, device)\n                feat_pp = self.backbone(slice_b)\n                e_t     = feat_pp.mean(dim=1)\n                logits_t = self.temporal(e_t)\n                last_logits = logits_t\n            mem_state = self._get_membrane()\n            if mem_state is None:\n                mem_state = torch.zeros(B, self.temporal_dim, device=device)\n            probs  = F.softmax(logits_t, dim=-1)\n            top2   = probs.topk(2, dim=-1).values\n            margin = top2[:, 0] - top2[:, 1]\n            if margin.min().item() > threshold:\n                return last_logits, t + 1, slice_order\n        return last_logits, T, slice_order\n\n    def get_firing_rates(self):\n        rates = {}\n        if hasattr(self.backbone, \"firing_rates\"):\n            rates.update(self.backbone.firing_rates())\n        if hasattr(self.temporal, \"firing_rates\"):\n            rates.update(self.temporal.firing_rates())\n        return rates\n\n    def mean_firing_rate(self):\n        rates = self.get_firing_rates()\n        return sum(rates.values()) / len(rates) if rates else 0.0\n\n    def set_gumbel_tau(self, tau):\n        self.gumbel_tau.fill_(tau)\n\n    def param_count(self):\n        bb   = sum(p.numel() for p in self.backbone.parameters())\n        temp = sum(p.numel() for p in self.temporal.parameters())\n        ssp  = sum(p.numel() for p in self.ssp.parameters())\n        return {\"backbone\": bb, \"temporal\": temp, \"ssp\": ssp, \"total\": bb + temp + ssp}\n"
FILES["data/__init__.py"] = ""
FILES["data/modelnet.py"] = "import os\nimport torch\nimport numpy as np\nfrom torch.utils.data import Dataset\n\n\nclass ModelNetDataset(Dataset):\n    def __init__(self, root, num_points=1024, split='train'):\n        self.root       = root\n        self.num_points = num_points\n        self.split      = split\n        self.files      = self._scan_files()\n        self.data, self.labels = self._load_all()\n\n    def _scan_files(self):\n        items = []\n        for class_name in sorted(os.listdir(self.root)):\n            class_path = os.path.join(self.root, class_name, self.split)\n            if not os.path.isdir(class_path):\n                continue\n            label = sorted(os.listdir(self.root)).index(class_name)\n            for f in os.listdir(class_path):\n                if f.endswith(('.npy', '.txt', '.off')):\n                    items.append((os.path.join(class_path, f), label))\n        return items\n\n    def _load_points(self, path):\n        if path.endswith('.npy'):\n            return np.load(path).astype(np.float32)\n        elif path.endswith('.txt'):\n            return np.loadtxt(path).astype(np.float32)\n        elif path.endswith('.off'):\n            return self._load_off(path)\n        raise ValueError(f\"Unsupported: {path}\")\n\n    def _load_off(self, path):\n        try:\n            import trimesh\n            mesh = trimesh.load(path)\n            pts, _ = trimesh.sample.sample_surface(mesh, self.num_points)\n            return pts.astype(np.float32)\n        except Exception:\n            # fallback: parse OFF manually\n            with open(path) as f:\n                lines = f.read().splitlines()\n            start = 1\n            if lines[0].strip() == 'OFF':\n                start = 1\n            elif lines[0].strip().startswith('OFF'):\n                lines[0] = lines[0][3:]\n                start = 0\n            n_verts = int(lines[start].split()[0])\n            verts = []\n            for i in range(start + 1, start + 1 + n_verts):\n                verts.append([float(v) for v in lines[i].split()[:3]])\n            pts = np.array(verts, dtype=np.float32)\n            idx = np.random.choice(len(pts), self.num_points,\n                                   replace=(len(pts) < self.num_points))\n            return pts[idx]\n\n    def _load_all(self):\n        all_pts, all_labels = [], []\n        for path, label in self.files:\n            pts = self._load_points(path)\n            if not path.endswith('.off'):\n                if pts.shape[0] >= self.num_points:\n                    idx = np.random.choice(pts.shape[0], self.num_points, replace=False)\n                    pts = pts[idx]\n                else:\n                    pad = self.num_points - pts.shape[0]\n                    rep = np.random.choice(pts.shape[0], pad, replace=True)\n                    pts = np.vstack([pts, pts[rep]])\n            all_pts.append(pts)\n            all_labels.append(label)\n        return np.array(all_pts), np.array(all_labels)\n\n    def __len__(self):\n        return len(self.labels)\n\n    def __getitem__(self, idx):\n        pts   = self.data[idx].copy()\n        label = self.labels[idx]\n        np.random.shuffle(pts)\n        return torch.tensor(pts, dtype=torch.float32), torch.tensor(label, dtype=torch.long)\n"
FILES["data/slicing.py"] = "import math\nimport torch\n\n\ndef farthest_point_sample(pts, n_samples):\n    N = pts.shape[0]\n    n_samples = min(n_samples, N)\n    device    = pts.device\n    selected  = torch.zeros(n_samples, dtype=torch.long, device=device)\n    distances = torch.full((N,), float('inf'), device=device)\n    farthest  = torch.randint(0, N, (1,), device=device).item()\n    for i in range(n_samples):\n        selected[i] = farthest\n        centroid = pts[farthest]\n        dist = ((pts - centroid) ** 2).sum(-1)\n        distances = torch.minimum(distances, dist)\n        farthest  = distances.argmax().item()\n    return selected\n\n\ndef slice_fps_hierarchical_batch(points, T=16):\n    B, N, C = points.shape\n    points_per_slice = N // T\n    device = points.device\n    all_slices = []\n    for b in range(B):\n        pts_b   = points[b]\n        fps_idx = farthest_point_sample(pts_b, T)\n        centres = pts_b[fps_idx]\n        diff    = pts_b.unsqueeze(0) - centres.unsqueeze(1)\n        dist2   = (diff ** 2).sum(-1)\n        assign  = dist2.argmin(dim=0)\n        slices_b = []\n        for t in range(T):\n            mask = (assign == t).nonzero(as_tuple=True)[0]\n            if mask.numel() == 0:\n                mask = torch.randperm(N, device=device)[:points_per_slice]\n            elif mask.numel() < points_per_slice:\n                reps = math.ceil(points_per_slice / mask.numel())\n                mask = mask.repeat(reps)[:points_per_slice]\n            else:\n                mask = mask[:points_per_slice]\n            slices_b.append(pts_b[mask])\n        all_slices.append(torch.stack(slices_b, dim=0))\n    return torch.stack(all_slices, dim=0)\n"
FILES["training/__init__.py"] = ""
FILES["training/optimizers.py"] = "import torch\n\ndef build_optimizer(model, lr=1e-3, weight_decay=1e-4):\n    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)\n"
FILES["training/metrics.py"] = "import torch\n\ndef accuracy(logits, labels):\n    preds = logits.argmax(dim=1)\n    return (preds == labels).float().mean().item()\n"
FILES["training/loss_active.py"] = "import torch\nimport torch.nn.functional as F\n\n\ndef active_loss(logits_final, logits_all, labels, model,\n                lam_aux=0.3, lam_exit=0.1, lam_fr=0.05):\n    # L_CE\n    l_ce = F.cross_entropy(logits_final, labels)\n\n    # L_aux\n    T = len(logits_all)\n    if T > 1:\n        l_aux = sum(F.cross_entropy(logits_all[t], labels)\n                    for t in range(T - 1)) / (T - 1)\n        l_aux = lam_aux * l_aux\n    else:\n        l_aux = torch.tensor(0.0, device=labels.device)\n\n    # L_exit\n    l_exit = torch.tensor(0.0, device=labels.device)\n    for t, lg in enumerate(logits_all):\n        max_prob = F.softmax(lg, dim=-1).max(dim=-1).values\n        weight_t = (T - t) / T\n        l_exit = l_exit + weight_t * (1.0 - max_prob).mean()\n    l_exit = lam_exit * l_exit / T\n\n    # L_fr\n    if hasattr(model, \"mean_firing_rate\"):\n        r = model.mean_firing_rate()\n        l_fr = lam_fr * (torch.tensor(r, dtype=torch.float32, device=labels.device)\n                         if not isinstance(r, torch.Tensor) else r.to(labels.device))\n    else:\n        l_fr = torch.tensor(0.0, device=labels.device)\n\n    total = l_ce + l_aux + l_exit + l_fr\n    breakdown = {\n        \"loss_ce\":    l_ce.item(),\n        \"loss_aux\":   l_aux.item(),\n        \"loss_exit\":  l_exit.item(),\n        \"loss_fr\":    l_fr.item() if isinstance(l_fr, torch.Tensor) else float(l_fr),\n        \"loss_total\": total.item(),\n    }\n    return total, breakdown\n"
FILES["training/train_active.py"] = "import torch\nimport torch.nn.functional as F\nimport time\nimport math\n\nfrom data.slicing import slice_fps_hierarchical_batch\nfrom training.loss_active import active_loss\nfrom training.metrics import accuracy\nfrom models.slice_selection_policy import compute_geometry_descriptors\n\n\ndef gumbel_tau(epoch, tau_0=1.0, tau_min=0.1, anneal_rate=0.05):\n    return max(tau_min, tau_0 * math.exp(-anneal_rate * epoch))\n\n\ndef prepare_fps_slices_and_geo(pts, T):\n    B, N, _ = pts.shape\n    pts_slices  = slice_fps_hierarchical_batch(pts, T=T)   # [B, T, N//T, 3]\n    fps_anchors = pts_slices.mean(dim=2)                   # [B, T, 3]\n    diffs       = pts.unsqueeze(2) - fps_anchors.unsqueeze(1)\n    dists       = (diffs ** 2).sum(-1)\n    assignments = dists.argmin(dim=-1)\n    geo = compute_geometry_descriptors(pts, fps_anchors, assignments)\n    return pts_slices, geo, fps_anchors, assignments\n\n\ndef train_active_epoch(model, dataloader, optimizer, device, epoch,\n                       num_slices=16, lam_aux=0.3, lam_exit=0.1, lam_fr=0.05,\n                       tau_0=1.0, tau_min=0.1, anneal_rate=0.05, verbose_every=20):\n    model.train()\n    tau = gumbel_tau(epoch, tau_0, tau_min, anneal_rate)\n    if hasattr(model, \"set_gumbel_tau\"):\n        model.set_gumbel_tau(tau)\n\n    total_ce = total_aux = total_exit = total_fr = total_tot = 0.0\n    total_acc = total_ent = 0.0\n    count = 0\n    start = time.time()\n\n    for batch_idx, (pts, labels) in enumerate(dataloader):\n        pts    = pts.to(device)\n        labels = labels.to(device)\n        B      = pts.size(0)\n\n        pts_slices, geo, _, _ = prepare_fps_slices_and_geo(pts, T=num_slices)\n        logits_final, logits_all = model.forward_active_train(pts_slices, geo)\n\n        loss, breakdown = active_loss(\n            logits_final, logits_all, labels, model,\n            lam_aux=lam_aux, lam_exit=lam_exit, lam_fr=lam_fr)\n\n        optimizer.zero_grad()\n        loss.backward()\n        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n        optimizer.step()\n\n        with torch.no_grad():\n            _mem = model._get_membrane()\n            if _mem is None:\n                _mem = torch.zeros(B, model.temporal_dim, device=device)\n            scores = model.ssp(_mem, geo, visited_mask=None)\n            probs  = torch.softmax(scores, dim=-1)\n            ent    = -(probs * (probs + 1e-9).log()).sum(dim=-1).mean()\n\n        total_ce   += breakdown[\"loss_ce\"]\n        total_aux  += breakdown[\"loss_aux\"]\n        total_exit += breakdown[\"loss_exit\"]\n        total_fr   += breakdown[\"loss_fr\"]\n        total_tot  += breakdown[\"loss_total\"]\n        total_acc  += accuracy(logits_final, labels)\n        total_ent  += ent.item()\n        count += 1\n\n        if (batch_idx + 1) % verbose_every == 0:\n            elapsed = time.time() - start\n            lr = optimizer.param_groups[0][\"lr\"]\n            print(f\"  [{batch_idx+1}/{len(dataloader)}] \"\n                  f\"CE={breakdown['loss_ce']:.4f} \"\n                  f\"Aux={breakdown['loss_aux']:.4f} \"\n                  f\"Exit={breakdown['loss_exit']:.4f} \"\n                  f\"FR={breakdown['loss_fr']:.4f} \"\n                  f\"Acc={total_acc/count:.3f} \"\n                  f\"Ent={total_ent/count:.3f} \"\n                  f\"tau={tau:.3f} LR={lr:.6f} {elapsed:.0f}s\")\n\n    n = max(count, 1)\n    return {\n        \"loss_ce\": total_ce/n, \"loss_aux\": total_aux/n,\n        \"loss_exit\": total_exit/n, \"loss_fr\": total_fr/n,\n        \"loss_total\": total_tot/n, \"acc_final\": total_acc/n,\n        \"policy_entropy\": total_ent/n, \"gumbel_tau\": tau,\n    }\n\n\ndef validate_active(model, dataloader, device, num_slices=16, threshold=0.7):\n    model.eval()\n    correct = total = 0\n    total_exit = total_fr = count = 0.0\n\n    with torch.no_grad():\n        for pts, labels in dataloader:\n            pts    = pts.to(device)\n            labels = labels.to(device)\n            B      = pts.size(0)\n            pts_slices, geo, _, _ = prepare_fps_slices_and_geo(pts, T=num_slices)\n            for b in range(B):\n                pts_b = pts_slices[b].unsqueeze(0)\n                geo_b = geo[b].unsqueeze(0)\n                lbl_b = labels[b].unsqueeze(0)\n                logits, exit_step, _ = model.forward_active_infer(pts_b, geo_b, threshold=threshold)\n                pred = logits.argmax(dim=-1)\n                correct += (pred == lbl_b).sum().item()\n                total   += 1\n                total_exit += exit_step\n            fr = model.mean_firing_rate()\n            total_fr += fr if isinstance(fr, float) else fr.item()\n            count    += 1\n\n    n = max(total, 1)\n    mean_fr    = total_fr / max(count, 1)\n    mean_exit  = total_exit / n\n    energy_r   = mean_fr * 0.274 * (mean_exit / num_slices)\n    return {\n        \"acc\":          correct / n,\n        \"mean_exit\":    mean_exit,\n        \"mean_fr\":      mean_fr,\n        \"energy_ratio\": energy_r,\n        \"savings\":      1.0 / max(energy_r, 1e-9),\n    }\n"
FILES["inference/__init__.py"] = ""
FILES["inference/active_inference.py"] = "import torch\nimport torch.nn.functional as F\nimport numpy as np\nfrom collections import defaultdict\n\nE_AC  = 2.3e-3\nE_MAC = 8.4e-3\nEFFICIENCY_RATIO = E_AC / E_MAC\n\n\ndef energy_ratio(firing_rate, exit_fraction):\n    return firing_rate * EFFICIENCY_RATIO * exit_fraction\n\n\ndef pareto_curve(model, dataset, device, num_slices=16, thresholds=None, prepare_fn=None):\n    if thresholds is None:\n        thresholds = [round(i * 0.05, 2) for i in range(21)]\n    if prepare_fn is None:\n        from training.train_active import prepare_fps_slices_and_geo\n        prepare_fn = prepare_fps_slices_and_geo\n\n    curve = []\n    for theta in thresholds:\n        correct = total = exit_sum = 0\n        fr_sum  = 0.0\n        model.eval()\n        with torch.no_grad():\n            for pts, label in dataset:\n                if pts.dim() == 2:\n                    pts = pts.unsqueeze(0)\n                pts = pts.to(device)\n                label_idx = label if isinstance(label, int) else label.item()\n                pts_slices, geo, _, _ = prepare_fn(pts, T=num_slices)\n                logits, exit_step, _ = model.forward_active_infer(pts_slices, geo, threshold=theta)\n                pred = logits.argmax(-1).item()\n                correct  += int(pred == label_idx)\n                total    += 1\n                exit_sum += exit_step\n                fr_sum   += model.mean_firing_rate()\n        n = max(total, 1)\n        mean_fr   = fr_sum / n\n        mean_exit = exit_sum / n\n        e_r = energy_ratio(mean_fr, mean_exit / num_slices)\n        m = {\"threshold\": theta, \"accuracy\": correct/n, \"mean_exit\": mean_exit,\n             \"mean_fr\": mean_fr, \"energy_ratio\": e_r, \"savings\": 1.0/max(e_r, 1e-9)}\n        curve.append(m)\n        print(f\"  theta={theta:.2f}  acc={m['accuracy']:.4f}  \"\n              f\"mean_exit={mean_exit:.2f}/{num_slices}  \"\n              f\"energy={e_r:.4f}  savings={m['savings']:.1f}x\")\n    curve.sort(key=lambda x: x[\"energy_ratio\"])\n    return curve\n"
FILES["plots_active.py"] = "import json\nimport numpy as np\nimport matplotlib\nmatplotlib.use(\"Agg\")\nimport matplotlib.pyplot as plt\nimport matplotlib.patches as mpatches\nimport os\n\nCOLORS = {\n    \"asp\":       \"#E6550D\",\n    \"ours_full\": \"#FD8D3C\",\n    \"ours_knn\":  \"#FDBE85\",\n    \"spt\":       \"#3182BD\",\n    \"spm\":       \"#6BAED6\",\n}\n\n\ndef plot_pareto(asp_curve, fixed_baselines, save_path=\"results/active/fig1_pareto.png\"):\n    fig, ax = plt.subplots(figsize=(8, 6))\n    energies = [p[\"energy_ratio\"] for p in asp_curve]\n    accs     = [p[\"accuracy\"] * 100 for p in asp_curve]\n    ax.plot(energies, accs, \"o-\", color=COLORS[\"asp\"], linewidth=2.5,\n            markersize=5, label=\"ASP (adaptive)\")\n    baseline_styles = {\n        \"ours_full\": (\"s\", COLORS[\"ours_full\"], \"ours_full (8.4x)\"),\n        \"ours_knn\":  (\"^\", COLORS[\"ours_knn\"],  \"ours_knn (11.1x)\"),\n        \"spt\":       (\"D\", COLORS[\"spt\"],        \"SPT (6.4x)\"),\n        \"spm\":       (\"P\", COLORS[\"spm\"],        \"SPM (3.5x)\"),\n    }\n    for name, meta in fixed_baselines.items():\n        if name in baseline_styles:\n            marker, color, label = baseline_styles[name]\n            ax.scatter(meta[\"energy_ratio\"], meta[\"accuracy\"]*100,\n                       marker=marker, color=color, s=100, zorder=6,\n                       label=label, edgecolors=\"black\", linewidths=0.7)\n    ax.set_xlabel(\"E_SNN / E_ANN\", fontsize=12)\n    ax.set_ylabel(\"Accuracy (%)\", fontsize=12)\n    ax.set_title(\"Energy-Accuracy Pareto Frontier (ModelNet10)\", fontsize=13)\n    ax.legend(loc=\"lower right\", fontsize=9)\n    ax.grid(True, alpha=0.3)\n    ax.set_xlim(left=0)\n    plt.tight_layout()\n    os.makedirs(os.path.dirname(save_path), exist_ok=True)\n    plt.savefig(save_path, dpi=150)\n    plt.close()\n    print(f\"Saved -> {save_path}\")\n\n\ndef plot_exit_distribution(exit_steps, num_slices=16, save_path=\"results/active/fig2_exit.png\"):\n    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))\n    bins = range(1, num_slices + 2)\n    ax1.hist(exit_steps, bins=bins, align=\"left\", rwidth=0.8,\n             color=COLORS[\"asp\"], alpha=0.8, edgecolor=\"black\", linewidth=0.5)\n    ax1.axvline(x=np.mean(exit_steps), color=\"black\", linestyle=\"--\",\n                linewidth=1.5, label=f\"Mean = {np.mean(exit_steps):.1f}\")\n    ax1.set_xlabel(\"Exit Timestep\"); ax1.set_ylabel(\"Samples\")\n    ax1.set_title(\"Exit Timestep Distribution\"); ax1.legend()\n    exits_sorted = np.sort(exit_steps)\n    p = np.arange(1, len(exits_sorted)+1) / len(exits_sorted)\n    ax2.plot(exits_sorted, p*100, color=COLORS[\"asp\"], linewidth=2.5)\n    ax2.set_xlabel(\"Exit Timestep\"); ax2.set_ylabel(\"Cumulative %\")\n    ax2.set_title(\"Exit CDF\"); ax2.grid(True, alpha=0.3)\n    plt.tight_layout()\n    os.makedirs(os.path.dirname(save_path), exist_ok=True)\n    plt.savefig(save_path, dpi=150); plt.close()\n    print(f\"Saved -> {save_path}\")\n\n\ndef plot_training_history(history, save_path=\"results/active/fig3_history.png\"):\n    epochs = [h[\"epoch\"] for h in history]\n    acc    = [h.get(\"acc_final\", 0)*100 for h in history]\n    loss   = [h.get(\"loss_total\", 0) for h in history]\n    tau    = [h.get(\"gumbel_tau\", 1.0) for h in history]\n    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))\n    ax1.plot(epochs, acc, color=COLORS[\"asp\"], linewidth=2)\n    ax1.set_xlabel(\"Epoch\"); ax1.set_ylabel(\"Train Accuracy (%)\")\n    ax1.set_title(\"Training Accuracy\"); ax1.grid(True, alpha=0.3)\n    ax2.plot(epochs, loss, color=COLORS[\"ours_full\"], linewidth=2, label=\"Total Loss\")\n    ax2_t = ax2.twinx()\n    ax2_t.plot(epochs, tau, color=\"grey\", linestyle=\"--\", linewidth=1.5, label=\"Gumbel tau\")\n    ax2.set_xlabel(\"Epoch\"); ax2.set_ylabel(\"Loss\")\n    ax2_t.set_ylabel(\"Gumbel tau\")\n    ax2.set_title(\"Loss + Gumbel Annealing\"); ax2.grid(True, alpha=0.3)\n    plt.tight_layout()\n    os.makedirs(os.path.dirname(save_path), exist_ok=True)\n    plt.savefig(save_path, dpi=150); plt.close()\n    print(f\"Saved -> {save_path}\")\n"

for rel_path, content in FILES.items():
    full_path = os.path.join(ROOT, rel_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, 'w') as f:
        f.write(content)

print(f'Written {len(FILES)} files to {ROOT}')
for k in sorted(FILES.keys()):
    print(' ', k)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 3 — Apply patches                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# ── Patch: fix in-place visited_mask update (causes autograd RuntimeError) ────
# visited_mask[b, idx] = True mutates a tensor that autograd already saved a
# reference to from the preceding masked_fill, corrupting the backward pass.
# Fix: replace with a non-in-place bitwise-OR update that creates a new tensor.
import os as _os

_path = _os.path.join(ROOT, 'models/active_snn.py')
with open(_path) as _f:
    _src = _f.read()

_old = (
    "            selected_idx = scores.masked_fill(visited_mask, float(\"-inf\")).argmax(dim=-1)\n"
    "            for b in range(B):\n"
    "                visited_mask[b, selected_idx[b]] = True\n"
)
_new = (
    "            selected_idx = scores.masked_fill(visited_mask, float(\"-inf\")).argmax(dim=-1)\n"
    "            # non-in-place update — avoids corrupting autograd saved tensors\n"
    "            visited_mask = visited_mask | F.one_hot(selected_idx, num_classes=T).bool()\n"
)

if _old in _src:
    _src = _src.replace(_old, _new)
    with open(_path, 'w') as _f:
        _f.write(_src)
    print('[OK] active_snn.py: fixed in-place visited_mask update')
else:
    print('[SKIP] Pattern not found (already patched)')


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 4 — Download ModelNet40                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
MN40_DIR = '/content/ModelNet40'

if not os.path.isdir(MN40_DIR):
    print('Downloading ModelNet40 via kagglehub (~1.9 GB)...')
    import kagglehub, shutil, glob
    p = kagglehub.dataset_download('balraj98/modelnet40-princeton-3d-object-dataset')
    print(f'  Downloaded to: {p}')
    mn40_src = None
    for root, dirs, _ in os.walk(p):
        if 'ModelNet40' in dirs:
            mn40_src = os.path.join(root, 'ModelNet40')
            break
    if mn40_src:
        shutil.copytree(mn40_src, MN40_DIR)
        print(f'  Copied: {mn40_src} -> {MN40_DIR}')
    else:
        zips = glob.glob(os.path.join(p, '*.zip'))
        if zips:
            os.system(f'unzip -q "{zips[0]}" -d /content/')
        else:
            # Princeton fallback
            ret = os.system('wget -q --timeout=60 https://modelnet.cs.princeton.edu/ModelNet40.zip -O /tmp/MN40.zip')
            if ret == 0 and os.path.getsize('/tmp/MN40.zip') > 1_000_000:
                os.system('unzip -q /tmp/MN40.zip -d /content/')
            else:
                raise RuntimeError(f'ModelNet40 not found in Kaggle download at {p}')
    if not os.path.isdir(MN40_DIR):
        raise RuntimeError(f'{MN40_DIR} still missing after download.')
    print('Done.')
else:
    print('ModelNet40 already present.')

classes = sorted([d for d in os.listdir(MN40_DIR)
                  if os.path.isdir(os.path.join(MN40_DIR, d))])
print(f'Classes ({len(classes)}):', classes[:6], '...')

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 5 — Parse args & configure                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs',     type=int, default=30,   help='Training epochs (30=quick, 150=SOTA)')
parser.add_argument('--batch_size', type=int, default=16,   help='Batch size')
parser.add_argument('--lr',         type=float, default=1e-3)
parser.add_argument('--num_slices', type=int, default=16)
parser.add_argument('--threshold',  type=float, default=0.7, help='Early-exit margin threshold')
parser.add_argument('--num_points', type=int, default=1024)
parser.add_argument('--out_dir',    type=str, default='/content/asp_results_mn40')
args, _ = parser.parse_known_args()  # ignore Jupyter kernel flags (-f kernel-xxx.json)

os.makedirs(args.out_dir, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'\nConfig: epochs={args.epochs} bs={args.batch_size} lr={args.lr} T={args.num_slices}')

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 6 — Dataset                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
sys.path.insert(0, '/content/asp')
from data.modelnet import ModelNetDataset
from torch.utils.data import DataLoader

train_ds = ModelNetDataset(root=MN40_DIR, split='train', num_points=args.num_points)
val_ds   = ModelNetDataset(root=MN40_DIR, split='test',  num_points=args.num_points)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)
print(f'MN40 — Train: {len(train_ds)}  Val: {len(val_ds)}  Batches/epoch: {len(train_loader)}')

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 7 — Model                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
from models.active_snn import ActiveSNN
from training.optimizers import build_optimizer
from training.train_active import train_active_epoch, validate_active

model = ActiveSNN(
    point_dims=[128, 256, 512],
    temporal_dim=512,
    num_classes=40,
    knn_k=16,
    d_ssp=64,
).to(device)

params = model.param_count()
print(f'Params — backbone: {params["backbone"]:,}  total: {params["total"]:,}')

optimizer = build_optimizer(model, lr=args.lr, weight_decay=1e-4)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 8 — Mount Drive & auto-resume                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
import json as _json

# Drive must be mounted from the Colab cell before running this script.
# The notebook mounts it; we just use the path directly here.
DRIVE_DIR        = '/content/drive/MyDrive/asp_mn40_checkpoints'
drive_epoch_ckpt = os.path.join(DRIVE_DIR, 'epoch_latest.pth')
drive_history    = os.path.join(DRIVE_DIR, 'history_mn40.json')
os.makedirs(DRIVE_DIR, exist_ok=True)

start_epoch = 0
best_acc    = 0.0
history     = []
best_ckpt   = os.path.join(args.out_dir, 'best_model_mn40.pth')

if os.path.isfile(drive_epoch_ckpt):
    print(f'Resuming from Drive checkpoint: {drive_epoch_ckpt}')
    ckpt = torch.load(drive_epoch_ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    start_epoch = ckpt['epoch'] + 1
    best_acc    = ckpt.get('best_acc', 0.0)
    if os.path.isfile(drive_history):
        with open(drive_history) as fh:
            history = _json.load(fh)
    print(f'  Resumed from epoch {ckpt["epoch"]}  best_acc={best_acc:.4f}')
else:
    print('No Drive checkpoint found — starting from scratch.')

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
# Fast-forward scheduler state to match start_epoch
for _ in range(start_epoch):
    scheduler.step()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 9 — Training loop                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print('\n' + '='*60)
print(f'Starting MN40 training — {args.epochs} epochs (from epoch {start_epoch})')
print('='*60)

for epoch in range(start_epoch, args.epochs):
    print(f'\n--- Epoch {epoch}/{args.epochs - 1} ---')
    train_m = train_active_epoch(
        model, train_loader, optimizer, device, epoch=epoch,
        num_slices=args.num_slices, verbose_every=20,
        anneal_rate=0.1)   # faster tau decay: reaches 0.1 by epoch ~23 (vs ~46 at 0.05)
    scheduler.step()

    val_m = {}
    if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
        val_m = validate_active(model, val_loader, device,
                                num_slices=args.num_slices,
                                threshold=args.threshold)
        val_acc = val_m['acc']
        print(f'[Val] Acc={val_acc:.4f}  '
              f'MeanExit={val_m["mean_exit"]:.2f}/{args.num_slices}  '
              f'FR={val_m["mean_fr"]:.3f}  '
              f'EnergyRatio={val_m["energy_ratio"]:.4f}  '
              f'Savings={val_m["savings"]:.1f}x')
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_acc': val_acc, 'args': vars(args)}, best_ckpt)
            print(f'  *** New best: {val_acc:.4f} -> {best_ckpt}')

    history.append({'epoch': epoch, **train_m, **val_m})
    with open(os.path.join(args.out_dir, 'history_mn40.json'), 'w') as fh:
        _json.dump(history, fh, indent=2)

    # ---- Save epoch checkpoint to Drive after every epoch ----
    torch.save({
        'epoch':          epoch,
        'model_state':    model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_acc':       best_acc,
        'args':           vars(args),
    }, drive_epoch_ckpt)
    with open(drive_history, 'w') as fh:
        _json.dump(history, fh, indent=2)
    print(f'  [Drive] epoch_latest.pth saved (epoch {epoch})')

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 10 — Verdict                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print('\n' + '='*60)
print(f'MN40 Best Val Acc: {best_acc:.4f}')
print(f'Checkpoint:        {best_ckpt}')
print('='*60)
# MN40 (40-class) thresholds — much lower than MN10 (10-class)
# SOTA on MN40 is ~93%; 30-epoch quick run typically lands 60-72%
if best_acc >= 0.65:
    print('VERDICT: Strong. Scale to 150 epochs on A100/H100 for SOTA.')
elif best_acc >= 0.55:
    print('VERDICT: Moderate. Try 150 epochs or tune lr/num_slices.')
else:
    print('VERDICT: Low — check dataset path and model config.')
print('\nDone.')
