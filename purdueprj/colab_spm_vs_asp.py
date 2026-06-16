"""
colab_spm_vs_asp.py - Official-Like SPM + ASP (Colab Edition)
=============================================================
Self-contained Colab script. All checkpoints saved to Google Drive.

Setup:
  from google.colab import files
  files.upload()
  !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
  !python colab_spm_vs_asp.py
"""

import json, math, os, random, shutil, subprocess, sys, time, warnings
warnings.filterwarnings("ignore")

ON_COLAB = "google.colab" in sys.modules or os.path.isdir("/content")
DRIVE_ROOT = None
if ON_COLAB:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        DRIVE_ROOT = "/content/drive/MyDrive/asp_spm_official"
        print(f"[Drive] Mounted -> {DRIVE_ROOT}")
    except Exception as e:
        print(f"[Drive] Mount failed ({e})")
        DRIVE_ROOT = "/content/asp_spm_official"
    os.makedirs(DRIVE_ROOT, exist_ok=True)

for pkg in ["trimesh", "kagglehub", "matplotlib"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-q", pkg], check=True)

import kagglehub, matplotlib, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, trimesh
from torch.utils.data import DataLoader, Dataset
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("PyTorch:", torch.__version__)
print("CUDA   :", torch.cuda.is_available())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ON_GPU = DEVICE == "cuda"
ON_KAGGLE = os.path.isdir("/kaggle/working")

if ON_COLAB and DRIVE_ROOT:
    WORK = DRIVE_ROOT
elif ON_KAGGLE:
    WORK = "/kaggle/working"
else:
    WORK = "/tmp/asp_official_like"
os.makedirs(WORK, exist_ok=True)

def env_int(n, d): return int(os.environ.get(n, str(d)))
def env_float(n, d): return float(os.environ.get(n, str(d)))

if ON_GPU:
    print("GPU    :", torch.cuda.get_device_name(0))
    EPOCHS=env_int("EPOCHS",300); BATCH=env_int("BATCH",16); NUM_POINTS=env_int("NUM_POINTS",1024)
    TIMESTEP=env_int("TIMESTEP",2); TRANS_DIM=env_int("TRANS_DIM",384); DEPTH=env_int("DEPTH",12)
    NUM_GROUP=env_int("NUM_GROUP",128); GROUP_SIZE=env_int("GROUP_SIZE",32)
    ASP_STEPS=env_int("ASP_STEPS",4); DROP_PATH=env_float("DROP_PATH",0.3)
    GRAD_ACCUM=env_int("GRAD_ACCUM",4); NUM_WORKERS=env_int("NUM_WORKERS",2)
    N_VOTE=env_int("N_VOTE",5)
    DATASET_NAMES=os.environ.get("DATASETS","ModelNet10,ModelNet40").split(",")
else:
    print("WARNING: CPU demo mode.")
    EPOCHS=env_int("EPOCHS",20); BATCH=env_int("BATCH",4); NUM_POINTS=env_int("NUM_POINTS",256)
    TIMESTEP=env_int("TIMESTEP",2); TRANS_DIM=env_int("TRANS_DIM",128); DEPTH=env_int("DEPTH",4)
    NUM_GROUP=env_int("NUM_GROUP",32); GROUP_SIZE=env_int("GROUP_SIZE",16)
    ASP_STEPS=env_int("ASP_STEPS",4); DROP_PATH=env_float("DROP_PATH",0.1)
    GRAD_ACCUM=env_int("GRAD_ACCUM",1); NUM_WORKERS=env_int("NUM_WORKERS",0)
    N_VOTE=env_int("N_VOTE",1)
    DATASET_NAMES=os.environ.get("DATASETS","ModelNet10").split(",")

EXPAND=env_float("EXPAND",1.1); LR=env_float("LR",1e-3)
WEIGHT_DECAY=env_float("WEIGHT_DECAY",0.1)
WARMUP_EP=env_int("WARMUP_EP",30 if ON_GPU else 3)
LABEL_SMOOTH=env_float("LABEL_SMOOTH",0.2)
EXIT_THR=env_float("EXIT_THR",0.45)
CHECKPOINT_EVERY=env_int("CHECKPOINT_EVERY",5)
RESUME=os.environ.get("RESUME_CHECKPOINTS","1").lower() not in ("0","false","no")
CLEAN_CKPTS=os.environ.get("CLEAN_CHECKPOINTS","0").lower() in ("1","true","yes")

CKPT_DIR=os.path.join(WORK,"official_like_ckpts")
EPOCH_DIR=os.path.join(CKPT_DIR,"epochs")
LOG_DIR=os.path.join(WORK,"logs")
for d in [CKPT_DIR,EPOCH_DIR,LOG_DIR]: os.makedirs(d,exist_ok=True)

if CLEAN_CKPTS:
    print("[CLEANUP] Removing checkpoint files...")
    for fn in os.listdir(CKPT_DIR):
        if fn.endswith((".pt",".pth")):
            try: os.remove(os.path.join(CKPT_DIR,fn)); print(f"  Removed: {fn}")
            except: pass

assert NUM_GROUP % ASP_STEPS == 0
print("[Config]")
print(f"  epochs={EPOCHS} batch={BATCH} points={NUM_POINTS} dim={TRANS_DIM} depth={DEPTH}")
print(f"  groups={NUM_GROUP} group_size={GROUP_SIZE} asp_steps={ASP_STEPS}")
print(f"  ckpts={CKPT_DIR}  epochs={EPOCH_DIR}  logs={LOG_DIR}")
print(f"  resume={RESUME}  drive={DRIVE_ROOT}")


# ── Dataset ───────────────────────────────────────────────────────────────────

def _download(name, slug):
    folder = os.path.join(WORK if ON_KAGGLE else "/tmp", name)
    if os.path.isdir(folder) and len(os.listdir(folder)) > 0:
        print(f"  {name}: already at {folder}"); return folder
    print(f"  Downloading {name} ...")
    path = kagglehub.dataset_download(slug)
    for root, dirs, _ in os.walk(path):
        if name in dirs:
            shutil.copytree(os.path.join(root, name), folder)
            print(f"  {name} -> {folder}"); return folder
        if len([d for d in dirs if os.path.isdir(os.path.join(root,d,"train"))]) >= 5:
            shutil.copytree(root, folder); print(f"  {name} -> {folder}"); return folder
    return path

MN10_DIR = MN40_DIR = None

def _augment(pts):
    n = pts.shape[0]
    keep = max(int(n * random.uniform(0.875, 1.0)), 1)
    idx = np.random.choice(n, keep, replace=False)
    pts2 = pts[idx]
    if keep < n:
        pad = np.random.choice(keep, n - keep, replace=True)
        pts2 = np.vstack([pts2, pts2[pad]])
    pts2 = pts2 * np.random.uniform(0.8, 1.25) + np.random.uniform(-0.1, 0.1, (1, 3))
    theta = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    rz = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
    pts2 = pts2 @ rz.T
    pts2 += np.clip(np.random.randn(*pts2.shape).astype(np.float32)*0.01, -0.05, 0.05)
    return pts2.astype(np.float32)

class ModelNetDataset(Dataset):
    def __init__(self, root, num_points=1024, split="train"):
        self.num_points = num_points; self.split = split
        self.files = self._scan(root)
        print(f"  [{split}] Loading {len(self.files)} files ...")
        self.data, self.labels = self._load_all()
        print(f"  [{split}] Loaded. Shape: {self.data.shape}")

    def _scan(self, root):
        items = []
        classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))])
        for cls in classes:
            p = os.path.join(root, cls, self.split)
            if not os.path.isdir(p): continue
            lbl = classes.index(cls)
            for f in os.listdir(p):
                if f.lower().endswith((".npy",".txt",".off")):
                    items.append((os.path.join(p,f), lbl))
        return items

    def _load_pts(self, path):
        if path.endswith(".npy"): return np.load(path).astype(np.float32)[:,:3]
        if path.endswith(".txt"): return np.loadtxt(path,delimiter=",").astype(np.float32)[:,:3]
        mesh = trimesh.load(path, force="mesh")
        pts, _ = trimesh.sample.sample_surface(mesh, self.num_points)
        return pts.astype(np.float32)

    def _load_all(self):
        all_pts, all_lbl = [], []
        for path, lbl in self.files:
            try:
                pts = self._load_pts(path); n = pts.shape[0]
                if n >= self.num_points: pts = pts[np.random.choice(n,self.num_points,replace=False)]
                else:
                    pad = np.random.choice(n, self.num_points-n, replace=True)
                    pts = np.vstack([pts, pts[pad]])
                all_pts.append(pts); all_lbl.append(lbl)
            except: pass
        return np.asarray(all_pts,dtype=np.float32), np.asarray(all_lbl,dtype=np.int64)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        pts = self.data[idx].copy()
        pts -= pts.mean(axis=0); pts /= np.max(np.linalg.norm(pts,axis=1))+1e-8
        if self.split == "train": pts = _augment(pts)
        np.random.shuffle(pts)
        return torch.tensor(pts,dtype=torch.float32), torch.tensor(self.labels[idx],dtype=torch.long)


# ── Point cloud grouping: FPS + KNN ──────────────────────────────────────────

def index_points(points, idx):
    b = points.shape[0]; view_shape = list(idx.shape)
    view_shape[1:] = [1]*(len(view_shape)-1)
    repeat_shape = list(idx.shape); repeat_shape[0] = 1
    bi = torch.arange(b,dtype=torch.long,device=points.device).view(view_shape).repeat(repeat_shape)
    return points[bi, idx]

def farthest_point_sample_batched(xyz, npoint):
    b,n,_ = xyz.shape; npoint = min(npoint, n)
    centroids = torch.zeros(b,npoint,dtype=torch.long,device=xyz.device)
    distance = torch.full((b,n),1e10,device=xyz.device)
    farthest = torch.randint(0,n,(b,),dtype=torch.long,device=xyz.device)
    batch_idx = torch.arange(b,dtype=torch.long,device=xyz.device)
    for i in range(npoint):
        centroids[:,i] = farthest
        centroid = xyz[batch_idx,farthest,:].view(b,1,3)
        dist = ((xyz-centroid)**2).sum(-1)
        distance = torch.minimum(distance,dist)
        farthest = distance.max(-1).indices
    return centroids

class OfficialLikeGroup(nn.Module):
    def __init__(self, num_group, group_size, expand=1.1, timestep=2):
        super().__init__()
        self.num_group=num_group; self.group_size=group_size
        self.expand=expand; self.timestep=timestep

    def _moving_centers(self, pts):
        b,n,_ = pts.shape
        step_f = int((self.expand-1.0)*self.num_group/self.timestep*2)
        step_b = int((self.expand-1.0)*self.num_group)
        total = self.num_group + (step_f+step_b)*(self.timestep-1)
        total = min(max(total,self.num_group), n)
        center_idx = farthest_point_sample_batched(pts.contiguous(), total)
        pool = index_points(pts, center_idx)
        if total < self.num_group + (step_f+step_b)*(self.timestep-1):
            repeat = math.ceil((self.num_group+(step_f+step_b)*(self.timestep-1))/total)
            pool = pool.repeat(1,repeat,1)
        centers = []
        for i in range(self.timestep):
            first = pool[:,i*step_f:i*step_f+(self.num_group-step_b)]
            start = (i-1)*step_b+self.num_group+(self.timestep-1)*step_f
            end = i*step_b+self.num_group+(self.timestep-1)*step_f
            second = pool[:,start:end]
            cur = torch.cat([first,second],dim=1)
            if cur.shape[1] < self.num_group:
                pad = cur[:,-1:].repeat(1,self.num_group-cur.shape[1],1)
                cur = torch.cat([cur,pad],dim=1)
            centers.append(cur[:,:self.num_group])
        return torch.stack(centers, dim=0)

    def forward(self, pts):
        b,n,_ = pts.shape
        centers = self._moving_centers(pts)
        fc = centers.reshape(self.timestep*b,self.num_group,3)
        fp = pts.unsqueeze(0).expand(self.timestep,-1,-1,-1).reshape(self.timestep*b,n,3)
        k = min(self.group_size, n)
        dist = torch.cdist(fc, fp)
        idx = dist.topk(k,dim=-1,largest=False).indices
        grouped = index_points(fp, idx).reshape(self.timestep,b,self.num_group,k,3)
        grouped = grouped - centers.unsqueeze(3)
        return grouped.contiguous(), centers.contiguous()


# ── Spiking modules ──────────────────────────────────────────────────────────

class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): ctx.save_for_backward(x); return (x>0).float()
    @staticmethod
    def backward(ctx, g): (x,)=ctx.saved_tensors; return g/(1.0+x.abs())**2

spike_fn = SurrogateSpike.apply

class SpikeAct(nn.Module):
    def __init__(self, vth=0.5):
        super().__init__(); self.vth=vth
        self.register_buffer("spike_sum",torch.tensor(0.0))
        self.register_buffer("elem_count",torch.tensor(0.0))
    def forward(self, x):
        y = spike_fn(x-self.vth)
        self.spike_sum = self.spike_sum+y.detach().sum()
        self.elem_count = self.elem_count+torch.tensor(y.numel(),dtype=torch.float32,device=y.device)
        return y
    def rate(self): return (self.spike_sum/self.elem_count).item() if self.elem_count.item()>0 else 0.0

class DropPath(nn.Module):
    def __init__(self, p=0.0): super().__init__(); self.p=p
    def forward(self, x):
        if not self.training or self.p==0: return x
        keep=1.0-self.p; shape=(x.shape[0],)+(1,)*(x.ndim-1)
        return x*torch.bernoulli(torch.full(shape,keep,device=x.device))/keep

class TokenBNSpike(nn.Module):
    def __init__(self, dim, vth=0.5):
        super().__init__(); self.bn=nn.BatchNorm1d(dim); self.spike=SpikeAct(vth)
    def forward(self, x):
        b,l,c = x.shape
        return self.spike(self.bn(x.reshape(b*l,c)).reshape(b,l,c))

class OfficialLikeEncoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.spk1=SpikeAct(); self.spk2=SpikeAct(); self.spk3=SpikeAct()
        self.first_conv1=nn.Conv2d(3,128,1); self.first_bn1=nn.BatchNorm2d(128)
        self.first_conv2=nn.Conv2d(128,256,1); self.first_bn2=nn.BatchNorm2d(256)
        self.second_conv1=nn.Conv2d(512,512,1); self.second_bn1=nn.BatchNorm2d(512)
        self.second_conv2=nn.Conv2d(512,encoder_channel,1); self.second_bn2=nn.BatchNorm2d(encoder_channel)

    def forward(self, point_groups):
        t,b,g,k,_ = point_groups.shape
        x = point_groups.flatten(0,1).permute(0,3,1,2).contiguous()
        x = self.spk1(self.first_bn1(self.first_conv1(x)))
        x = self.first_bn2(self.first_conv2(x))
        xg = x.max(dim=3,keepdim=True).values
        x = torch.cat([xg.expand(-1,-1,-1,k),x],dim=1)
        x = self.spk2(x)
        x = self.spk3(self.second_bn1(self.second_conv1(x)))
        x = self.second_bn2(self.second_conv2(x))
        x = x.max(dim=3).values.transpose(1,2).contiguous()
        return x.reshape(t,b,g,-1)

class PosEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(3,128,1),nn.BatchNorm1d(128),SpikeAct(),nn.Conv1d(128,dim,1),nn.BatchNorm1d(dim))
    def forward(self, centers):
        t,b,g,_ = centers.shape
        x = centers.flatten(0,1).permute(0,2,1).contiguous()
        return self.net(x).permute(0,2,1).contiguous().reshape(t,b,g,-1)

class MambaLiteMixer(nn.Module):
    def __init__(self, dim, expand=2):
        super().__init__()
        inner = dim*expand
        self.in_proj=nn.Linear(dim,inner*2); self.dwconv=nn.Conv1d(inner,inner,3,padding=1,groups=inner)
        self.scan_proj=nn.Linear(inner,inner); self.out_proj=nn.Linear(inner,dim)
    def forward(self, x):
        u,gate = self.in_proj(x).chunk(2,dim=-1)
        u = self.dwconv(u.transpose(1,2)).transpose(1,2); u = F.silu(u)
        steps = torch.arange(1,u.shape[1]+1,device=u.device,dtype=u.dtype).view(1,-1,1)
        state = torch.cumsum(u,dim=1)/steps
        u = u+self.scan_proj(state); u = u*torch.sigmoid(gate)
        return self.out_proj(u)

class OfficialLikeBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__(); self.norm_lif=TokenBNSpike(dim); self.mixer=MambaLiteMixer(dim); self.drop_path=DropPath(drop_path)
    def forward(self, hidden_states, residual=None):
        residual = self.drop_path(hidden_states)+residual if residual is not None else hidden_states
        return self.mixer(self.norm_lif(residual)), residual

class OfficialLikeMixerModel(nn.Module):
    def __init__(self, dim, depth, timestep, drop_path=0.3):
        super().__init__(); self.timestep=timestep
        self.layers = nn.ModuleList([OfficialLikeBlock(dim,drop_path=drop_path) for _ in range(depth)])
    def forward(self, tokens, pos):
        t,b,l,c = tokens.shape; x = (tokens+pos).reshape(t*b,l,c); residual=None
        for layer in self.layers: x,residual = layer(x,residual)
        x = x+residual if residual is not None else x
        return x.reshape(t,b,l,c)

class OfficialLikeHead(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(SpikeAct(),nn.Conv1d(dim,256,1),nn.BatchNorm1d(256),SpikeAct(),nn.Conv1d(256,128,1),nn.BatchNorm1d(128),SpikeAct(),nn.Conv1d(128,num_classes,1))
    def forward(self, x):
        t,b,_l,c = x.shape
        pooled = x.mean(dim=2).reshape(t*b,c,1)
        return self.net(pooled).reshape(t,b,-1,1).mean(dim=0).squeeze(-1)

class OfficialLikeSPM(nn.Module):
    def __init__(self, num_classes, dim=384, depth=12, num_group=128, group_size=32, timestep=2, expand=1.1, drop_path=0.3):
        super().__init__()
        self.num_classes=num_classes; self.dim=dim; self.depth=depth
        self.num_group=num_group; self.group_size=group_size; self.timestep=timestep
        self.group_divider=OfficialLikeGroup(num_group,group_size,expand,timestep)
        self.encoder=OfficialLikeEncoder(dim); self.pos_embed=PosEmbed(dim)
        self.blocks=OfficialLikeMixerModel(dim,depth,timestep,drop_path)
        self.drop_out=nn.Dropout(0.0); self.cls_head=OfficialLikeHead(dim,num_classes)

    def encode_groups(self, pts):
        neighborhoods, centers = self.group_divider(pts)
        return self.encoder(neighborhoods), self.pos_embed(centers), centers

    def forward_tokens(self, tokens, pos):
        return self.cls_head(self.blocks(self.drop_out(tokens), pos))

    def forward(self, pts):
        tokens,pos,_ = self.encode_groups(pts); return self.forward_tokens(tokens,pos)

    def get_firing_rates(self):
        return {n: m.rate() for n,m in self.named_modules() if isinstance(m,SpikeAct)}

    def mean_firing_rate(self):
        r=self.get_firing_rates(); return sum(r.values())/max(1,len(r))


# ── ASP wrapper ───────────────────────────────────────────────────────────────

class SliceSelectionPolicy(nn.Module):
    def __init__(self, mem_dim, geo_dim=7, hidden=128):
        super().__init__()
        self.mem_proj=nn.Linear(mem_dim,hidden,bias=False)
        self.geo_proj=nn.Sequential(nn.Linear(geo_dim,hidden),nn.GELU(),nn.Linear(hidden,hidden,bias=False))
        self.scale=math.sqrt(hidden)
    def forward(self, belief, geo, visited_mask=None):
        scores = torch.bmm(self.geo_proj(geo), self.mem_proj(belief).unsqueeze(-1)).squeeze(-1)/self.scale
        if visited_mask is not None: scores = scores.masked_fill(visited_mask.clone(), float("-inf"))
        return scores

class OfficialLikeASP(nn.Module):
    def __init__(self, base_model, asp_steps=4, d_ssp=128):
        super().__init__()
        self.base_model=base_model; self.asp_steps=asp_steps
        self.chunk_size=base_model.num_group//asp_steps
        self.ssp=SliceSelectionPolicy(base_model.dim,geo_dim=7,hidden=d_ssp)
        self.belief_proj=nn.Sequential(nn.Linear(base_model.num_classes,base_model.dim),nn.GELU(),nn.Linear(base_model.dim,base_model.dim))
        self.register_buffer("gumbel_tau",torch.tensor(1.0))

    @property
    def num_classes(self): return self.base_model.num_classes
    def set_gumbel_tau(self, tau): self.gumbel_tau.fill_(tau)
    def get_firing_rates(self): return self.base_model.get_firing_rates()
    def mean_firing_rate(self): return self.base_model.mean_firing_rate()

    def _chunkify(self, tokens, pos, centers, pts):
        t,b,g,c = tokens.shape; s,k = self.asp_steps, self.chunk_size
        tokens_c=tokens.reshape(t,b,s,k,c); pos_c=pos.reshape(t,b,s,k,c)
        centers_b=centers.mean(dim=0).reshape(b,s,k,3)
        chunk_center=centers_b.mean(dim=2); centroid=pts.mean(dim=1,keepdim=True)
        anchor_dist=(chunk_center-centroid).norm(dim=-1,keepdim=True)
        spread=(centers_b-chunk_center.unsqueeze(2)).norm(dim=-1).mean(dim=2,keepdim=True)
        coverage=torch.ones(b,s,1,device=pts.device)
        order=torch.linspace(0,1,s,device=pts.device).view(1,s,1).expand(b,-1,-1)
        geo=torch.cat([chunk_center,anchor_dist,spread,coverage,order],dim=-1)
        return tokens_c, pos_c, geo

    def _gather_chunk(self, chunks, idx):
        t,b,_s,k,c = chunks.shape
        return chunks.gather(dim=2,index=idx.view(1,b,1,1,1).expand(t,b,1,k,c)).squeeze(2)

    def forward_active_train(self, pts):
        tokens,pos,centers = self.base_model.encode_groups(pts)
        tok_c,pos_c,geo = self._chunkify(tokens,pos,centers,pts)
        b=pts.shape[0]; device=pts.device
        visited=torch.zeros(b,self.asp_steps,dtype=torch.bool,device=device)
        belief=torch.zeros(b,self.base_model.dim,device=device)
        sel_tok,sel_pos,logits_all = [],[],[]
        for _ in range(self.asp_steps):
            scores=self.ssp(belief,geo,visited)
            w=F.gumbel_softmax(scores,tau=float(self.gumbel_tau.item()),hard=True)
            idx=w.detach().argmax(dim=-1); visited.scatter_(1,idx.unsqueeze(1),True)
            sel_tok.append((w.view(1,b,self.asp_steps,1,1)*tok_c).sum(dim=2))
            sel_pos.append((w.view(1,b,self.asp_steps,1,1)*pos_c).sum(dim=2))
            logits=self.base_model.forward_tokens(torch.cat(sel_tok,dim=2),torch.cat(sel_pos,dim=2))
            logits_all.append(logits)
            belief=self.belief_proj(logits.detach().softmax(dim=-1))
        return logits_all[-1], logits_all

    @torch.no_grad()
    def forward_active_infer(self, pts, threshold=0.45):
        tokens,pos,centers = self.base_model.encode_groups(pts)
        tok_c,pos_c,geo = self._chunkify(tokens,pos,centers,pts)
        b=pts.shape[0]; device=pts.device
        visited=torch.zeros(b,self.asp_steps,dtype=torch.bool,device=device)
        belief=torch.zeros(b,self.base_model.dim,device=device)
        sel_tok,sel_pos = [],[]
        last_logits=None
        for step in range(self.asp_steps):
            scores=self.ssp(belief,geo,visited); idx=scores.argmax(dim=-1)
            visited.scatter_(1,idx.unsqueeze(1),True)
            sel_tok.append(self._gather_chunk(tok_c,idx))
            sel_pos.append(self._gather_chunk(pos_c,idx))
            logits=self.base_model.forward_tokens(torch.cat(sel_tok,dim=2),torch.cat(sel_pos,dim=2))
            last_logits=logits; belief=self.belief_proj(logits.softmax(dim=-1))
            top2=logits.softmax(dim=-1).topk(2,dim=-1).values
            if (top2[:,0]-top2[:,1]).min().item() > threshold: return logits, step+1
        return last_logits, self.asp_steps


# ── Training utilities ────────────────────────────────────────────────────────

def make_spm(nc):
    return OfficialLikeSPM(nc,dim=TRANS_DIM,depth=DEPTH,num_group=NUM_GROUP,group_size=GROUP_SIZE,timestep=TIMESTEP,expand=EXPAND,drop_path=DROP_PATH).to(DEVICE)

def make_asp(nc): return OfficialLikeASP(make_spm(nc),asp_steps=ASP_STEPS).to(DEVICE)

def smooth_ce(logits, labels): return F.cross_entropy(logits,labels,label_smoothing=LABEL_SMOOTH)

def gumbel_tau(epoch, tau_0=1.0, tau_min=0.1, rate=0.04): return max(tau_min,tau_0*math.exp(-rate*epoch))

def active_loss(logits_final, logits_all, labels, model):
    loss = smooth_ce(logits_final, labels)
    if len(logits_all)>1:
        aux = sum(smooth_ce(lg,labels) for lg in logits_all[:-1])
        loss = loss + 0.1*aux/(len(logits_all)-1)
    el = sum((len(logits_all)-i)/len(logits_all)*(1.0-lg.softmax(-1).max(-1).values).mean() for i,lg in enumerate(logits_all))
    loss = loss + 0.05*el/len(logits_all) + 0.01*model.mean_firing_rate()
    return loss

def make_scheduler(optimizer, epochs, warmup_epochs):
    def lr_lambda(ep):
        if ep < warmup_epochs: return (ep+1)/max(1,warmup_epochs)
        t = (ep-warmup_epochs)/max(1,epochs-warmup_epochs)
        return max(1e-2, 0.5*(1.0+math.cos(math.pi*t)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Checkpoint helpers (atomic save + backup recovery + epoch numbering) ──────

def save_ckpt(path, model, opt, sch, epoch, best, history):
    payload = {"epoch":epoch,"model":model.state_dict(),"optimizer":opt.state_dict(),
               "scheduler":sch.state_dict(),"best":best,"history":history,
               "config":{"dim":TRANS_DIM,"depth":DEPTH,"num_group":NUM_GROUP,"group_size":GROUP_SIZE,"asp_steps":ASP_STEPS},
               "timestamp":time.time()}
    try:
        tmp = path+".tmp"
        torch.save(payload, tmp)
        bak = path+".backup"
        if os.path.exists(path):
            try:
                if os.path.exists(bak): os.remove(bak)
                os.replace(path, bak)
            except: pass
        os.replace(tmp, path)
        if os.path.exists(bak):
            try: os.remove(bak)
            except: pass
    except Exception as e:
        print(f"  [WARN] Save failed: {e}")
        if os.path.exists(path+".tmp"):
            try: os.remove(path+".tmp")
            except: pass

    # Numbered epoch checkpoint to Drive
    if epoch > 0 and (epoch % CHECKPOINT_EVERY == 0 or epoch == EPOCHS):
        base = os.path.basename(path).replace("_latest.pt","")
        ep_path = os.path.join(EPOCH_DIR, f"{base}_ep{epoch:04d}.pt")
        try: torch.save(payload, ep_path); print(f"  [Ckpt] Epoch {epoch} -> {ep_path}")
        except: pass

def load_ckpt(path, model, opt, sch):
    if not RESUME or not os.path.isfile(path): return 0, 0.0, []
    for try_path in [path, path+".backup"]:
        if not os.path.isfile(try_path): continue
        try:
            ck = torch.load(try_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); sch.load_state_dict(ck["scheduler"])
            ep=int(ck["epoch"]); best=float(ck.get("best",0.0)); hist=ck.get("history",[])
            print(f"  [CKPT] Resumed from {os.path.basename(try_path)}, epoch {ep}, best={best*100:.2f}%")
            return ep, best, hist
        except Exception as e:
            print(f"  [CKPT] Failed {os.path.basename(try_path)}: {e}")
    print("  [CKPT] No valid checkpoint. Starting fresh.")
    return 0, 0.0, []


# ── Train/eval loops ─────────────────────────────────────────────────────────

def train_spm_epoch(model, loader, opt):
    model.train(); opt.zero_grad(); tl=ta=n=0
    for step,(pts,labels) in enumerate(loader):
        pts,labels = pts.to(DEVICE),labels.to(DEVICE)
        logits=model(pts); loss=smooth_ce(logits,labels)/GRAD_ACCUM
        if torch.isfinite(loss): loss.backward()
        if (step+1)%GRAD_ACCUM==0 or step+1==len(loader):
            nn.utils.clip_grad_norm_(model.parameters(),10.0); opt.step(); opt.zero_grad()
        b=pts.shape[0]; tl+=loss.item()*GRAD_ACCUM*b; ta+=(logits.argmax(1)==labels).sum().item(); n+=b
    return tl/n, ta/n

def train_asp_epoch(model, loader, opt, epoch):
    model.train(); model.set_gumbel_tau(gumbel_tau(epoch)); opt.zero_grad(); tl=ta=n=0
    for step,(pts,labels) in enumerate(loader):
        pts,labels = pts.to(DEVICE),labels.to(DEVICE)
        logits,logits_all = model.forward_active_train(pts)
        loss=active_loss(logits,logits_all,labels,model)/GRAD_ACCUM
        if torch.isfinite(loss): loss.backward()
        if (step+1)%GRAD_ACCUM==0 or step+1==len(loader):
            nn.utils.clip_grad_norm_(model.parameters(),10.0); opt.step(); opt.zero_grad()
        b=pts.shape[0]; tl+=loss.item()*GRAD_ACCUM*b; ta+=(logits.argmax(1)==labels).sum().item(); n+=b
    return tl/n, ta/n

@torch.no_grad()
def eval_spm(model, loader, n_vote=1):
    model.eval(); correct=total=0
    for pts,labels in loader:
        pts,labels = pts.to(DEVICE),labels.to(DEVICE)
        ps = torch.zeros(pts.shape[0],model.num_classes,device=DEVICE)
        for _ in range(n_vote):
            th=random.uniform(0,2*math.pi); c,s=math.cos(th),math.sin(th)
            rz=torch.tensor([[c,-s,0],[s,c,0],[0,0,1]],dtype=torch.float32,device=DEVICE)
            ps += model(pts@rz.T).softmax(dim=-1)
        correct+=(ps.argmax(1)==labels).sum().item(); total+=pts.shape[0]
    return correct/total

@torch.no_grad()
def eval_asp(model, loader):
    model.eval(); correct=total=slices=0
    for pts,labels in loader:
        pts,labels = pts.to(DEVICE),labels.to(DEVICE)
        logits,used = model.forward_active_infer(pts,EXIT_THR)
        correct+=(logits.argmax(1)==labels).sum().item(); total+=pts.shape[0]; slices+=used*pts.shape[0]
    return correct/total, slices/total


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_training(histories, save_dir):
    fig,axes = plt.subplots(1,len(histories),figsize=(7*len(histories),5),dpi=120)
    if len(histories)==1: axes=[axes]
    for ax,(ds,hist) in zip(axes,histories.items()):
        for key,color in [("spm","#2196F3"),("asp","#F44336")]:
            eps=[x["ep"] for x in hist[key]]; train=[x["tr"]*100 for x in hist[key]]
            vals=[(x["ep"],x["val"]*100) for x in hist[key] if x["val"] is not None]
            ax.plot(eps,train,color=color,linestyle="--",alpha=0.35)
            if vals: ax.plot([v[0] for v in vals],[v[1] for v in vals],color=color,marker="o",label=f"{key.upper()} val")
        ax.set_title(ds); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0,103); ax.grid(True,alpha=0.2); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(save_dir,"01_training_curves.png"),bbox_inches="tight"); plt.close()

def plot_bars(results, save_dir):
    names=list(results); x=np.arange(len(names))
    spm=[results[n]["spm_best"]*100 for n in names]; asp=[results[n]["asp_best"]*100 for n in names]
    fig,ax = plt.subplots(figsize=(max(6,3.5*len(names)),5),dpi=120); w=0.35
    ax.bar(x-w/2,spm,w,label="SPM",color="#2196F3"); ax.bar(x+w/2,asp,w,label="ASP",color="#F44336")
    for i,(s,a) in enumerate(zip(spm,asp)):
        ax.text(i-w/2,s+0.3,f"{s:.2f}%",ha="center",fontsize=9)
        ax.text(i+w/2,a+0.3,f"{a:.2f}%\n({a-s:+.2f}pp)",ha="center",fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("Best val acc (%)"); ax.set_title("SPM vs ASP"); ax.grid(axis="y",alpha=0.2); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(save_dir,"02_accuracy_bars.png"),bbox_inches="tight"); plt.close()


# ── Dataset config ────────────────────────────────────────────────────────────

def dataset_config():
    global MN10_DIR, MN40_DIR
    cfg = {}
    if "ModelNet10" in DATASET_NAMES:
        if MN10_DIR is None:
            print("\n[1] Downloading ModelNet10 ..."); MN10_DIR=_download("ModelNet10","balraj98/modelnet10-princeton-3d-object-dataset")
        cfg["ModelNet10"]={"root":MN10_DIR,"classes":10}
    if "ModelNet40" in DATASET_NAMES:
        if MN40_DIR is None:
            print("\n[1] Downloading ModelNet40 ..."); MN40_DIR=_download("ModelNet40","balraj98/modelnet40-princeton-3d-object-dataset")
        cfg["ModelNet40"]={"root":MN40_DIR,"classes":40}
    return cfg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    results={}; histories={}
    ds_config = dataset_config()

    for ds_name, ds_cfg in ds_config.items():
        print("\n"+"="*76)
        print(f"Dataset: {ds_name}  classes={ds_cfg['classes']}")
        print("="*76)
        train_ds=ModelNetDataset(ds_cfg["root"],NUM_POINTS,"train")
        val_ds=ModelNetDataset(ds_cfg["root"],NUM_POINTS,"test")
        train_l=DataLoader(train_ds,BATCH,shuffle=True,num_workers=NUM_WORKERS,pin_memory=ON_GPU,drop_last=True)
        val_l=DataLoader(val_ds,BATCH,shuffle=False,num_workers=NUM_WORKERS,pin_memory=ON_GPU)
        nc=ds_cfg["classes"]
        print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}  Batches/ep: {len(train_l)}")

        # SPM
        spm=make_spm(nc); spm_p=sum(p.numel() for p in spm.parameters())
        print(f"\n[SPM] params={spm_p:,}")
        spm_opt=torch.optim.AdamW(spm.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
        spm_sch=make_scheduler(spm_opt,EPOCHS,WARMUP_EP)
        spm_latest=os.path.join(CKPT_DIR,f"spm_{ds_name}_latest.pt")
        start_ep,best_spm,spm_hist = load_ckpt(spm_latest,spm,spm_opt,spm_sch)

        for ep in range(start_ep,EPOCHS):
            t0=time.time(); _,tr_acc=train_spm_epoch(spm,train_l,spm_opt); spm_sch.step()
            val_acc=None; is_best=False
            if (ep+1)%5==0 or ep+1==EPOCHS:
                val_acc=eval_spm(spm,val_l,N_VOTE)
                if val_acc>best_spm: best_spm=val_acc; is_best=True; torch.save(spm.state_dict(),os.path.join(CKPT_DIR,f"spm_{ds_name}_best.pth"))
                print(f"  [SPM] Ep {ep+1:3d}/{EPOCHS} tr={tr_acc:.4f} val={val_acc:.4f} {'*' if is_best else ' '} {time.time()-t0:.0f}s")
            spm_hist.append({"ep":ep+1,"tr":tr_acc,"val":val_acc})
            save_ckpt(spm_latest,spm,spm_opt,spm_sch,ep+1,best_spm,spm_hist)

        # ASP
        asp=make_asp(nc); asp_p=sum(p.numel() for p in asp.parameters())
        print(f"\n[ASP] params={asp_p:,}  (+SSP overhead: {asp_p-spm_p:,})")
        asp_opt=torch.optim.AdamW(asp.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
        asp_sch=make_scheduler(asp_opt,EPOCHS,WARMUP_EP)
        asp_latest=os.path.join(CKPT_DIR,f"asp_{ds_name}_latest.pt")
        start_ep,best_asp,asp_hist = load_ckpt(asp_latest,asp,asp_opt,asp_sch)
        best_asp_sl=ASP_STEPS

        for ep in range(start_ep,EPOCHS):
            t0=time.time(); _,tr_acc=train_asp_epoch(asp,train_l,asp_opt,ep); asp_sch.step()
            val_acc=val_sl=None; is_best=False
            if (ep+1)%5==0 or ep+1==EPOCHS:
                val_acc,val_sl=eval_asp(asp,val_l)
                if val_acc>best_asp: best_asp=val_acc; best_asp_sl=val_sl; is_best=True; torch.save(asp.state_dict(),os.path.join(CKPT_DIR,f"asp_{ds_name}_best.pth"))
                print(f"  [ASP] Ep {ep+1:3d}/{EPOCHS} tr={tr_acc:.4f} val={val_acc:.4f} {'*' if is_best else ' '} slices={val_sl:.2f}/{ASP_STEPS} {time.time()-t0:.0f}s")
            asp_hist.append({"ep":ep+1,"tr":tr_acc,"val":val_acc})
            save_ckpt(asp_latest,asp,asp_opt,asp_sch,ep+1,best_asp,asp_hist)

        fr_mean=asp.mean_firing_rate()
        results[ds_name]={"spm_best":best_spm,"asp_best":best_asp,"delta_pp":(best_asp-best_spm)*100,"asp_avg_chunks":best_asp_sl,"firing_rate":fr_mean}
        histories[ds_name]={"spm":spm_hist,"asp":asp_hist,"summary":results[ds_name]}

        # Save history to Drive logs
        try:
            hp=os.path.join(LOG_DIR,f"history_{ds_name}.json")
            with open(hp,"w") as hf: json.dump(histories[ds_name],hf,indent=2)
            print(f"  [Log] History -> {hp}")
        except: pass

        print(f"\nSummary {ds_name}")
        print(f"  SPM best : {best_spm*100:.2f}%")
        print(f"  ASP best : {best_asp*100:.2f}%")
        print(f"  Delta    : {(best_asp-best_spm)*100:+.2f} pp")
        print(f"  ASP chunks: {best_asp_sl:.2f}/{ASP_STEPS}")

    # Save results
    try:
        with open(os.path.join(CKPT_DIR,"final_results.json"),"w") as f: json.dump(results,f,indent=2)
    except: pass
    try: plot_training(histories,CKPT_DIR); plot_bars(results,CKPT_DIR)
    except Exception as e: print(f"[WARN] Plots failed: {e}")

    print("\n"+"="*76)
    print("FINAL RESULTS")
    print("="*76)
    print(f"{'Dataset':<14} {'SPM':>9} {'ASP':>9} {'Delta':>9} {'Chunks':>10}")
    for ds,r in results.items():
        print(f"{ds:<14} {r['spm_best']*100:>8.2f}% {r['asp_best']*100:>8.2f}% {r['delta_pp']:>+8.2f} {r['asp_avg_chunks']:>6.2f}/{ASP_STEPS}")
    print(f"\nOutputs saved to: {CKPT_DIR}")
    if ON_COLAB and DRIVE_ROOT:
        print(f"[Drive] All persisted to: {DRIVE_ROOT}")
        print(f"  Checkpoints : {CKPT_DIR}")
        print(f"  Epoch ckpts : {EPOCH_DIR}")
        print(f"  Logs        : {LOG_DIR}")

if __name__ == "__main__":
    main()
