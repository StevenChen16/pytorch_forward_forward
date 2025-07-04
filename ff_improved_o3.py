# FFNet激进ablation试验对比框架
# - 支持功能开关: Focal Loss, CutMix, LayerMargin, Goodness温度, 容量可变
# - 支持一次性A/B多方案对比
import time, copy, torch, math, random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# ========== Dataset ==========
BATCH_SIZE = 128
NUM_WORKERS = 2

# 支持CutMix数据增强
class CutMixCollate:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    def __call__(self, batch):
        imgs, labels = zip(*batch)
        imgs = torch.stack(imgs, 0)
        labels = torch.tensor(labels)
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        rand_index = torch.randperm(imgs.size(0))
        shuffled_imgs = imgs[rand_index]
        shuffled_labels = labels[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
        imgs[:,:,bbx1:bbx2,bby1:bby2] = shuffled_imgs[:,:,bbx1:bbx2,bby1:bby2]
        lam_ = 1 - ((bbx2-bbx1)*(bby2-bby1)/(imgs.size(-1)*imgs.size(-2)))
        return imgs, labels, shuffled_labels, lam_

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def get_loader(dataset='cifar10', use_cutmix=False):
    if dataset == 'mnist':
        tf = transforms.ToTensor()
        trainset = datasets.MNIST('./data', True, download=True, transform=tf)
        testset  = datasets.MNIST('./data', False,download=True, transform=tf)
        ic, sz, C = 1, 28, 10
    else:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
        ])
        test_tf = transforms.ToTensor()
        trainset = datasets.CIFAR10('./data', True,  download=True, transform=train_tf)
        testset  = datasets.CIFAR10('./data', False, download=True, transform=test_tf)
        ic, sz, C = 3, 32, 10
    collate_fn = CutMixCollate(1.0) if use_cutmix else None
    train_loader = DataLoader(trainset, BATCH_SIZE, True,  num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    test_loader  = DataLoader(testset,  BATCH_SIZE, False, num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, test_loader, ic, sz, C

# ========== Model ========== #
def goodness(z):
    return z.flatten(1).pow(2).sum(1)

class ConvFF(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x):
        z = self.act(self.bn(self.conv(x)))
        return z, goodness(z)

class FFNet(nn.Module):
    def __init__(self, ic, C, sz, channel_scale=1):
        super().__init__()
        width = [int(channel_scale*x) for x in [64,128,256,512]]
        self.backbone = nn.ModuleList([
            ConvFF(ic + C, width[0], 1),
            ConvFF(width[0], width[1], 2),
            ConvFF(width[1], width[2], 2),
            ConvFF(width[2], width[3], 2)
        ])
        self.head = nn.Linear(width[-1]*(sz//8)*(sz//8), C)
    def forward(self, x):
        gs=[]
        for blk in self.backbone:
            x,g = blk(x)
            gs.append(g)
        logits = self.head(x.flatten(1))
        return logits, gs

# ========== Ablation Losses =========== #
def focal_loss(logits,y,gamma=2,alpha=1):
    logp = F.log_softmax(logits,1)
    p = logp.exp()
    y_onehot = F.one_hot(y,logits.size(1)).float()
    ce = -(y_onehot*logp).sum(1)
    fl = (alpha * (1-p.gather(1,y[:,None]).squeeze())**gamma * ce).mean()
    return fl

def layer_margin(gs):
    return sum([(gs[i]-gs[i-1]).pow(2).mean() for i in range(1,len(gs))])

def df_loss(gp,gn,margin=0.5):
    gp,gn = torch.stack(gp,1), torch.stack(gn,1)
    return torch.log1p(torch.exp(torch.clamp(gn-gp-margin,-60,60))).mean()

def nce_loss(gp,gn,tau=0.07):
    logits = torch.stack([gp,gn],1)/tau
    return F.cross_entropy(torch.clamp(logits,-80,80), torch.zeros(gp.size(0),dtype=torch.long,device=gp.device))

def ce_loss(logits,y,smooth=0.1):
    n = logits.size(1)
    if smooth>0:
        smooth_y = (1-smooth)*F.one_hot(y,n)+smooth/n
        return -(smooth_y*F.log_softmax(logits,1)).sum(1).mean()
    return F.cross_entropy(logits,y)

# ========== Ablation Training ========= #
def train_epoch(model,loader,opt,C, ablation_cfg, device='cuda'):
    model.train(); tot=samp=0
    for batch in loader:
        # cutmix batch: imgs, y1, y2, lam
        if isinstance(batch, tuple) and len(batch)==4:
            x,y,y2,lam = [b.to(device) for b in batch]
            B,_,H,W = x.size()
            x_pos = torch.cat([x, expand_label(one_hot(y,C),H,W)],1)
            x_neg = torch.cat([x, expand_label(one_hot(y2,C),H,W)],1)
        else:
            x,y = batch
            x,y = x.to(device), y.to(device)
            B,_,H,W = x.size()
            y2 = neg_labels(y,C)
            x_pos = torch.cat([x, expand_label(one_hot(y,C),H,W)],1)
            x_neg = torch.cat([x, expand_label(one_hot(y2,C),H,W)],1)
        log_p, gp = model(x_pos)
        _,      gn = model(x_neg)
        loss = df_loss(gp,gn)
        if ablation_cfg['nce']:
            loss += 0.3*nce_loss(gp[-1],gn[-1])
        if ablation_cfg['focal']:
            loss += 1.0*focal_loss(log_p,y)
        else:
            loss += 1.0*ce_loss(log_p,y)
        if ablation_cfg['margin']:
            loss += 0.1*layer_margin(gp)
        opt.zero_grad(); loss.backward(); opt.step()
        tot+=loss.item()*B; samp+=B
    return tot/samp

def val_loss(model,loader,C, ablation_cfg, device='cuda'):
    model.eval(); tot=samp=0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            B,_,H,W = x.size()
            y2 = neg_labels(y,C)
            x_pos = torch.cat([x, expand_label(one_hot(y,C),H,W)],1)
            x_neg = torch.cat([x, expand_label(one_hot(y2,C),H,W)],1)
            log_p, gp = model(x_pos)
            _,      gn = model(x_neg)
            loss = df_loss(gp,gn)
            if ablation_cfg['nce']:
                loss += 0.3*nce_loss(gp[-1],gn[-1])
            if ablation_cfg['focal']:
                loss += 1.0*focal_loss(log_p,y)
            else:
                loss += 1.0*ce_loss(log_p,y)
            if ablation_cfg['margin']:
                loss += 0.1*layer_margin(gp)
            tot+=loss.item()*B; samp+=B
    return tot/samp

def predict(model,x,C):
    B,_,H,W = x.size(); scores=[]
    for cls in range(C):
        lbl = torch.full((B,),cls,device=x.device)
        inp = torch.cat([x,expand_label(one_hot(lbl,C),H,W)],1)
        logits,_ = model(inp)
        scores.append(logits[:,cls])
    return torch.stack(scores,1)

def evaluate(model,loader,C, device='cuda'):
    model.eval(); corr=tot=0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            pred = predict(model,x,C).argmax(1)
            corr+=(pred==y).sum().item(); tot+=y.size(0)
    return corr/tot

def expand_label(oh,H,W):
    return oh.unsqueeze(2).unsqueeze(3).expand(-1,-1,H,W)*0.02

@torch.no_grad()
def neg_labels(y,C):
    r = torch.randint_like(y,0,C)
    r[r==y] = (r[r==y]+1)%C
    return r

def one_hot(y,C):
    return F.one_hot(y,C).float()

# ========== Compare Runs ========== #
def ablation_experiment(epochs=10, dataset='cifar10', device='cuda'):
    """
    跑多组方案对比,每组dict:
      'name': 实验名
      'nce': 是否用nce损失
      'focal': 是否用focal loss
      'cutmix': 是否cutmix
      'margin': 是否加层间margin
      'scale': 通道倍增
    """
    configs = [
        {'name':'base',     'nce':1, 'focal':0, 'cutmix':0, 'margin':0, 'scale':1},
        {'name':'cutmix',   'nce':1, 'focal':0, 'cutmix':1, 'margin':0, 'scale':1},
        {'name':'focal',    'nce':1, 'focal':1, 'cutmix':0, 'margin':0, 'scale':1},
        {'name':'margin',   'nce':1, 'focal':0, 'cutmix':0, 'margin':1, 'scale':1},
        {'name':'cutmix+focal','nce':1, 'focal':1, 'cutmix':1, 'margin':0, 'scale':1},
        {'name':'cutmix+margin','nce':1, 'focal':0, 'cutmix':1, 'margin':1, 'scale':1},
        {'name':'all','nce':1, 'focal':1, 'cutmix':1, 'margin':1, 'scale':1},
        {'name':'all+wide','nce':1, 'focal':1, 'cutmix':1, 'margin':1, 'scale':1.6},
    ]
    results = {}
    for cfg in configs:
        print(f"\n--- Experiment: {cfg['name']} ---")
        tr,te,ic,sz,C = get_loader(dataset, use_cutmix=cfg['cutmix'])
        model = FFNet(ic,C,sz,channel_scale=cfg['scale']).to(device)
        try:
            start_time = time.time()
            # model = torch.compile(model)
            print(f"Model compiled in {time.time() - start_time:.2f} seconds")
        except: pass
        opt = optim.AdamW(model.parameters(),lr=5e-4,weight_decay=5e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
        best=0
        logs=[]
        for ep in range(1,epochs+1):
            t0 = time.time()
            train_loss = train_epoch(model,tr,opt,C,cfg, device=device)
            val_loss_  = val_loss(model,te,C,cfg)
            acc        = evaluate(model,te,C)
            sched.step()
            best=max(best,acc)
            elapsed = time.time()-t0
            logs.append({'ep':ep,'train_loss':train_loss,'val_loss':val_loss_,'val_acc':acc,'best':best})
            print(f"Ep{ep:2d}|{cfg['name']:12}|TrainLoss:{train_loss:.3f}|ValLoss:{val_loss_:.3f}|ValAcc:{acc:.3f}|Best:{best:.3f}|Time:{elapsed:.1f}s")
        results[cfg['name']] = logs
    return results

# 运行：
if __name__=='__main__':
    ablation_experiment(epochs=10, dataset='cifar10', device='cuda' if torch.cuda.is_available() else 'cpu')
