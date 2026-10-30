"""Hàm gốc từ notebook V2; không chạy cell Colab, tải dữ liệu hoặc test."""
import copy
import math
import os
import random
import time
import numpy as np
import cv2
import pyclipper
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from third_party.VietLPR_v2.models import FastDBNet, FastCRNN

IMG_CACHE = {}
CTC = nn.CTCLoss(blank=0, zero_infinity=True)


def load_img(p):
    img = cv2.imdecode(IMG_CACHE[p], cv2.IMREAD_COLOR) if p in IMG_CACHE else cv2.imread(p)
    if img is None:
        raise IOError(p)
    return img

def order_points(pts):
    """4 điểm (hoặc polygon) → [tl, tr, br, bl]. Polygon >4 điểm → minAreaRect trước."""
    pts = np.asarray(pts, np.float32).reshape(-1, 2)
    if len(pts) != 4:
        pts = cv2.boxPoints(cv2.minAreaRect(pts))
    xs = pts[np.argsort(pts[:, 0], kind='stable')]
    tl, bl = xs[:2][np.argsort(xs[:2, 1])]
    tr, br = xs[2:][np.argsort(xs[2:, 1])]
    return np.array([tl, tr, br, bl], np.float32)

def rec_target_size(quad_w, quad_h):
    if REC_RESIZE == 'stretch':
        return (REC_W, REC_H)
    return (int(np.clip(round(quad_w * REC_H / max(quad_h, 1)), 8, REC_W)), REC_H)

def crop_for_rec(img, quad):
    """Warp phối cảnh quad → ảnh REC_HxREC_W bằng 1 phép warp duy nhất (crop + resize cùng lúc)."""
    tl, tr, br, bl = quad
    qw = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    qh = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    w, h = rec_target_size(qw, qh)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    out = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if w < REC_W:
        l = (REC_W - w) // 2
        out = cv2.copyMakeBorder(out, 0, 0, l, REC_W - w - l, cv2.BORDER_CONSTANT, value=0)
    return out

def full_quad(img):
    h, w = img.shape[:2]
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)

def sort_lines(quads):
    """Thứ tự đọc: trên → dưới; cùng hàng thì trái → phải. Trả về list index."""
    if len(quads) <= 1:
        return list(range(len(quads)))
    info = [(q[:, 1].mean(), q[:, 0].mean(), q[:, 1].max() - q[:, 1].min()) for q in quads]
    rows = []
    for i in sorted(range(len(quads)), key=lambda i: info[i][0]):
        j = rows[-1][-1] if rows else None
        if j is not None and abs(info[i][0] - info[j][0]) < 0.5 * min(info[i][2], info[j][2]):
            rows[-1].append(i)
        else:
            rows.append([i])
    return [i for r in rows for i in sorted(r, key=lambda k: info[k][1])]

def photo_aug(img):
    """Biến đổi màu/chất lượng ảnh. Không đổi hình học → không ảnh hưởng nhãn."""
    if random.random() < 0.5:
        a, b = (random.uniform(0.6, 1.4), random.uniform(-40, 40))
        img = cv2.LUT(img, np.clip(np.arange(256) * a + b, 0, 255).astype(np.uint8))
    if random.random() < 0.3:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + random.randint(-12, 12)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.6, 1.4), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if random.random() < 0.15:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if random.random() < 0.05:
        img = 255 - img
    r = random.random()
    if r < 0.12:
        img = cv2.GaussianBlur(img, (3, 3), random.uniform(0.3, 1.2))
    elif r < 0.22:
        k = random.choice([3, 5])
        ker = np.zeros((k, k), np.float32)
        ker[k // 2] = 1.0 / k
        ker = cv2.warpAffine(ker, cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), random.uniform(0, 180), 1), (k, k))
        img = cv2.filter2D(img, -1, ker / max(ker.sum(), 1e-06))
    if random.random() < 0.25:
        h, w = img.shape[:2]
        s = random.uniform(0.4, 0.9)
        img = cv2.resize(cv2.resize(img, (max(4, int(w * s)), max(4, int(h * s))), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    if random.random() < 0.2:
        img = np.clip(img + np.random.normal(0, random.uniform(3, 12), img.shape), 0, 255).astype(np.uint8)
    if random.random() < 0.3:
        img = cv2.imdecode(cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(30, 90)])[1], 1)
    return img

def jitter_quad(q, p=0.85, out=(0.08, 0.15), inn=(0.015, 0.06), rot=3.0):
    """Lệch 4 góc box (nở ra tới 8%/15%, co vào tối đa 1.5%/6%) + xoay nhẹ
    → mô phỏng box mà DET dự đoán lệch so với GT."""
    if random.random() > p:
        return q
    tl, tr, br, bl = q
    w = max(np.linalg.norm(tr - tl), 1.0)
    h = max(np.linalg.norm(bl - tl), 1.0)
    u = (tr - tl) / w
    v = (bl - tl) / h
    o = q.copy()
    for i, (sx, sy) in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
        o[i] += sx * random.uniform(-inn[0], out[0]) * w * u + sy * random.uniform(-inn[1], out[1]) * h * v
    if rot:
        c = o.mean(0)
        a = math.radians(random.uniform(-rot, rot))
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], np.float32)
        o = (o - c) @ R.T + c
    return o.astype(np.float32)

def _fit_canvas(S, H, W2, H2):
    """Áp bước biến đổi S lên canvas hiện tại (W2,H2), dịch để canvas mới chứa trọn nó."""
    c = cv2.perspectiveTransform(np.array([[[0, 0]], [[W2, 0]], [[W2, H2]], [[0, H2]]], np.float64), S).reshape(-1, 2)
    mn, mx = (c.min(0), c.max(0))
    T = np.array([[1, 0, -mn[0]], [0, 1, -mn[1]], [0, 0, 1]], np.float64)
    return (T @ S @ H, float(mx[0] - mn[0]), float(mx[1] - mn[1]))

def det_geo_aug(img, polys):
    """Pad / xoay / phối cảnh / lật / crop — áp CÙNG một homography cho ảnh và polygon
    (bản cũ lật ảnh nhưng KHÔNG lật mask → nhãn sai 50% mẫu)."""
    h, w = img.shape[:2]
    H = np.eye(3)
    if random.random() < 0.6:
        l, r = (random.uniform(0, 0.25) * w, random.uniform(0, 0.25) * w)
        t, b = (random.uniform(0, 0.25) * h, random.uniform(0, 0.25) * h)
        H = np.array([[1, 0, l], [0, 1, t], [0, 0, 1]], np.float64)
        W2, H2 = (w + l + r, h + t + b)
    else:
        W2, H2 = (w, h)
    if random.random() < 0.5:
        R = np.vstack([cv2.getRotationMatrix2D((W2 / 2, H2 / 2), random.uniform(-10, 10), 1.0), [0, 0, 1]])
        H, W2, H2 = _fit_canvas(R, H, W2, H2)
    if random.random() < 0.3:
        src = np.float32([[0, 0], [W2, 0], [W2, H2], [0, H2]])
        dst = (src + np.random.uniform(-0.06, 0.06, (4, 2)) * [W2, H2]).astype(np.float32)
        H, W2, H2 = _fit_canvas(cv2.getPerspectiveTransform(src, dst).astype(np.float64), H, W2, H2)
    if random.random() < 0.5:
        H = np.array([[-1, 0, W2], [0, 1, 0], [0, 0, 1]], np.float64) @ H
    if random.random() < 0.3:
        cl, cr = (random.uniform(0, 0.04) * W2, random.uniform(0, 0.04) * W2)
        ct, cb = (random.uniform(0, 0.04) * H2, random.uniform(0, 0.04) * H2)
        H = np.array([[1, 0, -cl], [0, 1, -ct], [0, 0, 1]], np.float64) @ H
        W2 -= cl + cr
        H2 -= ct + cb
    W2, H2 = (max(8, int(round(W2))), max(8, int(round(H2))))
    if random.random() < 0.5:
        out = cv2.warpPerspective(img, H, (W2, H2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    else:
        out = cv2.warpPerspective(img, H, (W2, H2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=tuple((int(x) for x in np.random.randint(0, 256, 3))))
    new = []
    for p in polys:
        q = cv2.perspectiveTransform(p.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
        q[:, 0] = np.clip(q[:, 0], 0, W2 - 1)
        q[:, 1] = np.clip(q[:, 1], 0, H2 - 1)
        new.append(q.astype(np.float32))
    return (out, new)

def worker_init(wid):
    s = (torch.initial_seed() + wid) % 2 ** 32
    np.random.seed(s)
    random.seed(s)

class ModelEMA:
    """Trung bình động trọng số — val/test/export đều dùng bản EMA (ổn định, thường +0.5–2%)."""

    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.n = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])

def param_groups(model, wd):
    decay, no_decay = ([], [])
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or n.endswith('.pos') or n == 'pos' else decay).append(p)
    return [{'params': decay, 'weight_decay': wd}, {'params': no_decay, 'weight_decay': 0.0}]

def train_model(name, model, cfg, train_loader, step_fn, val_fn, epochs, lr):
    """step_fn(model, batch) → loss | val_fn(model) → dict có key 'score' (càng cao càng tốt)."""
    last_p, best_p = (f'{OUT_DIR}/{name}_last.pth', f'{OUT_DIR}/{name}_best.pth')
    opt = torch.optim.AdamW(param_groups(model, WEIGHT_DECAY), lr=lr, betas=(0.9, 0.99))
    n_it = len(train_loader)
    total = epochs * n_it
    warm = max(1, int(WARMUP_EPOCHS * n_it))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: (i + 1) / warm if i < warm else 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, (i - warm) / max(1, total - warm)))))
    scaler = torch.amp.GradScaler('cuda', enabled=AMP)
    ema = ModelEMA(model, decay=1 - 1 / max(100, 2 * n_it))
    start, best, bad, hist = (0, -1000000000.0, 0, [])
    if RESUME and os.path.exists(last_p):
        ck = torch.load(last_p, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck['model'])
        ema.ema.load_state_dict(ck['ema'])
        ema.n = ck['ema_n']
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        scaler.load_state_dict(ck['scaler'])
        start, best, bad, hist = (ck['epoch'] + 1, ck['best'], ck['bad'], ck['hist'])
        print(f'↻ Resume {name} từ epoch {start} (best score={best:.4f})')
    for ep in range(start, epochs):
        if bad >= PATIENCE:
            print(f'Early stop: {PATIENCE} epoch không cải thiện')
            break
        model.train()
        t0 = time.time()
        tot, n = (0.0, 0)
        pbar = tqdm(train_loader, desc=f'{name} {ep + 1}/{epochs}', leave=False)
        for batch in pbar:
            with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=AMP):
                loss = step_fn(model, batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            if torch.isfinite(loss):
                tot += loss.item()
                n += 1
            pbar.set_postfix(loss=f'{tot / max(1, n):.4f}')
        m = val_fn(ema.ema)
        m['loss'] = tot / max(1, n)
        m['epoch'] = ep + 1
        hist.append(m)
        improved = m['score'] > best
        if improved:
            best, bad = (m['score'], 0)
            torch.save({'model': ema.ema.state_dict(), 'cfg': cfg, 'metrics': m}, best_p)
        else:
            bad += 1
        torch.save({'model': model.state_dict(), 'ema': ema.ema.state_dict(), 'ema_n': ema.n, 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'scaler': scaler.state_dict(), 'epoch': ep, 'best': best, 'bad': bad, 'hist': hist, 'cfg': cfg}, last_p)
        msg = ' '.join((f'{k}={v:.4f}' for k, v in m.items() if k not in ('epoch', 'score')))
        print(f"[{name} {ep + 1:3d}] lr={opt.param_groups[0]['lr']:.2e} {msg} ({time.time() - t0:.0f}s){('  ✓ best' if improved else '')}")
    return hist

def offset_poly(poly, dist):
    """Co (dist<0) / nở (dist>0) polygon bằng Vatti clipping."""
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(np.round(poly * _SUB).astype(np.int64).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    res = pco.Execute(dist * _SUB)
    if not res:
        return None
    return np.array(max(res, key=lambda r: abs(cv2.contourArea(np.array(r, np.float32)))), np.float32) / _SUB

def make_det_gt(polys, ignores, W, H):
    """Shrink-map kiểu DBNet: polygon co lại theo D = A(1-r²)/L → 2 dòng của biển 2 tầng
    luôn tách rời nhau (bản cũ fill + dilate → 2 dòng dính nhau → phải tách bằng heuristic)."""
    gt = np.zeros((DET_M, DET_M), np.uint8)
    mask = np.ones((DET_M, DET_M), np.uint8)
    s = np.array([DET_M / W, DET_M / H], np.float32)
    for p, ig in zip(polys, ignores):
        q = (p + 0.5) * s - 0.5
        qi = [np.round(q * _SUB).astype(np.int32)]
        area, peri = (abs(cv2.contourArea(q)), cv2.arcLength(q, True))
        if ig or area < 4 or min(cv2.minAreaRect(q)[1]) < 2:
            cv2.fillPoly(mask, qi, 0, lineType=cv2.LINE_8, shift=4)
            continue
        k = offset_poly(q, -area * (1 - DET_SHRINK ** 2) / max(peri, 1e-06))
        if k is None or len(k) < 3:
            cv2.fillPoly(mask, qi, 0, lineType=cv2.LINE_8, shift=4)
            continue
        cv2.fillPoly(gt, [np.round(k * _SUB).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    return (gt, mask)

class DetDataset(Dataset):

    def __init__(self, items, augment):
        self.items, self.augment = (items, augment)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, anns = self.items[i]
        img = load_img(path)
        polys = [a['poly'].copy() for a in anns]
        ign = [a['ignore'] for a in anns]
        if self.augment:
            img, polys = det_geo_aug(img, polys)
            img = photo_aug(img)
        H, W = img.shape[:2]
        gt, mask = make_det_gt(polys, ign, W, H)
        x = cv2.resize(img, (DET_SIZE, DET_SIZE), interpolation=cv2.INTER_LINEAR)
        return (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), torch.from_numpy(gt)[None], torch.from_numpy(mask)[None])

def det_loss(logits, gt, mask, ohem=3.0):
    """BCE với OHEM (pos:neg = 1:3) + Dice — như DBNet."""
    gt, mask = (gt.float(), mask.float())
    bce = F.binary_cross_entropy_with_logits(logits, gt, reduction='none')
    pos = gt * mask
    neg = (1 - gt) * mask
    n_pos = int(pos.sum().item())
    n_neg = int(min(neg.sum().item(), max(n_pos * ohem, 64)))
    neg_loss = torch.topk((bce * neg).flatten(), n_neg).values.sum() if n_neg > 0 else bce.sum() * 0
    l_bce = ((bce * pos).sum() + neg_loss) / (n_pos + n_neg + 1e-06)
    prob = torch.sigmoid(logits)
    l_dice = 1 - 2 * (prob * gt * mask).sum() / ((prob * mask).sum() + (gt * mask).sum() + 1e-06)
    return l_bce + l_dice

def det_step(model, batch):
    x, gt, mask = (b.to(DEVICE, non_blocking=True) for b in batch)
    return det_loss(model(x.float().div_(255)).float(), gt, mask)

def _fit_line(x, y, default):
    """Bình phương tối thiểu y = k·x + b (cần ≥3 điểm trải đủ rộng)."""
    if len(x) >= 3 and np.ptp(x) > 1.0:
        xm, ym = (x.mean(), y.mean())
        k = float(((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum())
        return (k, float(ym - k * xm))
    return default

def fit_quad(cnt, unclip=0.0, pad_x=0.0):
    """Contour lõi → tứ giác [tl,tr,br,bl].
    - Cạnh TRÊN / DƯỚI: fit đường thẳng riêng (chỉ dùng 70% đoạn giữa → không bị góc bo tròn làm lệch)
      → giữ được hình thang do phối cảnh (minAreaRect luôn ép thành hình chữ nhật xoay).
    - Cạnh TRÁI / PHẢI: vuông góc với dòng tại 2 đầu (fit cạnh ngắn rất nhiễu → chữ bị xiên).
    - unclip: đẩy 4 cạnh ra D = A·unclip/L (kiểu DBNet);  pad_x: nới thêm 2 đầu pad_x × chiều cao dòng
      (cắt lẹm 1 ký tự là sai cả biển, còn dư nền thì REC đã quen nhờ augment)."""
    p = cnt.reshape(-1, 2).astype(np.float32)
    (cx, cy), (w, h), ang = cv2.minAreaRect(p)
    a = math.radians(ang if w >= h else ang + 90)
    U = np.array([math.cos(a), math.sin(a)], np.float32)
    if U[0] < 0:
        U = -U
    V = np.array([-U[1], U[0]], np.float32)
    c = np.array([cx, cy], np.float32)
    pu, pv = ((p - c) @ U, (p - c) @ V)
    mu = (pu > pu.min() + 0.15 * np.ptp(pu)) & (pu < pu.max() - 0.15 * np.ptp(pu))
    kt, bt = _fit_line(pu[mu & (pv < 0)], pv[mu & (pv < 0)], (0.0, float(pv.min())))
    kb, bb = _fit_line(pu[mu & (pv > 0)], pv[mu & (pv > 0)], (0.0, float(pv.max())))
    nl, nr = (float(pu.min()), float(pu.max()))
    corners = lambda: np.array([c + u * U + (k * u + b) * V for u, k, b in ((nl, kt, bt), (nr, kt, bt), (nr, kb, bb), (nl, kb, bb))], np.float32)
    if unclip:
        q = corners()
        D = abs(cv2.contourArea(q)) * unclip / max(cv2.arcLength(q, True), 1e-06)
        bt -= D * math.hypot(1, kt)
        bb += D * math.hypot(1, kb)
        nl -= D
        nr += D
    if pad_x:
        nl -= pad_x * (bb - bt)
        nr += pad_x * (bb - bt)
    return corners()

def det_postprocess(prob, W, H, thresh=None, box_thresh=None, unclip=None, pad_x=None, max_cand=6, min_size=1.5):
    thresh = DET_THRESH if thresh is None else thresh
    box_thresh = DET_BOX_TH if box_thresh is None else box_thresh
    unclip = DET_UNCLIP if unclip is None else unclip
    pad_x = DET_PAD_X if pad_x is None else pad_x
    Mh, Mw = prob.shape
    cnts, _ = cv2.findContours((prob > thresh).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:max_cand]
    out = []
    for c in cnts:
        if len(c) < 4 or min(cv2.minAreaRect(c)[1]) < min_size:
            continue
        x, y, w, h = cv2.boundingRect(c)
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [c - np.array([x, y], np.int32)], 1)
        score = cv2.mean(prob[y:y + h, x:x + w], m)[0]
        if score < box_thresh:
            continue
        q = fit_quad(c, unclip, pad_x)
        if min(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[3] - q[0])) < min_size + 2:
            continue
        q[:, 0] = np.clip((q[:, 0] + 0.5) * W / Mw - 0.5, 0, W)
        q[:, 1] = np.clip((q[:, 1] + 0.5) * H / Mh - 0.5, 0, H)
        out.append((q, float(score)))
    return out

def det_preprocess(img):
    return cv2.resize(img, (DET_SIZE, DET_SIZE), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)

@torch.no_grad()
def det_predict_probs(model, imgs, bs=256):
    model.eval()
    probs = []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(np.stack([det_preprocess(im) for im in imgs[i:i + bs]])).to(DEVICE)
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=AMP):
            p = torch.sigmoid(model(x.float() / 255).float())
        probs.extend(p[:, 0].cpu().numpy())
    return probs

def quad_iou(a, b):
    a = cv2.convexHull(a.astype(np.float32)).reshape(-1, 2)
    b = cv2.convexHull(b.astype(np.float32)).reshape(-1, 2)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return inter / union if union > 0 else 0.0

def match_boxes(pred, gts, iou_th=0.5):
    """Ghép 1-1 tham lam theo IoU → (tp, fp, fn, list iou của cặp khớp)."""
    pairs = sorted(((quad_iou(p, g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gts)), reverse=True)
    used_p, used_g, ious = (set(), set(), [])
    for iou, i, j in pairs:
        if iou < iou_th:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        ious.append(iou)
    return (len(ious), len(pred) - len(ious), len(gts) - len(ious), ious)

def eval_det(model, items, probs=None, **pp):
    imgs = [load_img(p) for p, _ in items]
    probs = det_predict_probs(model, imgs) if probs is None else probs
    TP = FP = FN = exact = 0
    ious = []
    for img, (_, anns), prob in zip(imgs, items, probs):
        H, W = img.shape[:2]
        pred = [q for q, _ in det_postprocess(prob, W, H, **pp)]
        gts = [a['poly'] for a in anns if not a['ignore']]
        tp, fp, fn, iu = match_boxes(pred, gts)
        TP += tp
        FP += fp
        FN += fn
        ious += iu
        exact += fp == 0 and fn == 0
    P = TP / max(1, TP + FP)
    R = TP / max(1, TP + FN)
    Hm = 2 * P * R / max(1e-09, P + R)
    return {'P': P, 'R': R, 'H': Hm, 'mIoU': float(np.mean(ious)) if ious else 0.0, 'img_exact': exact / max(1, len(items)), 'score': Hm + 0.001 * (np.mean(ious) if ious else 0)}

def build_det(cfg):
    return FastDBNet(width=cfg['width'], fpn_ch=cfg['fpn_ch'])

def encode_text(t):
    return [CHAR2IDX[c] for c in t if c in CHAR2IDX]

class RecDataset(Dataset):
    """Crop có sẵn trong rec/. Augment hình học = jitter 4 góc (mô phỏng box DET lệch) + warp 1 lần."""

    def __init__(self, items, augment):
        self.items, self.augment = ([(p, t) for p, t in items if encode_text(t)], augment)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, t = self.items[i]
        img = load_img(p)
        q = full_quad(img)
        if self.augment:
            q = jitter_quad(q)
        x = crop_for_rec(img, q)
        if self.augment:
            x = photo_aug(x)
        return (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), torch.tensor(encode_text(t)))

class RecFromDetDataset(RecDataset):
    """Cắt online từng dòng từ ảnh det.  50% dùng GT polygon (+jitter), 50% dùng CHÍNH box mà DET đã train
    dự đoán (nếu có) → REC quen đúng kiểu sai lệch của DET → giảm khoảng cách "REC trên crop GT" vs "end-to-end"."""

    def __init__(self, det_items, augment, pred_quads=None):
        self.items = [(p, order_points(a['poly']), a['text'], (pred_quads or {}).get((p, j))) for p, anns in det_items for j, a in enumerate(anns) if not a['ignore'] and encode_text(a['text'])]
        self.augment = augment

    def __getitem__(self, i):
        p, q, t, qp = self.items[i]
        img = load_img(p)
        if self.augment:
            q = jitter_quad(qp, p=0.5, out=(0.04, 0.08), inn=(0.01, 0.03)) if qp is not None and random.random() < 0.5 else jitter_quad(q, p=0.95)
        x = crop_for_rec(img, q)
        if self.augment:
            x = photo_aug(x)
        return (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), torch.tensor(encode_text(t)))

@torch.no_grad()
def predict_train_quads(det_m, det_items, chunk=2048):
    """Chạy DET trên ảnh train → {(path, idx_dòng): quad dự đoán khớp GT (IoU ≥ 0.5)}."""
    out = {}
    for s in range(0, len(det_items), chunk):
        part = det_items[s:s + chunk]
        imgs = [load_img(p) for p, _ in part]
        for img, (p, anns), prob in zip(imgs, part, det_predict_probs(det_m, imgs)):
            H, W = img.shape[:2]
            dets = [q for q, _ in det_postprocess(prob, W, H)]
            for j, a in enumerate(anns):
                if not dets:
                    break
                ious = [quad_iou(d, a['poly']) for d in dets]
                if max(ious) >= 0.5:
                    out[p, j] = dets[int(np.argmax(ious))]
    return out

def rec_collate(batch):
    xs, ts = zip(*batch)
    return (torch.stack(xs), torch.cat(ts), torch.tensor([len(t) for t in ts], dtype=torch.long))

def rec_step(model, batch):
    x, tg, ln = batch
    x = x.to(DEVICE, non_blocking=True).float().div_(255)
    logp = model(x).float().log_softmax(2).permute(1, 0, 2)
    T, B = logp.shape[:2]
    return CTC(logp, tg.to(DEVICE), torch.full((B,), T, dtype=torch.long, device=DEVICE), ln.to(DEVICE))

def ctc_greedy(logits):
    """logits (B,T,C) → list[(text, conf)]. conf = trung bình xác suất max tại các ký tự được giữ."""
    prob = logits.float().softmax(-1)
    mx, idx = prob.max(-1)
    idx, mx = (idx.cpu().numpy(), mx.cpu().numpy())
    out = []
    for row, pr in zip(idx, mx):
        keep = (row != 0) & np.r_[True, row[1:] != row[:-1]]
        out.append((''.join((CHARS[k - 1] for k in row[keep])), float(pr[keep].mean()) if keep.any() else 0.0))
    return out

@torch.no_grad()
def rec_predict(model, crops, bs=512):
    """crops: list ảnh uint8 (REC_H, REC_W, 3) → list[(text, conf)]"""
    model.eval()
    res = []
    for i in range(0, len(crops), bs):
        x = torch.from_numpy(np.stack(crops[i:i + bs]).transpose(0, 3, 1, 2)).to(DEVICE)
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=AMP):
            lg = model(x.float() / 255)
        res += ctc_greedy(lg)
    return res

def edit_distance(a, b):
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def text_metrics(preds, gts):
    acc = np.mean([p == g for p, g in zip(preds, gts)])
    cer = sum((edit_distance(p, g) for p, g in zip(preds, gts))) / max(1, sum((len(g) for g in gts)))
    return (float(acc), float(cer))

@torch.no_grad()
def eval_rec(model, loader):
    model.eval()
    preds, gts = ([], [])
    for x, tg, ln in loader:
        x = x.to(DEVICE, non_blocking=True)
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=AMP):
            lg = model(x.float() / 255)
        preds += [t for t, _ in ctc_greedy(lg)]
        s = 0
        for l in ln.tolist():
            gts.append(''.join((CHARS[k - 1] for k in tg[s:s + l].tolist())))
            s += l
    acc, cer = text_metrics(preds, gts)
    return {'acc': acc, 'cer': cer, 'score': acc - 0.1 * cer, '_preds': preds, '_gts': gts}

def build_rec(cfg):
    return FastCRNN(num_classes=len(cfg['chars']) + 1, img_h=cfg['h'], img_w=cfg['w'], width=cfg['width'], neck=cfg['neck'], neck_layers=cfg['neck_layers'], dim=cfg['dim'])
