"""
VietLPR v2 — inference độc lập bằng ONNX Runtime (chạy được trên PC / Raspberry Pi / Jetson CPU).
Cần: numpy, opencv-python, onnxruntime  +  det.onnx, rec.onnx, lpr_config.json

    from lpr_infer import LPR
    lpr = LPR("det.onnx", "rec.onnx", "lpr_config.json", threads=2)
    out = lpr(cv2.imread("plate.jpg"))        # ảnh crop biển số (BGR)
    print(out["plate"], out["lines"], out["conf"])
"""
import json, math
import numpy as np, cv2, onnxruntime as ort


def fit_quad(cnt, unclip=0.0, pad_x=0.0):
    """Contour lõi → tứ giác [tl,tr,br,bl]: fit cạnh trên/dưới, 2 cạnh bên vuông góc, nở ra theo unclip."""
    p = cnt.reshape(-1, 2).astype(np.float32)
    (cx, cy), (w, h), ang = cv2.minAreaRect(p)
    a = math.radians(ang if w >= h else ang + 90)
    U = np.array([math.cos(a), math.sin(a)], np.float32)
    if U[0] < 0: U = -U
    V = np.array([-U[1], U[0]], np.float32)
    c = np.array([cx, cy], np.float32)
    pu, pv = (p - c) @ U, (p - c) @ V
    mu = (pu > pu.min() + 0.15 * np.ptp(pu)) & (pu < pu.max() - 0.15 * np.ptp(pu))
    def fit(x, y, default):
        if len(x) >= 3 and np.ptp(x) > 1.0:
            xm, ym = x.mean(), y.mean(); k = float(((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum())
            return k, float(ym - k * xm)
        return default
    kt, bt = fit(pu[mu & (pv < 0)], pv[mu & (pv < 0)], (0.0, float(pv.min())))
    kb, bb = fit(pu[mu & (pv > 0)], pv[mu & (pv > 0)], (0.0, float(pv.max())))
    nl, nr = float(pu.min()), float(pu.max())
    corners = lambda: np.array([c + u * U + (k * u + b) * V for u, k, b in
                                ((nl, kt, bt), (nr, kt, bt), (nr, kb, bb), (nl, kb, bb))], np.float32)
    if unclip:
        q = corners()
        D = abs(cv2.contourArea(q)) * unclip / max(cv2.arcLength(q, True), 1e-6)
        bt -= D * math.hypot(1, kt); bb += D * math.hypot(1, kb); nl -= D; nr += D
    if pad_x:
        nl -= pad_x * (bb - bt); nr += pad_x * (bb - bt)
    return corners()


def sort_lines(quads):
    if len(quads) <= 1: return list(range(len(quads)))
    info = [(q[:, 1].mean(), q[:, 0].mean(), q[:, 1].max() - q[:, 1].min()) for q in quads]
    rows = []
    for i in sorted(range(len(quads)), key=lambda i: info[i][0]):
        j = rows[-1][-1] if rows else None
        if j is not None and abs(info[i][0] - info[j][0]) < 0.5 * min(info[i][2], info[j][2]): rows[-1].append(i)
        else: rows.append([i])
    return [i for r in rows for i in sorted(r, key=lambda k: info[k][1])]


class LPR:
    def __init__(self, det_path, rec_path, cfg_path, threads=1):
        self.cfg = json.load(open(cfg_path, encoding="utf-8"))
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.det = ort.InferenceSession(det_path, so, providers=["CPUExecutionProvider"])
        self.rec = ort.InferenceSession(rec_path, so, providers=["CPUExecutionProvider"])
        c = self.cfg
        self.S, self.rh, self.rw, self.chars = c["det_size"], c["rec_h"], c["rec_w"], c["chars"]

    # ---- DET ----
    def detect(self, img):
        H, W = img.shape[:2]; c = self.cfg
        x = cv2.resize(img, (self.S, self.S), interpolation=cv2.INTER_LINEAR)
        x = x.transpose(2, 0, 1)[None].astype(np.float32) * (1 / 255.0)
        prob = self.det.run(None, {"input": x})[0][0, 0]
        Mh, Mw = prob.shape
        cnts, _ = cv2.findContours((prob > c["det_thresh"]).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        out = []
        for cn in sorted(cnts, key=cv2.contourArea, reverse=True)[:6]:
            if len(cn) < 4 or min(cv2.minAreaRect(cn)[1]) < 1.5: continue
            bx, by, bw, bh = cv2.boundingRect(cn)
            m = np.zeros((bh, bw), np.uint8); cv2.fillPoly(m, [cn - np.array([bx, by], np.int32)], 1)
            score = cv2.mean(prob[by:by + bh, bx:bx + bw], m)[0]
            if score < c["det_box_thresh"]: continue
            q = fit_quad(cn, c["det_unclip"], c.get("det_pad_x", 0.0))
            if min(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[3] - q[0])) < 3.5: continue
            q[:, 0] = np.clip((q[:, 0] + 0.5) * W / Mw - 0.5, 0, W)
            q[:, 1] = np.clip((q[:, 1] + 0.5) * H / Mh - 0.5, 0, H)
            out.append((q, float(score)))
        return [out[k] for k in sort_lines([q for q, _ in out])]

    # ---- REC ----
    def crop(self, img, quad):
        tl, tr, br, bl = quad
        if self.cfg["rec_resize"] == "stretch":
            w, h = self.rw, self.rh
        else:
            qw = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)); qh = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
            w, h = int(np.clip(round(qw * self.rh / max(qh, 1)), 8, self.rw)), self.rh
        M = cv2.getPerspectiveTransform(quad.astype(np.float32), np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
        out = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if w < self.rw:
            l = (self.rw - w) // 2
            out = cv2.copyMakeBorder(out, 0, 0, l, self.rw - w - l, cv2.BORDER_CONSTANT, value=0)
        return out

    def recognize(self, crops):
        x = np.stack(crops).transpose(0, 3, 1, 2).astype(np.float32) * (1 / 255.0)
        prob = self.rec.run(None, {"input": x})[0]            # (B,T,C) đã softmax
        idx, mx = prob.argmax(-1), prob.max(-1)
        res = []
        for row, pr in zip(idx, mx):
            keep = (row != 0) & np.r_[True, row[1:] != row[:-1]]
            res.append(("".join(self.chars[k - 1] for k in row[keep]), float(pr[keep].mean()) if keep.any() else 0.0))
        return res

    def __call__(self, img):
        dets = self.detect(img)
        lines = self.recognize([self.crop(img, q) for q, _ in dets]) if dets else []
        return {"plate": "".join(t for t, _ in lines), "lines": lines, "boxes": [q for q, _ in dets],
                "conf": min([c for _, c in lines], default=0.0)}
