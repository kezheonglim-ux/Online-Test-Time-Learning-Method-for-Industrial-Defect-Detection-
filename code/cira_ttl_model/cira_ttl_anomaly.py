import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from ultralytics import YOLO


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class YOLOFeatureExtractor(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.yolo = YOLO(model_name)
        self.core = self.yolo.model.to(DEVICE).eval()

        for p in self.core.parameters():
            p.requires_grad = False

        self._cached_feat = None
        self.last_linear = None

        for m in self.core.modules():
            if isinstance(m, nn.Linear):
                self.last_linear = m

        if self.last_linear is None:
            raise RuntimeError("Final Linear layer not found in YOLO classification model.")

        def hook(module, input, output):
            self._cached_feat = input[0].detach()

        self.last_linear.register_forward_hook(hook)

    @torch.no_grad()
    def extract(self, x):
        self._cached_feat = None
        _ = self.core(x)

        if self._cached_feat is None:
            raise RuntimeError("Feature hook failed. No feature was captured.")

        feat = self._cached_feat

        if len(feat.shape) > 2:
            feat = torch.flatten(feat, start_dim=1)

        feat = F.normalize(feat, dim=1)
        return feat


class OnlineAdapter(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.proj = nn.Linear(feat_dim, feat_dim, bias=False)

        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(feat_dim))

    def forward(self, x):
        x = self.proj(x)
        x = F.normalize(x, dim=1)
        return x


class TTLAnomalyDetector:
    def __init__(
        self,
        adapter_path,
        memory_bank_path,
        threshold,
        model_name,
        img_size=224,
        top_k_references=5,
        reference_weight=0.7,
        global_weight=0.3,
        accept_margin=0.95,
        update_threshold=None,
        online_lr=1e-4,
        max_memory_bank=4000,
        online_steps=1,
        consistency_weight=1.0,
        anchor_weight=0.1
    ):
        
        self.img_size = int(img_size)
        self.threshold = float(threshold)
        
        self.update_threshold = (
            float(update_threshold)
            if update_threshold is not None
            else float(threshold) * float(accept_margin)
        )

        self.top_k_references = int(top_k_references)
        self.reference_weight = float(reference_weight)
        self.global_weight = float(global_weight)
        self.accept_margin = float(accept_margin)

        self.online_lr = float(online_lr)
        self.max_memory_bank = int(max_memory_bank)
        self.online_steps = int(online_steps)
        self.consistency_weight = float(consistency_weight)
        self.anchor_weight = float(anchor_weight)

        self.extractor = YOLOFeatureExtractor(model_name=model_name)

        self.memory_bank = torch.load(memory_bank_path, map_location="cpu")
        self.memory_bank = F.normalize(self.memory_bank.float(), dim=1)

        feat_dim = self.memory_bank.shape[1]

        self.adapter = OnlineAdapter(feat_dim).to(DEVICE)

        checkpoint = torch.load(adapter_path, map_location=DEVICE)

        if isinstance(checkpoint, dict) and "adapter_state_dict" in checkpoint:
            self.adapter.load_state_dict(checkpoint["adapter_state_dict"])
        else:
            self.adapter.load_state_dict(checkpoint)

        self.adapter.eval()
        self.optimizer = torch.optim.Adam(self.adapter.parameters(), lr=self.online_lr)

    def preprocess(self, img):
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        return x

    @torch.no_grad()
    def select_topk_references(self, embedding):
        bank = self.memory_bank.to(embedding.device)

        sim = embedding @ bank.T

        k = min(self.top_k_references, bank.shape[0])
        topk_sim, topk_idx = torch.topk(sim, k=k, dim=1)

        topk_refs = bank[topk_idx]
        return topk_refs, topk_sim

    @torch.no_grad()
    def open_ended_anomaly_score(self, embedding):
        bank = self.memory_bank.to(embedding.device)

        # 1. Global memory-bank score
        global_sim = embedding @ bank.T
        global_max_sim, _ = global_sim.max(dim=1)
        global_score = 1.0 - global_max_sim

        # 2. Top-K reference score
        _, topk_sim = self.select_topk_references(embedding)
        ref_sim = topk_sim.mean(dim=1)
        ref_score = 1.0 - ref_sim

        # 3. Same formula as notebook
        score = (
            self.reference_weight * ref_score
            + self.global_weight * global_score
        )

        return score

    @torch.no_grad()
    def nearest_normal_anchor(self, embedding):
        topk_refs, _ = self.select_topk_references(embedding)
        anchor = topk_refs.mean(dim=1)
        anchor = F.normalize(anchor, dim=1)
        return anchor

    def weak_aug(self, x):
        noise = torch.randn_like(x) * 0.01
        out = torch.clamp(x + noise, 0.0, 1.0)
        return out

    def strong_aug(self, x):
        noise = torch.randn_like(x) * 0.03
        out = torch.clamp(x + noise, 0.0, 1.0)

        if random.random() < 0.5:
            out = torch.flip(out, dims=[3])

        return out

    def online_update_open_ended(self, x):
        self.adapter.train()
        update_loss = None

        for _ in range(self.online_steps):
            xw = self.weak_aug(x)
            xs = self.strong_aug(x)

            with torch.no_grad():
                fw = self.extractor.extract(xw)
                fs = self.extractor.extract(xs)

            zw = self.adapter(fw)
            zs = self.adapter(fs)

            with torch.no_grad():
                anchor = self.nearest_normal_anchor(zw.detach()).to(DEVICE)

            consistency_loss = F.mse_loss(zs, zw.detach())
            anchor_loss = F.mse_loss(zw, anchor)

            loss = (
                self.consistency_weight * consistency_loss
                + self.anchor_weight * anchor_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            update_loss = float(loss.item())

        self.adapter.eval()
        return update_loss

    def score_only(self, img):
        """
        Calculate anomaly score without online update.
        Used for deployment auto-calibration.
        """
        x = self.preprocess(img)
    
        with torch.no_grad():
            feat = self.extractor.extract(x)
            emb = self.adapter(feat)
            score = float(self.open_ended_anomaly_score(emb)[0].item())
    
        return score    

    def predict(self, img, allow_update=True):
        x = self.preprocess(img)
    
        # ============================================================
        # Step 1 — Score before online update
        # ============================================================
        with torch.no_grad():
            feat_before = self.extractor.extract(x)
            emb_before = self.adapter(feat_before)
            score_before = float(self.open_ended_anomaly_score(emb_before)[0].item())
    
        # ============================================================
        # Step 2 — Decide whether this sample can update memory
        # ============================================================
        is_update_allowed = (
            allow_update and
            score_before < self.update_threshold
        )
    
        update_loss = None
    
        # ============================================================
        # Step 3 — Online test-time learning
        # ============================================================
        if is_update_allowed:
            update_loss = self.online_update_open_ended(x)
    
            with torch.no_grad():
                feat_updated = self.extractor.extract(x)
                emb_updated = self.adapter(feat_updated)
    
            self.memory_bank = torch.cat(
                [self.memory_bank, emb_updated.detach().cpu()],
                dim=0
            )
    
            if self.memory_bank.shape[0] > self.max_memory_bank:
                self.memory_bank = self.memory_bank[-self.max_memory_bank:]
    
        # ============================================================
        # Step 4 — Score after possible online update
        # ============================================================
        with torch.no_grad():
            feat_after = self.extractor.extract(x)
            emb_after = self.adapter(feat_after)
            score_after = float(self.open_ended_anomaly_score(emb_after)[0].item())
    
        # ============================================================
        # Step 5 — Final anomaly decision
        # ============================================================
        is_anomaly = score_after >= self.threshold
    
        return {
            "label": "anomaly" if is_anomaly else "normal",
            "is_anomaly": bool(is_anomaly),
    
            "anomaly_score": score_after,
            "score_before": score_before,
    
            "threshold": self.threshold,
            "anomaly_threshold": self.threshold,
            "update_threshold": self.update_threshold,
    
            "allow_update": bool(allow_update),
            "updated_memory": bool(is_update_allowed),
            "memory_size": int(self.memory_bank.shape[0]),
            "update_loss": update_loss,
    
            "device": DEVICE,
            "top_k_references": self.top_k_references,
            "reference_weight": self.reference_weight,
            "global_weight": self.global_weight,
            "accept_margin": self.accept_margin
        }

    def save_memory_bank(self, save_path):
        torch.save(self.memory_bank.cpu(), save_path)

    def predict_image(self, img):
        return self.predict(img)