import cv2
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class YOLOPatchExtractor(nn.Module):
    """Extract rev1.8 local YOLO26 patch features."""
    def __init__(self, model_name, patch_grid=14, feature_choice="last2"):
        super().__init__()
        choices = {"last1": [-1], "last2": [-2, -1], "last3": [-3, -2, -1]}
        if feature_choice not in choices:
            raise ValueError(f"Unsupported feature_choice: {feature_choice}")
        self.feature_indices = choices[feature_choice]
        self.patch_grid = int(patch_grid)
        self.yolo = YOLO(model_name)
        self.core = self.yolo.model.to(DEVICE).eval()
        for p in self.core.parameters():
            p.requires_grad = False
        self.local_cache = []
        for layer in list(self.core.model)[-12:]:
            layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if torch.is_tensor(output) and output.ndim == 4 and min(output.shape[-2:]) >= 4:
            self.local_cache.append(output)

    @torch.no_grad()
    def extract(self, x):
        self.local_cache = []
        _ = self.core(x)
        need = max(abs(i) for i in self.feature_indices)
        if len(self.local_cache) < need:
            raise RuntimeError(f"Not enough usable YOLO feature maps: {len(self.local_cache)}")
        parts = []
        for i in self.feature_indices:
            fmap = F.interpolate(
                self.local_cache[i],
                size=(self.patch_grid, self.patch_grid),
                mode="bilinear",
                align_corners=False,
            )
            parts.append(fmap.permute(0, 2, 3, 1))
        return F.normalize(torch.cat(parts, dim=-1), dim=-1)

class PatchAdapter(nn.Module):
    """Identity-initialized lightweight patch adapter."""
    def __init__(self, feat_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(feat_dim))
        self.bias = nn.Parameter(torch.zeros(feat_dim))

    def forward(self, x):
        return F.normalize(x * self.scale + self.bias, dim=-1)

class TTLAnomalyDetector:
    """rev1.8 patch detector with optional online CTTA updates."""
    def __init__(
        self,
        patch_memory_bank_path=None,
        threshold=999.0,
        model_name=None,
        img_size=384,
        feature_choice="last2",
        patch_grid=14,
        patch_top_fraction=0.05,
        update_threshold=None,
        accept_margin=0.95,
        online_lr=1e-4,
        online_steps=1,
        max_patch_memory=16000,
        consistency_weight=1.0,
        anchor_weight=0.1,
        patch_adapter_path=None,
        memory_bank_path=None,
        adapter_path=None,
        max_memory_bank=None,
        consistency_threshold=0.002,
        adapter_update_enabled=True,
        memory_update_enabled=True,
        **unused_kwargs,
    ):
        # Accept both new and old parameter names.
        if patch_memory_bank_path is None:
            patch_memory_bank_path = memory_bank_path
        if patch_adapter_path is None:
            patch_adapter_path = adapter_path
        if max_memory_bank is not None:
            max_patch_memory = max_memory_bank

        if not patch_memory_bank_path:
            raise ValueError("patch_memory_bank_path is required.")
        if not model_name:
            raise ValueError("model_name is required.")

        self.device = DEVICE
        self.img_size = int(img_size)
        self.threshold = float(threshold)
        self.accept_margin = float(accept_margin)
        self.update_threshold = float(update_threshold) if update_threshold is not None else self.threshold * self.accept_margin
        self.feature_choice = str(feature_choice)
        self.patch_grid = int(patch_grid)
        self.patch_top_fraction = float(patch_top_fraction)
        self.online_lr = float(online_lr)
        self.online_steps = int(online_steps)
        self.max_patch_memory = int(max_patch_memory)
        self.consistency_weight = float(consistency_weight)
        self.anchor_weight = float(anchor_weight)
        self.consistency_threshold = float(consistency_threshold)
        self.adapter_update_enabled = bool(adapter_update_enabled)
        self.memory_update_enabled = bool(memory_update_enabled)
        
        self.extractor = YOLOPatchExtractor(model_name, self.patch_grid, self.feature_choice)

        bank = torch.load(patch_memory_bank_path, map_location="cpu")
        if isinstance(bank, dict):
            bank = bank.get("patch_memory_bank", bank.get("memory_bank", bank))
        if not torch.is_tensor(bank) or bank.ndim != 2:
            raise ValueError("patch_memory_bank.pt must contain a 2D tensor.")
        self.patch_memory_bank = F.normalize(bank.float(), dim=1)

        feat_dim = int(self.patch_memory_bank.shape[1])
        self.patch_adapter = PatchAdapter(feat_dim).to(DEVICE)

        # A renamed old ttl_adapter.pt is not compatible. Ignore it safely.
        if patch_adapter_path and Path(patch_adapter_path).exists():
            try:
                checkpoint = torch.load(patch_adapter_path, map_location=DEVICE)
                if isinstance(checkpoint, dict) and "patch_adapter_state_dict" in checkpoint:
                    checkpoint = checkpoint["patch_adapter_state_dict"]
                self.patch_adapter.load_state_dict(checkpoint)
                print(f"Loaded patch adapter: {patch_adapter_path}")
            except Exception as exc:
                print("WARNING: incompatible patch_adapter.pt ignored.")
                print(f"Reason: {exc}")
                print("Using fresh identity patch adapter.")

        self.patch_adapter.eval()
        self.optimizer = torch.optim.Adam(self.patch_adapter.parameters(), lr=self.online_lr)

    @property
    def memory_bank(self):
        return self.patch_memory_bank

    def preprocess(self, img):
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return torch.tensor(img).unsqueeze(0).float().to(DEVICE)

    @torch.no_grad()
    def _patches(self, x):
        return self.patch_adapter(self.extractor.extract(x))

    @torch.no_grad()
    def patch_anomaly_score(self, patches):
        q = F.normalize(patches.reshape(-1, patches.shape[-1]).float(), dim=1)
        bank = self.patch_memory_bank.to(q.device)
        dist = 1.0 - (q @ bank.T).max(dim=1).values
        k = max(1, int(np.ceil(len(dist) * self.patch_top_fraction)))
        return torch.topk(dist, k=k, largest=True).values.mean()

    def weak_aug(self, x):
        """Small perturbation for consistency learning."""
        noise = torch.randn_like(x) * 0.01
        return torch.clamp(x + noise, 0.0, 1.0)

    def strong_aug(self, x):
        """Stronger perturbation while preserving patch alignment."""
        noise = torch.randn_like(x) * 0.03
        return torch.clamp(x + noise, 0.0, 1.0)

    @torch.no_grad()
    def nearest_patch_anchors(self, patches):
        """Find nearest normal-memory anchor for each adapted patch."""
        q = F.normalize(
            patches.reshape(-1, patches.shape[-1]).float(),
            dim=1,
        )
        bank = self.patch_memory_bank.to(q.device)
        sim = q @ bank.T
        idx = sim.argmax(dim=1)
        anchors = bank[idx]
        return anchors.reshape_as(patches)

    def score_only(self, img):
        x = self.preprocess(img)
        with torch.no_grad():
            return float(self.patch_anomaly_score(self._patches(x)).item())

    def online_update(self, x):
        self.patch_adapter.train()
    
        update_loss = None
    
        for _ in range(self.online_steps):
    
            x_weak = self.weak_aug(x)
            x_strong = self.strong_aug(x)
    
            with torch.no_grad():
                weak_feature = self.extractor.extract(x_weak)
                strong_feature = self.extractor.extract(x_strong)
    
            weak_patch = self.patch_adapter(weak_feature)
            strong_patch = self.patch_adapter(strong_feature)
    
            with torch.no_grad():
                anchor = self.nearest_patch_anchors(
                    weak_patch.detach()
                )
    
            consistency_loss = F.mse_loss(
                strong_patch,
                weak_patch.detach()
            )
    
            anchor_loss = F.mse_loss(
                weak_patch,
                anchor
            )
    
            loss = (
                self.consistency_weight * consistency_loss
                + self.anchor_weight * anchor_loss
            )
    
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
            update_loss = float(loss.item())
    
        self.patch_adapter.eval()
    
        return update_loss

    @torch.no_grad()
    def measure_consistency(self, x):
        x_weak = self.weak_aug(x)
        x_strong = self.strong_aug(x)
    
        weak_feature = self.extractor.extract(x_weak)
        strong_feature = self.extractor.extract(x_strong)
    
        weak_patch = self.patch_adapter(weak_feature)
        strong_patch = self.patch_adapter(strong_feature)
    
        consistency_error = float(
            F.mse_loss(
                weak_patch,
                strong_patch
            ).item()
        )
    
        return consistency_error

    def predict(self, img, allow_update=True):
        x = self.preprocess(img)

        with torch.no_grad():
            patches = self._patches(x)
            score_before = float(
                self.patch_anomaly_score(patches).item()
            )

        updated_memory = False
        update_loss = None
        adapter_updated = False
        adapter_delta_norm = 0.0

        consistency_error = self.measure_consistency(x)
        
        score_gate_pass = (
            score_before < self.update_threshold
        )
        
        consistency_gate_pass = (
            consistency_error < self.consistency_threshold
        )

        # TEMPORARY: disable online learning during
        # trusted-normal consistency calibration.
        # update_allowed = False
        
        update_allowed = (
            allow_update
            and score_gate_pass
            and consistency_gate_pass
        )

        if update_allowed:
            # Adapter-only or Full CTTA
            if self.adapter_update_enabled:
                before = torch.cat([
                    p.detach().flatten()
                    for p in self.patch_adapter.parameters()
                ]).clone()

                update_loss = self.online_update(x)

                after = torch.cat([
                    p.detach().flatten()
                    for p in self.patch_adapter.parameters()
                ])

                adapter_delta_norm = float(
                    torch.norm(after - before).item()
                )
                adapter_updated = adapter_delta_norm > 0.0

            # Memory-only or Full CTTA
            if self.memory_update_enabled:
                with torch.no_grad():
                    updated_patches = self._patches(x)
                    new_patches = updated_patches.reshape(
                        -1,
                        updated_patches.shape[-1],
                    ).cpu()

                self.patch_memory_bank = torch.cat(
                    [self.patch_memory_bank, new_patches],
                    dim=0,
                )

                if len(self.patch_memory_bank) > self.max_patch_memory:
                    self.patch_memory_bank = self.patch_memory_bank[
                        -self.max_patch_memory:
                    ]

                self.patch_memory_bank = F.normalize(
                    self.patch_memory_bank.float(),
                    dim=1,
                )
                updated_memory = True

        with torch.no_grad():
            score_after = float(
                self.patch_anomaly_score(
                    self._patches(x)
                ).item()
            )

        is_anomaly = score_after >= self.threshold

        return {
            "label": "anomaly" if is_anomaly else "normal",
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": score_after,
            "score_before": score_before,
            "threshold": self.threshold,
            "anomaly_threshold": self.threshold,
            "update_threshold": self.update_threshold,
            "update_allowed": bool(update_allowed),
            "updated_memory": bool(updated_memory),
            "memory_size": int(self.patch_memory_bank.shape[0]),
            "update_loss": update_loss,
            "adapter_updated": bool(adapter_updated),
            "adapter_delta_norm": adapter_delta_norm,
            "device": DEVICE,
            "score_method": "patch_nearest_normal",
            "feature_choice": self.feature_choice,
            "patch_grid": self.patch_grid,
            "patch_top_fraction": self.patch_top_fraction,
            "consistency_error": consistency_error,
            "consistency_threshold": self.consistency_threshold,
            "score_gate_pass": bool(score_gate_pass),
            "consistency_gate_pass": bool(consistency_gate_pass),
            "adapter_update_enabled": bool(self.adapter_update_enabled),
            "memory_update_enabled": bool(self.memory_update_enabled),
        }

    def save_patch_memory_bank(self, path):
        torch.save(self.patch_memory_bank.cpu(), path)

    def save_memory_bank(self, path):
        self.save_patch_memory_bank(path)

    def save_patch_adapter(self, path):
        torch.save(
            {"patch_adapter_state_dict": self.patch_adapter.state_dict()},
            path,
        )
