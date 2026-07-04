# cira_ttl_anomaly.py
# 
# Memory-bank anomaly detector with online test-time learning.
# 
# Front summary of the code flow
# ------------------------------
# 1. Library import and device setup
#    - Import OpenCV, PyTorch, NumPy, random utilities and Ultralytics YOLO.
#    - Select CUDA if available, otherwise use CPU.
# 
# 2. YOLOFeatureExtractor
#    - Load the YOLO classification model.
#    - Freeze all YOLO weights.
#    - Capture the image feature embedding before the final linear layer.
#    - YOLO is used only as a frozen feature extractor, not as a bounding-box detector.
# 
# 3. OnlineAdapter
#    - Define a lightweight linear adapter after the frozen YOLO feature extractor.
#    - Initialize the adapter as an identity mapping.
#    - Only this adapter is updated during online test-time learning.
# 
# 4. TTLAnomalyDetector initialization
#    - Load category-specific files:
#      memory_bank.pt, ttl_adapter.pt and threshold.json value.
#    - Set anomaly scoring parameters, update threshold and online learning settings.
#    - Prepare the optimizer for updating only the adapter.
# 
# 5. Image preprocessing
#    - Resize image, convert BGR to RGB, normalize pixel values and convert to tensor.
# 
# 6. Memory-bank comparison
#    - Compare the current image embedding with the normal memory bank.
#    - Select Top-K most similar normal references.
#    - Calculate anomaly score using Top-K reference score and global memory-bank score.
# 
# 7. Online adaptation
#    - Apply weak and strong augmentations to an accepted normal-like image.
#    - Update the adapter using consistency loss and anchor loss.
#    - Add accepted normal-like embeddings into the memory bank.
# 
# 8. Prediction
#    - Calculate anomaly score before possible update.
#    - Check whether the sample is safe for update using update_threshold.
#    - If accepted, update the adapter and memory bank.
#    - Recalculate anomaly score after possible update.
#    - Compare final anomaly score with anomaly threshold to output normal or anomaly.


# Image processing and deep learning libraries
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from ultralytics import YOLO


# Use GPU when available; otherwise use CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------
# Frozen YOLO feature extractor
# ------------------------------------------------------------
class YOLOFeatureExtractor(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        # Load YOLO classification model and set it to evaluation mode.
        self.yolo = YOLO(model_name)
        self.core = self.yolo.model.to(DEVICE).eval()

        # Freeze YOLO weights. The backbone is not retrained during deployment.
        for p in self.core.parameters():
            p.requires_grad = False

        # These variables are used to capture the feature before the final linear layer.
        self._cached_feat = None
        self.last_linear = None

        # Locate the final linear layer in the YOLO classification model.
        for m in self.core.modules():
            if isinstance(m, nn.Linear):
                self.last_linear = m

        if self.last_linear is None:
            raise RuntimeError("Final Linear layer not found in YOLO classification model.")

        # Forward hook captures the input to the final linear layer.
        # This captured tensor is treated as the image feature embedding.
        def hook(module, input, output):
            self._cached_feat = input[0].detach()

        self.last_linear.register_forward_hook(hook)

    @torch.no_grad()
    def extract(self, x):
        # Run YOLO forward pass. The hook stores the feature embedding.
        self._cached_feat = None
        _ = self.core(x)

        if self._cached_feat is None:
            raise RuntimeError("Feature hook failed. No feature was captured.")

        feat = self._cached_feat

        if len(feat.shape) > 2:
            feat = torch.flatten(feat, start_dim=1)

        # L2-normalize the feature so cosine similarity can be used reliably.
        feat = F.normalize(feat, dim=1)
        return feat


# ------------------------------------------------------------
# Lightweight online adapter
# ------------------------------------------------------------
class OnlineAdapter(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        # Linear projection keeps the same feature dimension.
        self.proj = nn.Linear(feat_dim, feat_dim, bias=False)

        # Start as identity mapping, so initial features remain unchanged.
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(feat_dim))

    def forward(self, x):
        x = self.proj(x)
        x = F.normalize(x, dim=1)
        return x


# ------------------------------------------------------------
# Main anomaly detector with online test-time learning
# ------------------------------------------------------------
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
        
        # Image size and anomaly threshold used for final normal/anomaly decision.
        self.img_size = int(img_size)
        self.threshold = float(threshold)
        
        # Update threshold is stricter than anomaly threshold by default.
        # It controls whether a sample is safe enough for online updating.
        self.update_threshold = (
            float(update_threshold)
            if update_threshold is not None
            else float(threshold) * float(accept_margin)
        )

        # Scoring settings: local Top-K score and global memory-bank score.
        self.top_k_references = int(top_k_references)
        self.reference_weight = float(reference_weight)
        self.global_weight = float(global_weight)
        self.accept_margin = float(accept_margin)

        # Online adaptation settings.
        self.online_lr = float(online_lr)
        self.max_memory_bank = int(max_memory_bank)
        self.online_steps = int(online_steps)
        self.consistency_weight = float(consistency_weight)
        self.anchor_weight = float(anchor_weight)

        # Frozen YOLO feature extractor.
        self.extractor = YOLOFeatureExtractor(model_name=model_name)

        # Load category-specific normal memory bank and normalize it.
        self.memory_bank = torch.load(memory_bank_path, map_location="cpu")
        self.memory_bank = F.normalize(self.memory_bank.float(), dim=1)

        feat_dim = self.memory_bank.shape[1]

        # Create adapter using the memory-bank feature dimension.
        self.adapter = OnlineAdapter(feat_dim).to(DEVICE)

        # Load category-specific adapter weights.
        checkpoint = torch.load(adapter_path, map_location=DEVICE)

        if isinstance(checkpoint, dict) and "adapter_state_dict" in checkpoint:
            self.adapter.load_state_dict(checkpoint["adapter_state_dict"])
        else:
            self.adapter.load_state_dict(checkpoint)

        # Optimizer updates only the adapter parameters during online learning.
        self.adapter.eval()
        self.optimizer = torch.optim.Adam(self.adapter.parameters(), lr=self.online_lr)

    def preprocess(self, img):
        """Prepare OpenCV image for YOLO feature extraction."""
        # Resize, convert BGR to RGB, normalize to [0, 1] and convert to tensor.
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        return x

    @torch.no_grad()
    def select_topk_references(self, embedding):
        """Select Top-K most similar normal references from the memory bank."""
        bank = self.memory_bank.to(embedding.device)

        # Cosine similarity because both embedding and memory bank are normalized.
        sim = embedding @ bank.T

        k = min(self.top_k_references, bank.shape[0])
        topk_sim, topk_idx = torch.topk(sim, k=k, dim=1)

        # topk_refs are used for local scoring and anchor loss.
        topk_refs = bank[topk_idx]
        return topk_refs, topk_sim

    @torch.no_grad()
    def open_ended_anomaly_score(self, embedding):
        """Calculate anomaly score using Top-K and global memory-bank comparison."""
        bank = self.memory_bank.to(embedding.device)

        # 1. Global memory-bank score
        # Measures similarity to the closest feature in the full memory bank.
        global_sim = embedding @ bank.T
        global_max_sim, _ = global_sim.max(dim=1)
        global_score = 1.0 - global_max_sim

        # 2. Top-K reference score
        # Measures average similarity to the selected Top-K normal references.
        _, topk_sim = self.select_topk_references(embedding)
        ref_sim = topk_sim.mean(dim=1)
        ref_score = 1.0 - ref_sim

        # 3. Final anomaly score
        # Higher score means the image is less similar to normal references.
        score = (
            self.reference_weight * ref_score
            + self.global_weight * global_score
        )

        return score

    @torch.no_grad()
    def nearest_normal_anchor(self, embedding):
        """Build an anchor feature from the mean of Top-K normal references."""
        topk_refs, _ = self.select_topk_references(embedding)
        anchor = topk_refs.mean(dim=1)
        anchor = F.normalize(anchor, dim=1)
        return anchor

    def weak_aug(self, x):
        """Apply weak augmentation for consistency learning."""
        noise = torch.randn_like(x) * 0.01
        out = torch.clamp(x + noise, 0.0, 1.0)
        return out

    def strong_aug(self, x):
        """Apply stronger augmentation for consistency learning."""
        noise = torch.randn_like(x) * 0.03
        out = torch.clamp(x + noise, 0.0, 1.0)

        if random.random() < 0.5:
            out = torch.flip(out, dims=[3])

        return out

    def online_update_open_ended(self, x):
        """Update only the adapter using accepted normal-like samples."""
        self.adapter.train()
        update_loss = None

        for _ in range(self.online_steps):
            # Create two augmented views of the same accepted sample.
            xw = self.weak_aug(x)
            xs = self.strong_aug(x)

            # YOLO remains frozen. Only features are extracted.
            with torch.no_grad():
                fw = self.extractor.extract(xw)
                fs = self.extractor.extract(xs)

            # Adapter produces the updated feature representations.
            zw = self.adapter(fw)
            zs = self.adapter(fs)

            # Anchor is the average of Top-K normal references.
            with torch.no_grad():
                anchor = self.nearest_normal_anchor(zw.detach()).to(DEVICE)

            # Consistency loss keeps augmented views close.
            # Anchor loss keeps the accepted sample close to normal references.
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
        """Run anomaly detection and optional online test-time update."""
        x = self.preprocess(img)
    
        # ============================================================
        # Step 1: Calculate anomaly score before possible online update
        # ============================================================
        with torch.no_grad():
            feat_before = self.extractor.extract(x)
            emb_before = self.adapter(feat_before)
            score_before = float(self.open_ended_anomaly_score(emb_before)[0].item())
    
        # ============================================================
        # Step 2: Decide whether this image is safe for online update
        # update_threshold = threshold × accept_margin by default.
        # ============================================================
        is_update_allowed = score_before < self.update_threshold
    
        update_loss = None
    
        # ============================================================
        # Step 3: Online test-time learning
        # Only confidently normal-like samples update the adapter and memory bank.
        # ============================================================
        if allow_update and is_update_allowed:
            update_loss = self.online_update_open_ended(x)
    
            with torch.no_grad():
                feat_updated = self.extractor.extract(x)
                emb_updated = self.adapter(feat_updated)
    
            # Add accepted normal-like embedding into the memory bank.
            self.memory_bank = torch.cat(
                [self.memory_bank, emb_updated.detach().cpu()],
                dim=0
            )
    
            # Keep only the most recent features when the memory bank exceeds the cap.
            if self.memory_bank.shape[0] > self.max_memory_bank:
                self.memory_bank = self.memory_bank[-self.max_memory_bank:]
    
        # ============================================================
        # Step 4: Recalculate anomaly score after possible online update
        # ============================================================
        with torch.no_grad():
            feat_after = self.extractor.extract(x)
            emb_after = self.adapter(feat_after)
            score_after = float(self.open_ended_anomaly_score(emb_after)[0].item())
    
        # ============================================================
        # Step 5: Final anomaly decision
        # The final decision uses the anomaly threshold, not the update threshold.
        # ============================================================
        is_anomaly = score_after >= self.threshold
    
        return {
            "label": "anomaly" if is_anomaly else "normal",
            "is_anomaly": bool(is_anomaly),
    
            "anomaly_score": score_after,
            "score_before": score_before,
    
            "allow_update": bool(allow_update),
            "threshold": self.threshold,
            "anomaly_threshold": self.threshold,
            "update_threshold": self.update_threshold,
    
            "updated_memory": bool(is_update_allowed),
            "memory_size": int(self.memory_bank.shape[0]),
            "update_loss": update_loss,
    
            "device": DEVICE,
            "top_k_references": self.top_k_references,
            "reference_weight": self.reference_weight,
            "global_weight": self.global_weight,
            "accept_margin": self.accept_margin
           
        }

    def predict_image(self, img):
        """Compatibility wrapper used by the Flask application."""
        return self.predict(img)