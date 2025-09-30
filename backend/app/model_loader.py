import time, json, os
import numpy as np
import torch
import torch.nn as nn
import joblib


class CosAE(nn.Module):
    def __init__(self, d=30):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.LeakyReLU(), nn.Linear(64, d)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class ModelSvc:
    def __init__(self):
        self.scaler = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.version = "v1.0.0"

    def load_from_dir(self, path="models/v1.0"):
        # 필수 아티팩트 로딩
        self.scaler = joblib.load(os.path.join(path, "scaler.pkl"))
        ae_state = torch.load(os.path.join(path, "ae_state.pt"), map_location="cpu")
        self.model = CosAE(d=30).to(self.device)
        self.model.load_state_dict(ae_state)
        self.model.eval()
        # model_card에서 버전·권고 threshold 로드(있으면)
        card_path = os.path.join(path, "model_card.json")
        if os.path.exists(card_path):
            with open(card_path, "r", encoding="utf-8") as f:
                card = json.load(f)
            self.version = card.get("version", self.version)
        return True

    def predict_batch(self, X_np, threshold):
        t0 = time.time()
        # X_np: (N,30) float32
        Xs = self.scaler.transform(X_np)
        X_t = torch.tensor(Xs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            recon = self.model(X_t)
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            cosine_sim = cos(X_t, recon).detach().cpu().numpy()
        scores = 1.0 - cosine_sim
        is_abn = scores >= threshold
        latency_ms = int((time.time() - t0) * 1000)
        return scores.astype(float), is_abn.astype(bool), latency_ms


# 초기화 도우미
model_svc = ModelSvc()
