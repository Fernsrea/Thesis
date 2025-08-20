from typing import Optional
import torch, torch.nn as nn, torch.nn.functional as F
import pytorch_lightning as pl

class VAELightning(nn.Module):
    def __init__(self, ckpt_path: str, device="cuda"):
        super().__init__()
        # load checkpoint
        ckpt = torch.load("vavae-imagenet256-f16d32-dinov2.pt", map_location="cpu")
        
        # build model with same config (f=16, d=32, img_size=256)
        vae = ViTVAE(img_size=256, latent_dim=32, patch_size=16)
        
        # load weights
        vae.load_state_dict(ckpt["state_dict"], strict=True)
        vae.eval()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x ∈ [-1,1], shape [B,3,256,256]
        return self.model.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.model.decode(z)

class DummyDiT(nn.Module):
    def __init__(self, in_ch, model_dim=768):
        super().__init__()
        # project latent to model_dim; time & cond embeddings are added
        self.in_proj = nn.Conv2d(in_ch, model_dim, 1)
        self.time_mlp = nn.Sequential(nn.Linear(1, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim))
        self.blocks = nn.Sequential(*[nn.Conv2d(model_dim, model_dim, 3, 1, 1) for _ in range(6)])
        self.out = nn.Conv2d(model_dim, in_ch-1, 1)  # predict epsilon for z (latent channels), not mask

    def forward(self, zt, t_emb, cond_embed):
        h = self.in_proj(zt) + cond_embed + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.blocks(h)
        return self.out(h)

def timestep_embedding(t, device):
    return torch.tensor(t, dtype=torch.float32, device=device).view(-1, 1) / 1000.0

class InpaintLightningModule(pl.LightningModule):
    def __init__(self,
                 lr=1e-4,
                 weight_decay=1e-2,
                 ema_decay=0.999,
                 latent_channels=4,
                 vae_scale=8,
                 model_dim=768):
        super().__init__()
        self.save_hyperparameters()

        # replace these with actual VAE/DiT from LightningDiT in your env
        self.vae = VAELightning("vavae-imagenet256-f16d32-dinov2.pt", device=self.device)
        # conditioning: [z_masked(C) + mask(1)] => (C+1)
        self.cond_adapter = nn.Conv2d(latent_channels + 1, model_dim, 1)
        self.dit = DummyDiT(in_ch=latent_channels + 1, model_dim=model_dim)  # in_ch for zt plus (we reuse channels)

        # simple cosine noise schedule params
        self.register_buffer("betas", torch.linspace(1e-4, 0.02, 1000), persistent=False)
        alphas = 1.0 - self.betas
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0), persistent=False)

    # --- diffusion helpers ---
    def q_sample(self, z0, t, noise):
        ac = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return torch.sqrt(ac) * z0 + torch.sqrt(1.0 - ac) * noise

   def training_step(self, batch, batch_idx):
        x, m, x_m = batch["image"], batch["mask"], batch["masked"]
        
        # Encode with pretrained VAE
        with torch.no_grad():
            z0 = self.vae.encode(x)
            z_m = self.vae.encode(x_m)
        
        # Downsample mask to latent size
        m_lat = F.interpolate(m, size=z0.shape[-2:], mode="nearest")
        
        # Diffusion forward process
        t = torch.randint(0, self.num_timesteps, (x.size(0),), device=self.device)
        eps = torch.randn_like(z0)
        zt = self.q_sample(z0, t, eps)
        
        # Conditioning
        cond = torch.cat([z_m, m_lat], dim=1)
        cond_embed = self.cond_adapter(cond)
        
        # Predict noise
        t_emb = timestep_embedding(t, self.device)
        eps_hat = self.dit(zt, t_emb, cond_embed)
        
        # Loss
        w = 0.7 * m_lat + 0.3
        loss = ((eps_hat - eps) ** 2 * w).mean()
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, _):
        x, m, x_m = batch["image"], batch["mask"], batch["masked"]
        with torch.no_grad():
            z0 = self.vae.encode(x)
            z_m = self.vae.encode(x_m)
            m_lat = F.interpolate(m, size=z0.shape[-2:], mode="nearest")

            # single-step eval of noise loss
            t = torch.full((x.size(0),), 500, device=self.device, dtype=torch.long)
            eps = torch.randn_like(z0)
            zt = self.q_sample(z0, t, eps)
            cond = torch.cat([z_m, m_lat], dim=1)
            cond_embed = self.cond_adapter(cond)
            t_emb = timestep_embedding(t, self.device)
            eps_hat = self.dit(torch.cat([zt, m_lat], dim=1), t_emb, cond_embed)
            loss = F.mse_loss(eps_hat, eps)

            # quick qualitative decode: one DDIM-like step (placeholder)
            z_pred = zt - eps_hat  # NOT a proper sampler; replace with your sampler later
            x_pred = self.vae.decode(z_pred)
            # compute masked PSNR
            mse = ((x_pred - x) ** 2 * m).sum() / (m.sum() * x.size(1) + 1e-8)
            psnr = -10.0 * torch.log10(mse + 1e-8)

        self.log_dict({"val/loss": loss, "val/psnr_mask": psnr}, prog_bar=True)
        return {"loss": loss}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
