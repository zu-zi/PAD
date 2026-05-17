"""
PAD model: CLIP ViT-B/16 with TA-Prompt (text side) and VA-Prompt (visual side).

Components:
    * ``VAPromptPool``    -- per-layer G/E prompt pool, PAD Sec. 3.4.2.
    * ``TAPromptLearner`` -- class-conditional ``"A photo of a X X X X person."`` tokens.
    * ``TextEncoder``     -- frozen CLIP text encoder.
    * ``PADModel``        -- glues the above together and owns the classification heads.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from timm.layers import trunc_normal_

from .clip import clip


# ===========================================================================
# VA-Prompt (PAD Sec. 3.4.2)
# ===========================================================================
def _l2(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(p=2, dim=dim, keepdim=True).clamp(min=eps)


class VAPromptPool(nn.Module):
    """Per-layer ``{G-Prompt, E-Prompt pool}`` with lifelong slot allocation."""

    def __init__(self, num_layers: int, embed_dim: int = 768,
                 g_len: int = 6, e_len: int = 6, pool_size: int = 36, top_k: int = 4,
                 embedding_key: str = "mean", key_learnable: bool = True):
        super().__init__()
        assert num_layers > 0 and 0 < top_k <= pool_size
        assert embedding_key in ("mean", "cls")
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.g_len = g_len
        self.e_len = e_len
        self.pool_size = pool_size
        self.top_k = top_k
        self.embedding_key = embedding_key
        self.key_learnable = key_learnable

        self.g_prompt = nn.ParameterList(
            nn.Parameter(self._init((g_len, embed_dim))) for _ in range(num_layers)
        )
        self.e_prompt = nn.ParameterList(
            nn.Parameter(self._init((pool_size, e_len, embed_dim))) for _ in range(num_layers)
        )
        if key_learnable:
            self.e_key = nn.ParameterList(
                nn.Parameter(self._init((pool_size, embed_dim))) for _ in range(num_layers)
            )
        else:
            self.e_key = nn.ParameterList(
                nn.Parameter(torch.empty(0), requires_grad=False) for _ in range(num_layers)
            )

        # Lifelong bookkeeping buffers.
        self.register_buffer("active_sizes", torch.zeros(num_layers, 1, dtype=torch.long))
        self.register_buffer("frozen_masks", torch.zeros(num_layers, pool_size, dtype=torch.bool))
        self.register_buffer("trainable_masks", torch.zeros(num_layers, pool_size, dtype=torch.bool))

    @staticmethod
    def _init(shape) -> torch.Tensor:
        t = torch.zeros(shape)
        nn.init.uniform_(t, a=-0.02, b=0.02)
        return t

    def _query(self, x_noprompt: torch.Tensor, cls_feat: Optional[torch.Tensor]) -> torch.Tensor:
        if self.embedding_key == "mean":
            q = x_noprompt.mean(dim=1)
        else:
            q = cls_feat if cls_feat is not None else x_noprompt[:, 0, :]
        return _l2(q, dim=-1)

    def _keys(self, layer_id: int) -> torch.Tensor:
        k = self.e_key[layer_id] if self.key_learnable else self.e_prompt[layer_id].mean(dim=1)
        return _l2(k, dim=-1)

    @torch.no_grad()
    def _topk(self, layer_id: int, q: torch.Tensor) -> torch.Tensor:
        keys = self._keys(layer_id)
        used = int(self.active_sizes[layer_id, 0].item())
        if used <= 0:
            used = min(self.pool_size, self.top_k)
        sim = q @ keys[:used].t()
        k_eff = min(self.top_k, used)
        _, idx = torch.topk(sim, k=k_eff, dim=1)
        return idx

    def forward_for_layer(self, layer_id: int, x_noprompt: torch.Tensor,
                          cls_feat: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B, _, C = x_noprompt.shape
        device, dtype = x_noprompt.device, x_noprompt.dtype
        q = self._query(x_noprompt, cls_feat)
        idx = self._topk(layer_id, q)
        ep = self.e_prompt[layer_id][idx].reshape(B, -1, C).to(device=device, dtype=dtype)
        gp = self.g_prompt[layer_id].unsqueeze(0).expand(B, -1, -1).to(device=device, dtype=dtype)
        batched = torch.cat([gp, ep], dim=1)
        return {"batched": batched, "strip_len": batched.size(1), "topk_idx": idx}

    @torch.no_grad()
    def allocate_new_domain_slots(self, slots_per_layer: int) -> None:
        """Freeze previously-used slots and open ``slots_per_layer`` fresh ones."""
        for li in range(self.num_layers):
            used_before = int(self.active_sizes[li, 0].item())
            used_after = min(used_before + slots_per_layer, self.pool_size)
            frozen = self.frozen_masks[li].clone()
            if used_before > 0:
                frozen[:used_before] = True
            self.frozen_masks[li] = frozen
            trainable = torch.zeros_like(self.trainable_masks[li])
            if used_after > used_before:
                trainable[used_before:used_after] = True
            self.trainable_masks[li] = trainable
            self.active_sizes[li, 0] = used_after

    def extra_repr(self) -> str:
        return (
            f"L={self.num_layers} C={self.embed_dim} "
            f"G={self.g_len} E={self.e_len} pool={self.pool_size} topk={self.top_k}"
        )


# ===========================================================================
# Text branch (TA-Prompt + frozen CLIP text encoder)
# ===========================================================================
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)                 # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)                 # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        eot = tokenized_prompts.argmax(dim=-1)
        return x[torch.arange(x.shape[0]), eot] @ self.text_projection


class TAPromptLearner(nn.Module):
    """Class-conditional ``"A photo of a X X X X person."`` tokens (PAD Sec. 3.3)."""

    def __init__(self, num_class: int, dtype, token_embedding: nn.Embedding,
                 n_ctx: int = 4, n_cls_ctx: int = 4):
        super().__init__()
        ctx_init = "A photo of a " + " ".join(["X"] * n_cls_ctx) + " person."
        ctx_dim = 512

        tokenized = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized).type(dtype)
        self.tokenized_prompts = tokenized

        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        self.register_buffer("token_prefix", embedding[:, : n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx :, :])
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label: torch.Tensor) -> torch.Tensor:
        cls_ctx = self.cls_ctx[label]
        b = label.shape[0]
        return torch.cat(
            [self.token_prefix.expand(b, -1, -1), cls_ctx, self.token_suffix.expand(b, -1, -1)],
            dim=1,
        )


# ===========================================================================
# initialisers
# ===========================================================================
def _init_kaiming(m: nn.Module):
    cls = m.__class__.__name__
    if cls.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)
    elif cls.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif cls.find("BatchNorm") != -1 and m.affine:
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


def _init_classifier(m: nn.Module):
    if m.__class__.__name__.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def _load_clip(backbone: str, h_res: int, w_res: int, stride: int):
    url = clip._MODELS[backbone]
    path = clip._download(url)
    try:
        model = torch.jit.load(path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(path, map_location="cpu")
    return clip.build_model(state_dict or model.state_dict(), h_res, w_res, stride)


# ===========================================================================
# main model
# ===========================================================================
class PADModel(nn.Module):
    def __init__(self, num_classes: int, cfg):
        super().__init__()
        if cfg.MODEL.NAME != "ViT-B-16":
            raise ValueError(f"PAD currently supports ViT-B/16 only (got {cfg.MODEL.NAME}).")
        self.model_name = cfg.MODEL.NAME
        self.num_classes = num_classes
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.in_planes_proj = 512

        # Visual heads.
        self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
        self.classifier.apply(_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, num_classes, bias=False)
        self.classifier_proj.apply(_init_classifier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(_init_kaiming)

        # CLIP backbone + VA-Prompt pool.
        h_res = (cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1
        w_res = (cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1
        clip_model = _load_clip(self.model_name, h_res, w_res, cfg.MODEL.STRIDE_SIZE[0])
        clip_model.to("cuda")
        self.image_encoder = clip_model.visual

        if cfg.VA_PROMPT.ENABLE:
            num_layers = int(self.image_encoder.transformer.layers)
            self.image_encoder.va_prompt = VAPromptPool(
                num_layers=num_layers,
                embed_dim=self.in_planes,
                g_len=int(cfg.VA_PROMPT.G_LEN),
                e_len=int(cfg.VA_PROMPT.E_LEN),
                pool_size=int(cfg.VA_PROMPT.POOL_SIZE),
                top_k=int(cfg.VA_PROMPT.TOP_K),
                embedding_key=str(cfg.VA_PROMPT.EMBEDDING_KEY),
                key_learnable=bool(cfg.VA_PROMPT.KEY_LEARNABLE),
            ).to(next(self.image_encoder.parameters()).device)
            self.image_encoder.va_strip_after_block = bool(cfg.VA_PROMPT.STRIP_AFTER_BLOCK)

        # Text branch.
        self.prompt_learner = TAPromptLearner(
            num_classes, clip_model.dtype, clip_model.token_embedding,
            n_ctx=int(cfg.TA_PROMPT.N_CTX),
            n_cls_ctx=int(cfg.TA_PROMPT.N_CLS_CTX),
        )
        self.text_encoder = TextEncoder(clip_model)

    def forward(self, x: torch.Tensor = None, label: torch.Tensor = None,
                get_image: bool = False, get_text: bool = False,
                return_train_outputs: bool = False):
        if get_text:
            prompts = self.prompt_learner(label)
            return self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)

        if get_image:
            _, _, proj = self.image_encoder(x)
            return proj[:, 0]

        feat_pen, feat_last, feat_proj = self.image_encoder(x)
        f_pen = feat_pen[:, 0]
        f = feat_last[:, 0]
        f_proj = feat_proj[:, 0]
        feat = self.bottleneck(f)
        feat_p = self.bottleneck_proj(f_proj)

        if self.training or return_train_outputs:
            score = self.classifier(feat)
            score_p = self.classifier_proj(feat_p)
            return [score, score_p], [f_pen, f, f_proj], f_proj
        if self.neck_feat == "after":
            return torch.cat([feat, feat_p], dim=1)
        return torch.cat([f, f_proj], dim=1)


def make_model(cfg, num_class: int) -> PADModel:
    return PADModel(num_class, cfg)
