import torch
import joblib
from modeling_magic import MAGIC, apply_magic_config

def load_magic_checkpoint(
    cfg,
    checkpoint_dir,
    device="cpu",
):
    model = MAGIC(
        num_encoder_layers=cfg.model.get("num_encoder_layers", 2),
        num_decoder_layers=cfg.model.get("num_decoder_layers", 2),
        num_virtual_tokens=cfg.model.get("num_virtual_tokens", 0),
        device=device,
    )

    model = apply_magic_config(cfg, model)
    model.load_state_dict(
        torch.load(f"{checkpoint_dir}/model.pt", map_location=device)
    )
    model.to(device).eval()

    input_proj = torch.nn.Linear(
        2, cfg.model.get("d_model", 256)
    ).to(device)
    input_proj.load_state_dict(
        torch.load(f"{checkpoint_dir}/input_proj.pt", map_location=device)
    )

    lm_head = torch.nn.Linear(
        cfg.model.get("d_model", 256),
        joblib.load(f"{checkpoint_dir}/label_encoder.joblib").classes_.shape[0],
    ).to(device)
    lm_head.load_state_dict(
        torch.load(f"{checkpoint_dir}/lm_head.pt", map_location=device)
    )

    scaler = joblib.load(f"{checkpoint_dir}/scaler.joblib")
    label_encoder = joblib.load(f"{checkpoint_dir}/label_encoder.joblib")
    preprocessors = joblib.load(f"{checkpoint_dir}/preprocessors.joblib")

    return model, input_proj, lm_head, scaler, label_encoder, preprocessors
