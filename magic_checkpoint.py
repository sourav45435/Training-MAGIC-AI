import os
import torch
import joblib

def save_magic_checkpoint(
    out_dir,
    model,
    input_proj,
    lm_head,
    scaler,
    label_encoder,
    preprocessors,
):
    os.makedirs(out_dir, exist_ok=True)

    torch.save(model.state_dict(), f"{out_dir}/model.pt")
    torch.save(input_proj.state_dict(), f"{out_dir}/input_proj.pt")
    torch.save(lm_head.state_dict(), f"{out_dir}/lm_head.pt")

    joblib.dump(scaler, f"{out_dir}/scaler.joblib")
    joblib.dump(label_encoder, f"{out_dir}/label_encoder.joblib")
    joblib.dump(preprocessors, f"{out_dir}/preprocessors.joblib")

    print(f"✅ MAGIC checkpoint saved to {out_dir}")
