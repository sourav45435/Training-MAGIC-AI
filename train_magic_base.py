from save_magic_checkpoint import save_magic_checkpoint
# (rest of training code unchanged)

save_magic_checkpoint(
    out_dir="checkpoints/magic_base",
    model=model,
    input_proj=input_proj,
    lm_head=lm_head,
    scaler=scaler,
    label_encoder=label_encoder,
    preprocessors={"features": ["length", "uppercase_count"]},
)
