from transformers import pipeline

print("Running baseline comparisons...\n")

baselines = {
    "distilgpt2": pipeline("text-generation", model="distilgpt2"),
}

prompt = "Explain what MAGIC architecture is."

for name, pipe in baselines.items():
    out = pipe(prompt, max_length=100)[0]["generated_text"]
    print(f"\n{name.upper()} OUTPUT:\n{out}")
