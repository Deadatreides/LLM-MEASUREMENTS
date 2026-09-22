import json, os, sys, time

sys.path.insert(0, os.path.dirname(__file__))
from env_fix import fix_cuda_dll_path

fix_cuda_dll_path()
from llama_cpp import Llama  # noqa: E402
from chat_render import render_prompt, STOP_SEQUENCES  # noqa: E402

BASE = r"<PROJECT_ROOT>\trace-probe\models"
with open(os.path.join(os.path.dirname(__file__), "models.json"), encoding="utf-8") as f:
    registry = json.load(f)

results = []
for m in registry["models"]:
    path = os.path.join(BASE, m["path"])
    mid = m["model_id"]
    rec = {"model_id": mid}
    try:
        t0 = time.time()
        llm = Llama(model_path=path, n_gpu_layers=-1, n_ctx=2048, verbose=False)
        load_s = time.time() - t0

        enable_thinking = False if m["family"] == "qwen3" else None
        msgs = [{"role": "user", "content": "What is 12 + 7? Answer with only the number."}]
        prompt = render_prompt(llm, mid, msgs, enable_thinking=enable_thinking)

        t1 = time.time()
        out = llm.create_completion(
            prompt=prompt,
            max_tokens=128,
            temperature=0.0,
            seed=42,
            stop=STOP_SEQUENCES.get(mid),
        )
        gen_s = time.time() - t1
        text = out["choices"][0]["text"]
        usage = out.get("usage", {})
        rec.update(
            status="OK",
            load_s=round(load_s, 2),
            gen_s=round(gen_s, 2),
            output=text.strip()[:300],
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
        del llm
    except Exception as e:
        rec.update(status="FAIL", error=f"{type(e).__name__}: {e}")
    results.append(rec)
    print(json.dumps(rec, ensure_ascii=False))

out_path = os.path.join(os.path.dirname(__file__), "smoke_test2_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("Saved to", out_path)
