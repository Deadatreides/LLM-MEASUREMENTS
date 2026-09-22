import json, os, sys, time

sys.path.insert(0, os.path.dirname(__file__))
from env_fix import fix_cuda_dll_path

fix_cuda_dll_path()
from llama_cpp import Llama  # noqa: E402

BASE = r"<PROJECT_ROOT>\trace-probe\models"
with open(os.path.join(os.path.dirname(__file__), "models.json"), encoding="utf-8") as f:
    registry = json.load(f)

results = []
for m in registry["models"]:
    path = os.path.join(BASE, m["path"])
    rec = {"model_id": m["model_id"]}
    try:
        t0 = time.time()
        kwargs = dict(model_path=path, n_gpu_layers=-1, n_ctx=2048, verbose=False)
        if m.get("chat_format_override"):
            kwargs["chat_format"] = m["chat_format_override"]
        llm = Llama(**kwargs)
        load_s = time.time() - t0
        chat_kwargs = {}
        if m["family"] == "qwen3":
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
        t1 = time.time()
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": "What is 12 + 7? Answer with only the number."}],
            max_tokens=64,
            temperature=0.0,
            seed=42,
            **chat_kwargs,
        )
        gen_s = time.time() - t1
        text = out["choices"][0]["message"]["content"]
        usage = out.get("usage", {})
        rec.update(
            status="OK",
            load_s=round(load_s, 2),
            gen_s=round(gen_s, 2),
            output=text.strip()[:200],
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
        del llm
    except Exception as e:
        rec.update(status="FAIL", error=f"{type(e).__name__}: {e}")
    results.append(rec)
    print(json.dumps(rec, ensure_ascii=False))

out_path = os.path.join(os.path.dirname(__file__), "smoke_test_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("Saved to", out_path)
