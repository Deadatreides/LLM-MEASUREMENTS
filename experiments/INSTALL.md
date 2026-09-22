# Установка

Три вещи, на которых установка спотыкается именно на этой машине. Всё проверено.

## 1. torch должен быть CUDA-сборкой

Сейчас в системном Python стоит `torch 2.12.1` **CPU-сборка** (`torch.cuda.is_available()`
= False). `bitsandbytes` на ней не работает, а без него нет 4-bit и int8 — то есть нет
контроля квантования (§8), обязательного раздела отчёта.

Нужна ровно та же версия, но из CUDA-индекса — тогда ничего больше в системе не поедет:

```bash
pip install torch==2.12.1+cu126 --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
```

Версию (`2.12.1`) **обязательно указывать явно**. Без пина pip уходит в перебор версий
по индексу PyTorch и висит минутами, не начиная загрузку.

CUDA 12.6 выбрана под драйвер 571.96 (CUDA 12.8) и GTX 1660 Super (sm_75) — поддерживается.

## 2. Соединение с download.pytorch.org рвётся

Колесо весит **2.6 ГБ**, а соединение здесь сбрасывается каждые 30–90 секунд. Важно:
`curl --retry` и pip перезапускают передачу **с нуля**, а не докачивают, поэтому дальше
~85 МБ они не уходят никогда.

В каталоге уже лежит частично скачанное колесо:

```
trace-probe/wheels/torch-2.12.1+cu126-cp311-cp311-win_amd64.whl   (~780 МБ из 2621 МБ)
```

Докачать его (докачка по Range-заголовку, переживает обрывы, можно запускать повторно):

```bash
python "trace-probe/fetch.py" "https://download.pytorch.org/whl/cu126/torch-2.12.1%2Bcu126-cp311-cp311-win_amd64.whl" "trace-probe/wheels/torch-2.12.1+cu126-cp311-cp311-win_amd64.whl"
```

и потом поставить локальный файл:

```bash
pip install "trace-probe/wheels/torch-2.12.1+cu126-cp311-cp311-win_amd64.whl"
```

## 3. Остальное

```bash
pip install transformers accelerate bitsandbytes safetensors huggingface_hub numpy matplotlib pyarrow
```

## 4. Кэш моделей — не на C:

На `C:` осталось ~12.8 ГБ, а моделей качать ~17 ГБ. Кэш HuggingFace надо увести на `H:`
(504 ГБ свободно). `probe.py` сам выставляет `HF_HOME` в `trace-probe/hf_cache`, но если
качать модели вручную — задать переменную окружения:

```bash
setx HF_HOME "<PROJECT_ROOT>\trace-probe\hf_cache"
```

## Проверка, что всё встало

```bash
python -c "import torch,transformers,bitsandbytes; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Ожидается `2.12.1+cu126 True NVIDIA GeForce GTX 1660 SUPER`.

## Что качается потом само

Модели тянутся при первом запуске `probe.py` (в `hf_cache`):

| тег | модель | объём |
|---|---|---|
| B / B′ | `Qwen/Qwen3-1.7B` | ~3.4 ГБ |
| A / A′ | `allenai/OLMoE-1B-7B-0924` | ~13.8 ГБ |
