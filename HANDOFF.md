# FORGE — Proje Devir Notu (AI Handoff)

Bu dosya, FORGE projesini başka bir ortamda (başka bir Claude/ChatGPT oturumu, başka bir makine) devam ettirmek isteyen bir yapay zekaya verilmek üzere hazırlanmıştır. Projenin **nihai amacı**, şimdiye kadar alınan **tasarım kararları** ve **mevcut durum** burada özetlenmiştir.

---

## 1. Nihai Amaç

**FORGE** = **F**unction-calling **O**pen-source **R**untime for **G**rounded **E**valuation.

Akbank gibi **veri egemenliği** gereksinimi olan kurumların, açık kaynak LLM'leri kendi **on-prem HPC** ortamlarında çalıştırıp değerlendirmesini sağlayan bir framework. Veri kurum dışına çıkmaz.

Framework'ün yapacakları:
1. HPC'de (Slurm ile) bir GPU compute node üzerinde LLM inference server'ı ayağa kaldırmak.
2. Önceden indirilmiş **base** ve **quantize edilmiş** (başlangıçta AWQ) modelleri keşfetmek ve bir web arayüzünden seçilebilir kılmak.
3. Seçilen modelle **chat** yapabilmek (streaming).
4. Seçilen modelin **function-calling**, genel bilgi ve **performans (throughput/latency)** açısından benchmark'ını koşmak.
5. İleride kullanıcının kendi datasetleriyle benchmark yapmasına izin vermek (örn. çağrı-kalite değerlendirmesi).
6. İleride framework içinde model quantize etmek.

Birincil kullanım senaryosu: Akbank içi çağrı-kalite değerlendirmesi gibi hassas veri gerektiren iş yüklerinde hangi açık kaynak / quantize modelin en uygun olduğuna karar vermek.

---

## 2. Alınan Tasarım Kararları

### 2.1 Mimari
```
[Local Browser] --SSH tunnel--> [HPC login] --> [Compute Node]
                                                   ├─ FastAPI (REST + SSE)
                                                   ├─ vLLM AsyncLLMEngine (in-process)
                                                   └─ (gelecek) Benchmark job runner
[Next.js Frontend]  local makinede çalışır; API_BASE = http://localhost:<port>
```

- Backend kullanıcı tarafından **elle** `sbatch` ile compute node'da başlatılır. (İlk sürümde UI'dan slurm job submit **YOK**.)
- Frontend local makinede (kullanıcının laptopu) çalışır; backend'e **SSH port forwarding** ile erişir:
  `ssh -L <port>:<compute-node>:<port> <user>@<login-node>`
- Kimlik doğrulama YOK — güvenlik SSH tünelinden geliyor.

### 2.2 Teknoloji seçimleri
| Katman | Seçim | Neden |
|---|---|---|
| Backend dili | Python 3.11 | vLLM/transformers/lm-eval ekosistemi |
| Backend framework | FastAPI + uvicorn | async, SSE, pydantic v2 |
| Inference | **vLLM** `AsyncLLMEngine` (in-process) | production-grade, OpenAI-uyumlu API, AWQ desteği |
| GPU/sys info | `pynvml` + `psutil` | standart |
| Config | `pydantic-settings`, `FORGE_*` env vars | |
| Frontend | **Next.js 14 + TS + shadcn/ui + Tailwind** | kurumsal görünüm, TanStack Query + Recharts planlı |
| Benchmark suite'leri (planlı) | BFCL + lm-eval-harness + vLLM `benchmark_serving` | |
| Job state | In-memory dict + JSONL append (DB yok, ilk sürümde) | |

### 2.3 Kapsam daraltmaları (kullanıcı talebiyle)
- **İlk sürümde framework içinde quantize etme YOK.** Modeller cluster'da **önceden hazır** bulunur. `autoawq` bağımlılığı **kaldırıldı**. Quantize işlemi framework dışında (manuel script ile) yapılır.
- **İlk sürümde custom dataset benchmark YOK.** Sadece hazır BFCL / lm-eval / perf suite'leri planlı. Custom JSONL upload sonraki sürüme bırakıldı.
- Quantization yöntemi olarak **sadece AWQ** ile başlanıyor (GPTQ / FP8 / bitsandbytes sonra).

### 2.4 Dizin varsayımları (HPC)
- `FORGE_MODELS_DIR=/cta/users/quantization/base_models` — base modeller
- `FORGE_QUANT_DIR=/cta/users/quantization/quantized_models` — quantize edilmiş modeller
- `FORGE_DATASETS_DIR=/cta/users/quantization/datasets` — benchmark datasetleri
- `FORGE_PORT=8007` — uvicorn portu
- Her model klasörü HuggingFace formatında olmalı (`config.json` içermeli). `config.json`'daki `quantization_config.quant_method` alanından quant tipi otomatik tespit edilir.

### 2.5 Slurm kararları
- Partition/account/qos = `cuda` (kullanıcının cluster'ına özel, kullanıcı elle ayarladı).
- `--gres=gpu:1`, `--cpus-per-task=8`, `--mem=32G`, `--time=08:00:00`.
- `module load cuda/12.8` + `module load python/3.11.9-5lbnnno`.
- Venv: `~/.venv` (login node'da bir kere kurulur, shared FS üzerinden compute node erişir).
- Slurm script her çalışmada log'a `hostname` ve tunnel komutunu basar.

### 2.6 Benchmark stratejisi (planlı, henüz kod yok)
- **En basit ilk benchmark: BFCL `simple` kategorisi** (~400 örnek, function-calling, tek skor).
- **lm-eval-harness**: `mmlu`, `arc_easy`, `hellaswag` — vLLM'in OpenAI-uyumlu endpoint'ine karşı koşulur.
- **Perf**: `vllm.entrypoints.benchmark_serving` — throughput, TTFT, ITL.
- **Custom** (sonraki sürüm): JSONL formatı `{"messages":[...], "expected": "..."}` veya `{"prompt":..., "tools":[...], "expected_call":{...}}`; skorlama exact-match / schema-match / LLM-as-judge.

---

## 3. Mevcut Durum (v0.1 — çalışan minimum backend)

### 3.1 Tamamlanan
Backend iskeleti **yazıldı ve localde test edildi** (Python 3.11 venv, vLLM/NVML olmadan; `/system`, `/models`, `/` endpoint'leri 200 döndü).

**Dosya yapısı:**
```
FORGE/
├── README.md
├── HANDOFF.md            (bu dosya)
└── backend/
    ├── pyproject.toml    # fastapi, uvicorn, pydantic, pydantic-settings,
    │                     # psutil, nvidia-ml-py, httpx; [vllm], [bench] extras
    ├── slurm/
    │   └── start_backend.sbatch   # partition=cuda, gpu:1, port 8007, module load'lar
    └── app/
        ├── __init__.py
        ├── main.py                # FastAPI app, CORS, router'lar
        ├── config.py              # FORGE_* env vars (pydantic-settings)
        ├── schemas.py             # SystemInfo, GPUInfo, ModelEntry, LoadModelRequest,
        │                          # LoadedModelStatus, ChatMessage, ChatRequest
        ├── api/
        │   ├── __init__.py
        │   ├── system.py          # GET /system
        │   ├── models.py          # GET /models, GET /models/loaded,
        │   │                      # POST /models/load, POST /models/unload
        │   └── inference.py       # POST /chat (SSE stream)
        └── services/
            ├── __init__.py
            ├── gpu_info.py        # pynvml + psutil, NVML yoksa graceful degrade
            ├── model_registry.py  # MODELS_DIR + QUANT_DIR tarama, config.json parsing,
            │                      # quant_method otomatik tespit
            └── vllm_runner.py     # AsyncLLMEngine wrapper, lazy import,
                                   # tek-model lock, streaming chat, chat template otomatik
```

### 3.2 Çalışan endpoint'ler
| Method | Path | Ne yapar |
|---|---|---|
| GET | `/` | meta (versiyon, dir'ler) |
| GET | `/system` | hostname, slurm job id/node, CPU, RAM, GPU listesi (VRAM/util/temp) |
| GET | `/models` | MODELS_DIR + QUANT_DIR taraması, her model için path/size/quant |
| GET | `/models/loaded` | şu an yüklü model var mı |
| POST | `/models/load` | vLLM engine başlat (path, dtype, max_model_len, gpu_memory_utilization, quantization) |
| POST | `/models/unload` | engine kapat, VRAM boşalt |
| POST | `/chat` | SSE stream: `data: {"delta": "..."}\n\n`, sonunda `data: [DONE]` |

### 3.3 Kritik davranışlar
- **vLLM lazy import**: kurulu olmasa bile server boot olur; sadece `/models/load` 503 döner. Laptopta geliştirme mümkün.
- **Tek-model**: yeni load gelirse öncekini otomatik unload eder. Concurrent load'ı `asyncio.Lock` ile engeller.
- **Chat template**: yüklü modelin tokenizer'ını kullanarak `apply_chat_template` çağırır; template yoksa `role: content` fallback.
- **Unload**: `engine.shutdown()` + `torch.cuda.empty_cache()`.
- **CORS**: `localhost:3000` için açık (Next.js dev server).
- **Graceful GPU absence**: NVML yoksa `gpus: []` döner, crash etmez.

### 3.4 Test edildi
- Python 3.11 venv'de `pip install -e .` ✓
- `uvicorn app.main:app` boot ✓
- `GET /`, `/system`, `/models` laptopta (vLLM/NVML yok) doğru yanıt ✓
- HPC'de henüz test edilmedi (kullanıcı repo'yu GitHub'a push edip cluster'a çekecek).

### 3.5 HENÜZ YOK
- **Frontend** (Next.js projesi başlatılmadı)
- **Benchmark suite'leri** (BFCL / lm-eval / perf runner'ları — sadece tasarım var)
- **Job manager + log streaming** (`/jobs/{id}/logs` WS endpoint'i tasarımda var, kod yok)
- **Custom dataset upload / scoring**
- **Quantize endpoint'i**
- **Kalıcı state** (DB yok, restart = geçmiş kayıp)
- **Çoklu eşzamanlı model**
- **Auth** (sadece SSH tunnel güvenliği)
- **CI/testler**

---

## 4. Bir Sonraki Adımlar (öneri sırası)

1. **HPC'de v0.1'i doğrula**:
   - Repo'yu login node'a clone et, `~/.venv` kur, `pip install -e backend/` + `pip install vllm`.
   - `sbatch backend/slurm/start_backend.sbatch` → `squeue` → compute node adı → tunnel → `curl /system`, `/models`.
   - Bir AWQ modeli `/models/load` et, `/chat` ile streaming test et.
2. **Frontend iskeleti**: Next.js 14 + shadcn + Tailwind. Sayfalar: Dashboard (`/system`+loaded), Models (list+load/unload), Chat (SSE consumer).
3. **İlk benchmark**: `app/services/bench/runner.py` (job manager, in-memory + JSONL) + `bfcl.py` (simple kategorisi). `POST /benchmark`, `GET /benchmark/{id}`, WS `/jobs/{id}/logs`.
4. **lm-eval ve perf** runner'ları.
5. Sonraki sürüm: custom dataset upload + scoring, framework içi quantize, Slurm job submit UI.

---

## 5. Kullanıcı Profili & Tercihler

- Kullanıcı Türkçe konuşuyor, cevapların Türkçe olması bekleniyor.
- Akbank kullanım senaryosu: kurum içi on-prem HPC, veri dışarı çıkamaz.
- Cluster: Slurm, partition/account/qos `cuda`, Python modülü `python/3.11.9-5lbnnno`, CUDA `12.8`.
- Kullanıcı backend'i compute node'da elle `sbatch` ile başlatmayı tercih ediyor (ilk sürümde UI'dan submit istemiyor).
- Kullanıcı: "somut çalışan en küçük parçadan başla" dedi → iterative, doğrulanabilir yaklaşım tercih ediliyor.
- Kullanıcı kendi git/GitHub işlemlerini kendi yapıyor (AI'dan commit/push beklemiyor).
- Kullanıcı kapsamı daraltmayı seviyor: quantize ve custom dataset benchmark'ı bilinçli olarak sonraki sürüme attı.

---

## 6. Referans Komutlar

**Lokal geliştirme:**
```bash
cd backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
FORGE_MODELS_DIR=/tmp/x FORGE_QUANT_DIR=/tmp/y uvicorn app.main:app --reload
```

**HPC:**
```bash
# login node — bir kere
module load python/3.11.9-5lbnnno
python -m venv ~/.venv && source ~/.venv/bin/activate
pip install -e ~/FORGE/backend
pip install vllm

# her oturum
sbatch ~/FORGE/backend/slurm/start_backend.sbatch
squeue -u $USER
cat forge-backend-<jobid>.out   # node adı + tunnel komutu

# laptop
ssh -L 8007:<compute-node>:8007 <user>@<login-node>
curl http://localhost:8007/system
curl http://localhost:8007/models
```

**Model yükle + chat:**
```bash
curl -X POST http://localhost:8007/models/load \
  -H 'content-type: application/json' \
  -d '{"path":"/cta/users/quantization/quantized_models/<model-awq>","quantization":"awq"}'

curl -N -X POST http://localhost:8007/chat \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"merhaba"}],"max_tokens":128}'
```
