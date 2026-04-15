# FORGE — Teknik Dokümantasyon

**F**unction-calling **O**pen-source **R**untime for **G**rounded **E**valuation  
**Versiyon:** 0.1  

Akbank gibi veri egemenliği gerektiren kurumların, açık kaynak LLM'leri kendi on-prem HPC ortamlarında serve edip benchmark yapmasını sağlayan bir framework. Veri kurum dışına çıkmaz.

---

## Içindekiler

1. [Mimari Genel Bakış](#1-mimari-genel-bakış)
2. [Dizin Yapısı](#2-dizin-yapısı)
3. [Backend](#3-backend)
   - [Konfigürasyon](#31-konfigürasyon)
   - [API Endpoint'leri](#32-api-endpointleri)
   - [Servisler](#33-servisler)
   - [Şemalar (Pydantic)](#34-şemalar-pydantic)
4. [Frontend](#4-frontend)
   - [Sayfa Yapısı](#41-sayfa-yapısı)
   - [API Katmanı](#42-api-katmanı)
   - [State Yönetimi](#43-state-yönetimi)
   - [Stil ve Tema](#44-stil-ve-tema)
5. [Benchmark Sistemi](#5-benchmark-sistemi)
   - [MCQ (Çoktan Seçmeli)](#51-mcq-çoktan-seçmeli)
   - [Function Calling](#52-function-calling)
   - [Veri Hazırlama](#53-veri-hazırlama)
6. [vLLM Entegrasyonu](#6-vllm-entegrasyonu)
7. [Deployment (HPC / Slurm)](#7-deployment-hpc--slurm)
8. [Veri Akışları](#8-veri-akışları)

---

## 1. Mimari Genel Bakış

```
┌──────────────────┐          SSH Tunnel            ┌─────────────────────────────┐
│  Kullanıcı       │  ◄──────────────────────────►  │  HPC Compute Node           │
│  Laptop          │   localhost:8007               │                             │
│                  │                                │  ┌───────────────────────┐  │
│  Next.js 14      │  ─── HTTP / SSE ───────────►   │  │ FastAPI (uvicorn)     │  │
│  (port 3000)     │                                │  │   ├─ /system          │  │
│                  │                                │  │   ├─ /models          │  │
│  React 18        │                                │  │   ├─ /chat (SSE)      │  │
│  TanStack Query  │                                │  │   └─ /benchmarks      │  │
│  Tailwind CSS    │                                │  ├───────────────────────┤  │
└──────────────────┘                                │  │ vLLM AsyncLLMEngine   │  │
                                                    │  │   └─ GPU (CUDA)       │  │
                                                    │  └───────────────────────┘  │
                                                    │                             │
                                                    │  /shared/models/base/       │
                                                    │  /shared/models/quant/      │
                                                    └─────────────────────────────┘
```

**Temel prensipler:**
- Backend ve frontend ayrı process olarak çalışır
- Backend HPC'de `sbatch` ile başlatılır, frontend kullanıcının laptop'unda
- Aralarındaki bağlantı SSH tunnel üzerinden
- Kimlik doğrulama yok — güvenlik SSH tunnel'dan gelir
- vLLM lazy import: GPU olmadan da backend boot olur
- Tek model aynı anda yüklü olabilir
- Benchmark çalışırken chat engellenir

---

## 2. Dizin Yapısı

```
FORGE/
├── README.md                          # Kısa proje tanıtımı (EN)
├── HANDOFF.md                         # AI devir notu (TR)
├── DOCS.md                            # Bu dosya
│
├── backend/
│   ├── pyproject.toml                 # Bağımlılıklar ve proje metadata
│   ├── slurm/
│   │   └── start_backend.sbatch       # Slurm job script
│   ├── scripts/
│   │   ├── build_arc_easy.py          # ARC-Easy dataset oluşturucu
│   │   ├── build_mmlu.py              # MMLU dataset oluşturucu
│   │   └── build_bfcl_simple.py       # BFCL dataset oluşturucu
│   ├── data/
│   │   ├── benchmarks/
│   │   │   ├── arc_easy.jsonl         # ~2.4K soru
│   │   │   ├── mmlu.jsonl             # ~14K soru, 57 konu
│   │   │   └── bfcl_simple.jsonl      # ~400 function-calling örneği
│   │   └── history/
│   │       └── benchmark_runs.jsonl   # Benchmark sonuç geçmişi
│   └── app/
│       ├── main.py                    # FastAPI app, CORS, router mount
│       ├── config.py                  # FORGE_* env vars (pydantic-settings)
│       ├── schemas.py                 # Tüm Pydantic modelleri
│       ├── api/
│       │   ├── system.py              # GET /system
│       │   ├── models.py              # Model CRUD endpoint'leri
│       │   ├── inference.py           # POST /chat (SSE streaming)
│       │   └── benchmarks.py          # Benchmark endpoint'leri
│       └── services/
│           ├── gpu_info.py            # pynvml + psutil ile sistem bilgisi
│           ├── model_registry.py      # Disk tarama, model keşfi
│           ├── vllm_runner.py         # vLLM engine wrapper (singleton)
│           └── bench/
│               ├── registry.py        # Benchmark tanım registry'si
│               ├── manager.py         # Job kuyruğu ve orkestrasyon
│               ├── history.py         # JSONL dosyasına persist
│               ├── loader.py          # JSONL dataset yükleme
│               ├── mcq.py             # MCQ skorlama motoru
│               ├── arc.py             # ARC-Easy runner
│               ├── mmlu.py            # MMLU runner
│               └── fc.py              # Function-calling runner
│
└── frontend/
    ├── package.json                   # Next.js 14, React 18, TanStack Query
    ├── .env.example                   # NEXT_PUBLIC_FORGE_API
    ├── tailwind.config.ts             # Akbank temalı dark mode renkleri
    ├── app/
    │   ├── layout.tsx                 # Root layout (Nav + Providers)
    │   ├── providers.tsx              # QueryClientProvider
    │   ├── globals.css                # Tailwind + custom component class'ları
    │   ├── page.tsx                   # Dashboard (/)
    │   ├── models/page.tsx            # Model yönetimi (/models)
    │   ├── chat/page.tsx              # Chat arayüzü (/chat)
    │   └── benchmark/page.tsx         # Benchmark runner (/benchmark)
    ├── components/
    │   └── Nav.tsx                    # Header navigation + bağlantı durumu
    └── lib/
        ├── api.ts                     # Backend API client + SSE streaming
        └── types.ts                   # TypeScript tip tanımları
```

---

## 3. Backend

**Stack:** Python 3.11 · FastAPI · uvicorn · pydantic v2 · vLLM

### 3.1 Konfigürasyon

> `backend/app/config.py`

Tüm ayarlar `FORGE_` prefix'li environment variable'lardan okunur (pydantic-settings).

| Env Var | Varsayılan | Açıklama |
|---------|-----------|----------|
| `FORGE_MODELS_DIR` | `/shared/models/base` | Base model dizini |
| `FORGE_QUANT_DIR` | `/shared/models/quant` | Quantize model dizini |
| `FORGE_DATASETS_DIR` | `/shared/datasets` | Benchmark dataset dizini |
| `FORGE_HOST` | `0.0.0.0` | Sunucu bind adresi |
| `FORGE_PORT` | `8000` | Sunucu portu |
| `FORGE_CORS_ORIGINS` | `["http://localhost:3000"]` | İzin verilen CORS origin'leri |

**Benchmark geçmişi** `backend/data/history/benchmark_runs.jsonl` dosyasına append-only olarak yazılır.

### 3.2 API Endpoint'leri

> `backend/app/api/`

#### Sistem Bilgisi

| Method | Path | Yanıt | Açıklama |
|--------|------|-------|----------|
| `GET` | `/` | JSON | Versiyon, model dizinleri |
| `GET` | `/system` | `SystemInfo` | Hostname, Slurm bilgisi, CPU, RAM, GPU listesi |

#### Model Yönetimi

| Method | Path | İstek | Yanıt | Açıklama |
|--------|------|-------|-------|----------|
| `GET` | `/models` | — | `ModelList` | Disk'teki tüm modelleri listele |
| `GET` | `/models/loaded` | — | `LoadedModelStatus` | Yüklü model durumu |
| `POST` | `/models/load` | `LoadModelRequest` | `LoadedModelStatus` | Modeli vLLM engine'e yükle |
| `POST` | `/models/unload` | — | `LoadedModelStatus` | Modeli kaldır, VRAM boşalt |

**Hata durumları:**
- `503`: vLLM yüklü değil
- `500`: Model yükleme hatası

#### Chat (Inference)

| Method | Path | İstek | Yanıt | Açıklama |
|--------|------|-------|-------|----------|
| `POST` | `/chat` | `ChatRequest` | SSE stream | Streaming token üretimi |

**SSE event formatı:**
```
data: {"delta": "Merhaba"}        ← her yeni token
data: {"delta": ", nasılsın?"}
data: {"usage": {"prompt_tokens": 12, "completion_tokens": 5}}
data: [DONE]                       ← stream sonu
```

**Önkoşullar:**
- Model yüklü olmalı (yoksa `409`)
- Benchmark çalışmıyor olmalı (yoksa `409`)

#### Benchmark

| Method | Path | İstek | Yanıt | Açıklama |
|--------|------|-------|-------|----------|
| `GET` | `/benchmarks` | — | `BenchmarkList` | Mevcut benchmark'ları listele |
| `POST` | `/benchmarks/run` | `RunBenchmarkRequest` | `BenchJobView` | Benchmark başlat |
| `GET` | `/benchmarks/jobs` | — | `list[BenchJobView]` | Tüm job'ları listele |
| `GET` | `/benchmarks/jobs/{id}` | — | `BenchJobView` | Tek job durumu |
| `POST` | `/benchmarks/jobs/{id}/cancel` | — | `BenchJobView` | Job'u iptal et |
| `GET` | `/benchmarks/history` | — | `list[BenchHistoryEntry]` | Geçmiş sonuçlar |

### 3.3 Servisler

#### GPU & Sistem Bilgisi (`services/gpu_info.py`)

`pynvml` ile her GPU için ad, VRAM (toplam/kullanılan/boş), kullanım yüzdesi ve sıcaklık bilgisi toplar. NVML yoksa boş liste döner, crash etmez.

`psutil` ile hostname, CPU sayısı, RAM bilgisi alınır. Slurm ortam değişkenleri (`SLURM_JOB_ID`, `SLURM_NODELIST`) varsa bunlar da dahil edilir.

#### Model Registry (`services/model_registry.py`)

`FORGE_MODELS_DIR` ve `FORGE_QUANT_DIR` dizinlerini tarar. Bir klasörün model olabilmesi için HuggingFace formatında `config.json` içermesi gerekir.

**Otomatik quantization tespiti:** `config.json` içindeki `quantization_config.quant_method` alanından (AWQ, GPTQ vb.) otomatik olarak tespit edilir.

Her model için `ModelEntry` döner: ad, path, tür (base/quant), boyut (bytes), quantization metodu.

#### vLLM Runner (`services/vllm_runner.py`)

Tüm projenin merkezi servisi. Singleton olarak `runner` global değişkeninde yaşar.

**Önemli özellikler:**
- **Lazy import:** `from vllm import ...` sadece `load()` çağrıldığında yapılır. GPU olmayan makinede server boot olur.
- **Tek model kilidi:** `asyncio.Lock` ile eşzamanlı load engellenir. Yeni load gelirse önceki otomatik unload olur.
- **Chat template:** Modelin tokenizer'ındaki `chat_template` ile prompt formatlanır. Template yoksa `role: content` fallback'i.
- **Benchmark kilidi:** `bench_running` flag'i ile benchmark sırasında chat engellenir. Lock değil bool — chat hızlıca 409 döner.

**Unload süreci (5 adım):**
1. `engine.shutdown()` — vLLM arka plan worker'larını durdur
2. Tüm referansları `None` yap — GC toparlasın
3. `torch.distributed.destroy_process_group()` — NCCL kaynak sızıntısı önle
4. 3 tur `gc.collect()` — döngüsel referansları kır
5. `torch.cuda.empty_cache()` + `reset_peak_memory_stats()` — GPU bellek boşalt

### 3.4 Şemalar (Pydantic)

> `backend/app/schemas.py`

```
SystemInfo
├── hostname: str
├── slurm_job_id: str | None
├── slurm_node: str | None
├── cpu_count: int
├── ram_total_bytes: int
├── ram_available_bytes: int
└── gpus: list[GPUInfo]
         ├── index: int
         ├── name: str
         ├── memory_total / used / free: int
         ├── utilization_pct: float
         └── temperature_c: float | None

ModelEntry
├── name: str
├── path: str
├── kind: "base" | "quant"
├── size_bytes: int
└── quantization: str | None        # "awq", "gptq", vb.

LoadModelRequest
├── path: str
├── dtype: str = "auto"
├── max_model_len: int | None       # None = modelin kendi max'i
├── gpu_memory_utilization: float = 0.90
└── quantization: str | None

ChatRequest
├── messages: list[ChatMessage]      # role + content
├── max_tokens: int = 512
├── temperature: float = 0.7
├── top_p: float = 0.95
└── stream: bool = true

RunBenchmarkRequest
├── benchmark: "arc_easy" | "mmlu" | "bfcl_simple"
├── limit: int | None                # None = tüm dataset
├── temperature: float = 0.0
├── max_tokens: int = 4              # MCQ için 4, FC için 256
└── seed: int | None

BenchJobView
├── id: str (UUID)
├── benchmark: str
├── status: queued | running | completed | failed | cancelled
├── model_path / model_quant: str | None
├── started_at / finished_at: float | None
├── examples_done / examples_total: int
├── correct: int
├── score: float | None              # correct / examples_done
├── error: str | None
├── limit / temperature / max_tokens / seed: ...
```

---

## 4. Frontend

**Stack:** Next.js 14 (App Router) · React 18 · TypeScript · TailwindCSS · TanStack Query

### 4.1 Sayfa Yapısı

#### Dashboard (`/`) — `app/page.tsx`

Sistem durumu izleme sayfası. 3 saniyede bir otomatik yenilenir.

**Gösterilen bilgiler:**
- Hostname, Slurm Job ID, node adı
- CPU çekirdek sayısı, RAM kullanımı
- GPU kartları: her biri için ad, sıcaklık, VRAM ve kullanım progress bar'ları
- Yüklü model bilgisi (varsa)

#### Models (`/models`) — `app/models/page.tsx`

Model listeleme, yükleme ve kaldırma sayfası.

**Özellikler:**
- Tüm base ve quant modeller kart olarak listelenir
- Her kartta: ad, tür badge'i, quantization badge'i, boyut (GB), Load butonu
- Yüklü model yeşil kenarlıkla vurgulanır + "LOADED" badge
- Model yüklemeden önce ayarlanabilir parametreler:
  - **Max Context Length** (varsayılan: 32768)
  - **GPU Memory %** (varsayılan: 0.90)
- Unload butonu (model yüklüyken görünür)

#### Chat (`/chat`) — `app/chat/page.tsx`

Gerçek zamanlı SSE streaming chat arayüzü.

**Sol panel (chat alanı):**
- Mesaj geçmişi (localStorage'da saklanır)
- Streaming sırasında yanıp sönen cursor animasyonu
- Enter ile gönder, Shift+Enter ile yeni satır
- Temizle ve Durdur butonları

**Sağ panel (ayarlar):**
- System prompt textarea
- Temperature slider (0–2)
- Max tokens slider (32–4096)
- Performans metrikleri: throughput (token/s), toplam süre, token sayıları

#### Benchmark (`/benchmark`) — `app/benchmark/page.tsx`

Benchmark çalıştırma ve sonuç takip sayfası.

**Sol panel:**
- Benchmark seçimi (kartlar halinde): ARC-Easy, MMLU, BFCL Simple
- Aktif job gösterimi: progress bar, doğru sayısı, skor, süre, parametreler
- İptal butonu (çalışan job için)
- Geçmiş sonuçlar listesi: her biri için model, skor, tarih, parametreler

**Sağ panel (kontroller):**
- Örnek sayısı slider (10 – max)
- "Tüm dataset" checkbox'ı
- Run butonu

### 4.2 API Katmanı

> `frontend/lib/api.ts`

Backend ile iletişim tek bir `api` objesi üzerinden yapılır. `NEXT_PUBLIC_FORGE_API` env var'ından base URL alınır (varsayılan: `http://localhost:8000`).

```typescript
api.system()              // GET /system
api.models()              // GET /models
api.loaded()              // GET /models/loaded
api.load(req)             // POST /models/load
api.unload()              // POST /models/unload
api.benchmarks()          // GET /benchmarks
api.runBenchmark(req)     // POST /benchmarks/run
api.benchJob(id)          // GET /benchmarks/jobs/{id}
api.benchJobs()           // GET /benchmarks/jobs
api.cancelBenchJob(id)    // POST /benchmarks/jobs/{id}/cancel
api.benchHistory()        // GET /benchmarks/history
```

**Streaming chat** ayrı bir `streamChat()` fonksiyonu ile yapılır:
- SSE formatını parse eder (`data: {...}\n\n`)
- `onDelta` callback ile her token anında UI'a iletilir
- `onUsage` callback ile son metrikleri iletir
- Client tarafında TTFT ve throughput hesaplanır
- `AbortController` ile iptal desteği

### 4.3 State Yönetimi

**TanStack Query** ile server state yönetimi:

| Query Key | Endpoint | Yenileme Aralığı | Kullanıldığı Sayfa |
|-----------|----------|-------------------|---------------------|
| `["system"]` | `/system` | 3s | Dashboard |
| `["models"]` | `/models` | Manuel | Models |
| `["loaded"]` | `/models/loaded` | 3–5s | Dashboard, Models, Chat, Nav |
| `["benchmarks"]` | `/benchmarks` | Manuel | Benchmark |
| `["benchHistory"]` | `/benchmarks/history` | 5s | Benchmark |
| `["benchJob", id]` | `/benchmarks/jobs/{id}` | 1s (aktifken) | Benchmark |

**Local state:**
- Chat mesajları ve system prompt → `localStorage`
- Form input'ları (temperature, max_tokens vb.) → React `useState`

### 4.4 Stil ve Tema

> `frontend/tailwind.config.ts` + `frontend/app/globals.css`

Akbank kurumsal kimliğine uygun **dark tema**:

| Token | Renk | Kullanım |
|-------|------|----------|
| `forge-bg` | `#0b0d10` | Ana arka plan |
| `forge-panel` | `#12151a` | Kart / panel arka planı |
| `forge-border` | `#1f242c` | Kenarlıklar |
| `forge-text` | `#e6e8eb` | Ana metin |
| `forge-muted` | `#8a8f98` | İkincil metin |
| `forge-accent` | `#e30613` | Akbank kırmızısı — butonlar, aktif durumlar |

**Custom CSS class'ları:** `.card`, `.btn`, `.btn-accent`, `.input`, `.label`, `.badge`

---

## 5. Benchmark Sistemi

> `backend/app/services/bench/`

### Genel Akış

```
POST /benchmarks/run
        │
        ▼
   BenchManager.submit()
        │
        ├─ Validasyon: model yüklü mü? Başka benchmark çalışıyor mu?
        ├─ BenchJob oluştur (status: queued)
        ├─ runner.acquire_bench()  ← chat'i engelle
        └─ asyncio.Task başlat → _run(job)
                │
                ▼
          status: running
                │
        ┌───────┴───────┐
        │               │
    MCQ runner      FC runner
   (arc/mmlu)     (bfcl_simple)
        │               │
        │  Her örnek:    │  Her örnek:
        │  1. Prompt     │  1. Prompt + tools
        │     formatla   │     (chat template)
        │  2. generate_  │  2. generate_one()
        │     one()      │  3. Function call
        │  3. Harf       │     parse et
        │     parse et   │  4. Skorla
        │  4. Karşılaştır│     (isim + argüman
        │                │      eşleşmesi)
        └───────┬───────┘
                │
                ▼
        status: completed
        score hesapla, history'ye yaz
        runner.release_bench()  ← chat'i aç
```

**Frontend polling:** Aktif job varken her 1 saniyede `GET /benchmarks/jobs/{id}` ile ilerleme takip edilir.

### 5.1 MCQ (Çoktan Seçmeli)

> `bench/mcq.py` — ARC-Easy ve MMLU tarafından kullanılır

**Prompt formatı:**
```
Answer the multiple-choice question with a single letter only.

Question: Hangi gezegen Güneş'e en yakındır?
A) Venüs
B) Merkür
C) Mars
D) Jüpiter
Answer:
```

**Skorlama:** Model çıktısından ilk geçerli harf (A/B/C/D) alınır, gold cevapla karşılaştırılır. `temperature=0` ile deterministik üretim.

**Mevcut MCQ benchmark'ları:**

| ID | Adı | Dataset | Örnek Sayısı | Özellik |
|----|-----|---------|-------------|---------|
| `arc_easy` | ARC-Easy | `arc_easy.jsonl` | ~2.4K | İlkokul seviye bilim soruları |
| `mmlu` | MMLU | `mmlu.jsonl` | ~14K | 57 farklı konu (matematik, tarih, hukuk vb.) |

### 5.2 Function Calling

> `bench/fc.py` — BFCL Simple benchmark'ı

Model, verilen tool tanımlarına göre doğru fonksiyon çağrısı üretmeli.

**Örnek girdi:**
```json
{
  "messages": [{"role": "user", "content": "Üçgenin alanını hesapla, taban 10, yükseklik 5"}],
  "tools": [{"type": "function", "function": {"name": "calculate_triangle_area", ...}}],
  "expected_call": {"name": "calculate_triangle_area", "arguments": {"base": [10], "height": [5]}}
}
```

**Prompt oluşturma:** Modelin tokenizer'ındaki `chat_template`'e `tools` parametresi ile uygulanır. Bu, modelin kendi tool-calling formatını kullanmasını sağlar.

**Çıktı parse stratejileri** (sırayla denenir):
1. Ham JSON: `{"name": "func", "arguments": {...}}`
2. XML tag: `<tool_call>{...}</tool_call>`
3. Python stili: `func_name(arg1=val1, arg2=val2)`
4. OpenAI stili: `{"function": {"name": ..., "arguments": ...}}`

**Skorlama:**
- Fonksiyon adı tam eşleşmeli
- Tüm beklenen argümanlar mevcut olmalı
- Her argüman değeri kabul edilen alternatiflerden biriyle eşleşmeli
- Sayısal karşılaştırma: `"10"` == `10` (tip dönüşümü yapılır)

### 5.3 Veri Hazırlama

> `backend/scripts/`

Benchmark datasetleri orijinal kaynaklardan indirilerek JSONL formatına dönüştürülür:

| Script | Kaynak | Çıktı |
|--------|--------|-------|
| `build_arc_easy.py` | HuggingFace (parquet) | `data/benchmarks/arc_easy.jsonl` |
| `build_mmlu.py` | Berkeley tarball (CSV) | `data/benchmarks/mmlu.jsonl` |
| `build_bfcl_simple.py` | GitHub gorilla repo (JSON) | `data/benchmarks/bfcl_simple.jsonl` |

Çalıştırma:
```bash
cd backend
python scripts/build_arc_easy.py
python scripts/build_mmlu.py
python scripts/build_bfcl_simple.py
```

---

## 6. vLLM Entegrasyonu

> `backend/app/services/vllm_runner.py`

`VLLMRunner` sınıfı projenin çekirdeğidir. Modül sonunda `runner = VLLMRunner()` ile singleton oluşturulur.

### Model Yükleme

```python
AsyncEngineArgs(
    model=path,                     # HuggingFace model dizini
    dtype="auto",                   # Otomatik tip seçimi
    max_model_len=32768,            # Context window sınırı
    gpu_memory_utilization=0.90,    # GPU belleğinin %90'ını kullan
    quantization="awq",             # Quantization metodu (opsiyonel)
)
```

`max_model_len` belirtilmezse model kendi max değerini kullanır. GPU belleği yetersizse hata verir — bu yüzden frontend'den ayarlanabilir.

### Streaming Chat

```
format_prompt()  →  SamplingParams  →  engine.generate()  →  yield delta'lar
```

1. Tokenizer'ın `chat_template`'i ile mesajlar prompt'a dönüştürülür
2. `SamplingParams` ile temperature, top_p, max_tokens ayarlanır
3. `engine.generate()` async generator olarak token token üretir
4. Her yeni token `{"type": "delta", "text": "..."}` olarak yield edilir
5. Sonunda `{"type": "usage", ...}` ile token sayıları bildirilir

### Non-Streaming (Benchmark)

`generate_one()` metodu benchmark runner'ları tarafından kullanılır. Tüm üretim bitene kadar bekler, tek string döner. `stop` parametresi ile erken durma desteği vardır.

---

## 7. Deployment (HPC / Slurm)

> `backend/slurm/start_backend.sbatch`

### Slurm Job Parametreleri

```
Partition/Account/QoS : cuda
GPU                   : 1 (--gres=gpu:1)
CPU                   : 8 çekirdek
Bellek                : 32 GB
Süre                  : 8 saat
Modüller              : cuda/12.8, python/3.11.9
```

### İlk Kurulum (Login Node — Bir Kere)

```bash
module load python/3.11.9-5lbnnno
python -m venv ~/.venv
source ~/.venv/bin/activate
pip install -e ~/FORGE/backend
pip install vllm
```

### Her Oturum

```bash
# 1. Backend'i başlat
sbatch ~/FORGE/backend/slurm/start_backend.sbatch
squeue -u $USER                          # compute node adını bul
cat forge-backend-<jobid>.out            # tunnel komutunu oku

# 2. SSH tunnel (laptop'tan)
ssh -L 8007:<compute-node>:8007 <user>@<login-node>

# 3. Frontend'i başlat (laptop'ta)
cd frontend
NEXT_PUBLIC_FORGE_API=http://localhost:8007 npm run dev

# 4. Tarayıcıda aç
open http://localhost:3000
```

### Lokal Geliştirme (GPU Olmadan)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e .
FORGE_MODELS_DIR=/tmp/models FORGE_QUANT_DIR=/tmp/quant uvicorn app.main:app --reload

# Ayrı terminalde:
cd frontend
npm install && npm run dev
```

vLLM olmadan `/system`, `/models` çalışır; `/models/load` → `503` döner.

---

## 8. Veri Akışları

### Model Yükleme

```
Frontend                          Backend
   │                                 │
   │  POST /models/load              │
   │  {path, max_model_len,          │
   │   gpu_memory_utilization,       │
   │   quantization}                 │
   │ ──────────────────────────────► │
   │                                 │  VLLMRunner.load()
   │                                 │  ├─ Önceki model varsa unload
   │                                 │  ├─ vLLM import et
   │                                 │  ├─ AsyncEngineArgs oluştur
   │                                 │  └─ AsyncLLMEngine başlat
   │  {loaded: true, path: "..."}    │
   │ ◄────────────────────────────── │
```

### Chat Streaming

```
Frontend                          Backend
   │                                 │
   │  POST /chat                     │
   │  {messages, max_tokens,         │
   │   temperature, stream: true}    │
   │ ──────────────────────────────► │
   │                                 │  format_prompt()
   │                                 │  engine.generate() başlat
   │                                 │
   │  data: {"delta": "Merhaba"}     │  ← her token
   │ ◄────────────────────────────── │
   │  data: {"delta": " dünya"}      │
   │ ◄────────────────────────────── │
   │  data: {"usage": {              │  ← son
   │    "prompt_tokens": 12,         │
   │    "completion_tokens": 5}}     │
   │ ◄────────────────────────────── │
   │  data: [DONE]                   │
   │ ◄────────────────────────────── │
```

### Benchmark Çalıştırma

```
Frontend                          Backend
   │                                 │
   │  POST /benchmarks/run           │
   │  {benchmark: "arc_easy",        │
   │   limit: 100}                   │
   │ ──────────────────────────────► │
   │                                 │  BenchManager.submit()
   │  {id: "abc", status: "queued"}  │  ├─ Job oluştur
   │ ◄────────────────────────────── │  └─ Background task başlat
   │                                 │
   │  GET /benchmarks/jobs/abc       │     (arka planda çalışır)
   │ ──────────────────────────────► │     örnek 1... 2... 3...
   │  {status: "running",            │
   │   examples_done: 42,            │
   │   correct: 38,                  │
   │   score: 0.905}                 │
   │ ◄────────────────────────────── │
   │                                 │
   │  (1 saniye sonra tekrar)        │
   │  GET /benchmarks/jobs/abc       │
   │ ──────────────────────────────► │
   │  {status: "completed",          │
   │   examples_done: 100,           │
   │   correct: 87,                  │
   │   score: 0.870}                 │
   │ ◄────────────────────────────── │
   │                                 │  → history JSONL'e yaz
   │                                 │  → bench kilidini aç
```

---

## Bağımlılıklar

### Backend (`pyproject.toml`)

| Paket | Versiyon | Amaç |
|-------|---------|------|
| `fastapi` | >=0.115 | Web framework |
| `uvicorn[standard]` | >=0.30 | ASGI server |
| `pydantic` | >=2.7 | Veri validasyonu |
| `pydantic-settings` | >=2.4 | Env var config |
| `psutil` | >=5.9 | CPU/RAM bilgisi |
| `nvidia-ml-py` | >=12.535 | GPU bilgisi (pynvml) |
| `httpx` | >=0.27 | HTTP client |
| `vllm` | >=0.6.0 | LLM inference *(opsiyonel extra)* |
| `lm-eval` | >=0.4.4 | Benchmark *(opsiyonel extra)* |
| `pyarrow` | >=17 | Parquet okuma *(opsiyonel extra)* |

### Frontend (`package.json`)

| Paket | Versiyon | Amaç |
|-------|---------|------|
| `next` | 14.2.15 | React framework |
| `react` | 18.3.1 | UI library |
| `@tanstack/react-query` | ^5.59.0 | Server state yönetimi |
| `tailwindcss` | 3.4.13 | Utility-first CSS |
| `typescript` | ^5.6.2 | Tip güvenliği |
