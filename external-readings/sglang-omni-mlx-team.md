# Building an MLX squad for SGLang-Omni

Researched 2026-09-11. Primary sources: [sglang-omni README](https://github.com/sgl-project/sglang-omni), [Roadmap #1081](https://github.com/sgl-project/sglang-omni/issues/1081), [Apple Silicon RFC #1967](https://github.com/sgl-project/sglang-omni/issues/1967), [Qwen3-ASR cookbook](https://sgl-project.github.io/sglang-omni/cookbook/qwen3_asr.html), [developer reference](https://sgl-project.github.io/sglang-omni/developer_reference/main.html), [MLX install docs](https://ml-explore.github.io/mlx/build/html/install.html), [MLX v0.32.x releases](https://github.com/ml-explore/mlx/releases), [mlx-lm v0.31.x](https://github.com/ml-explore/mlx-lm/releases), plus a small X search on Typeless (recent counts + two relevancy pages).

## What SGLang-Omni is

SGLang-Omni is **not** a dictation app. It is a multi-stage serving runtime for ASR, TTS, music, and omni models. It owns pipeline topology, stage lifecycle, inter-stage transport, and an OpenAI-compatible HTTP surface (`/v1/audio/transcriptions`, `/v1/audio/speech`, multimodal chat). Autoregressive execution is composed with [SGLang](https://github.com/sgl-project/sglang).

Apple Silicon is **experimental**. The only merged, documented MLX model path as of this note is Qwen3-ASR. A large Apple Silicon RFC ([#1967](https://github.com/sgl-project/sglang-omni/issues/1967), opened 2026-09-05) is actively claiming the rest of the catalog.

Two Apple backends are intentional, not accidental:

| Path | How | Role |
| --- | --- | --- |
| Native MLX | `SGLANG_USE_MLX=1` | Primary Metal performance path |
| Torch / MPS | default | Compatibility / correctness fallback |

The RFC is explicit: this does **not** add an `mlx-audio` runtime dependency. mlx-audio may be used as a reference implementation; runners stay inside SGLang-Omni. That is the insulation against the MLX ecosystem churn described below.

## ASR and TTS, in serving terms

**ASR (Automatic Speech Recognition)** maps audio → text. In this stack that is `/v1/audio/transcriptions`, optionally SSE (`stream=true`) or live PCM over `/v1/realtime?intent=transcription`. Quality is measured as WER/CER; serving is measured as RTF / RTFx, time-to-first-token, and memory.

**TTS (Text-to-Speech)** maps text (plus optional voice / style) → waveform. In this stack that is `/v1/audio/speech`, with batch and streaming variants. Typical pipeline: text/tokenizer → autoregressive speech tokens → vocoder/codec. Quality is UTMOS / speaker similarity; serving is time-to-first-audio, inter-chunk gap, and playback underrun.

Apple already ships Speech / dictation, and mlx-audio already runs many of the same models locally. Those are **model runners**. SGLang-Omni's job is the **serving contract**: OpenAI-compatible APIs, multi-stage graphs (thinker → talker → vocoder), request admission, KV budgets, streaming semantics, and the same code path from a MacBook to CUDA / XPU / NPU.

What actually makes ASR/TTS feel faster and easier on Mac:

1. **Native MLX, not Torch MPS.** On Fun-ASR-Nano, #1967 reports MLX mean 0.28–0.30 s vs Torch/MPS 0.54 s (~1.9×). On Qwen3-ASR-0.6B (8 GB Mac), weight load 1.28 s vs 4.51 s; warm request ~0.33–0.41 s vs 0.58–0.61 s.
2. **Quantization that actually fits unified memory.** Same Qwen3-ASR checkpoint: official weights ~1.46 GB resident / 6164 KV tokens; converted 4-bit ~0.66 GB / 7527 KV tokens. Do not assume an mlx-community artifact is required — Qwen3-ASR MLX can load the official checkpoint.
3. **Keep concurrency at 1 for latency.** MLX can batch, but `max_running_requests=1` is the documented latency profile. Raise it only for throughput.
4. **Audio decode is a real bottleneck.** Cookbook requires Homebrew `ffmpeg@7` (not FFmpeg 9) and `DYLD_LIBRARY_PATH` on the *final* `sgl-omni` process because SIP strips `DYLD_*`.
5. **Streaming and chunking.** Pseudo-streaming token deltas; live PCM + server VAD for dictation-like UX; long audio chunked at 30 s by default (Torch MPS native cap is 60 s; MLX follows the model-native 1200 s).
6. **Shared Apple ASR runners** ([#1981](https://github.com/sgl-project/sglang-omni/pull/1981)) so each new model is not a one-off Metal port.
7. **Do not wrap mlx-audio.** Pinning mlx-audio + mlx-vlm + mlx-lm together is how oMLX hit `resample_audio` import breakage. SGLang-Omni's policy avoids that class of failure.

## Why SGLang-Omni vs Typeless (and Apple Speech)

Typeless is a **consumer dictation product**: system-wide injection, LLM cleanup, filler removal, formatting, 100+ languages, Mac/Windows/iOS/Android. Independent write-ups and Typeless's own privacy language describe **cloud** transcription (AWS), not a local ASR model. "On-device history" refers to where transcripts are stored, not where audio is decoded.

Apple Speech / macOS Dictation is a **closed on-device (or Apple-cloud) recognizer** with limited model choice and no OpenAI serving API.

mlx-audio is a **library + small OpenAI-compatible server** for Apple Silicon STT/TTS. Fast to demo; not a multi-stage production runtime.

SGLang-Omni's advantages are therefore not "better popup-free dictation tomorrow." They are:

1. **Local by construction.** No high-demand queue, no word quota, no upgrade interstitial over the caret.
2. **Same API as the GPU cluster.** An app can point at `localhost:8000` on a Mac and at a CUDA SGLang-Omni deployment in the lab without rewriting clients.
3. **Model choice.** Qwen3-ASR, Fun-ASR, Whisper, MOSS-TD (diarization), Qwen3-TTS, Higgs, Fish S2-Pro, MiniMax Music 3, Qwen3-Omni — not one vendor model.
4. **Multi-stage omni, not STT-only.** Thinker/talker, vocoders, music, uploaded voices. Typeless does not do this.
5. **Serving primitives Apple Speech does not expose:** admission control, KV budgets, SSE, realtime WebSocket, replica/MPS sharing (CUDA today; the abstraction is the point).
6. **Permission surface.** A local OpenAI server does not need ScreenCaptureKit, CGEventTap, or Accessibility DOM scraping to transcribe a file. Typeless's product quality *comes from* that context; it is also the privacy complaint.
7. **Research/prod continuity.** Contributors can land a Mac path that later rides the CUDA optimizations in #1081 (streaming ASR, full duplex, PD disaggregation, step distillation).

Typeless still wins on **product layer**: global hotkey, cursor injection, LLM rewrite, personal vocabulary, zero-setup UX. If the goal is "replace Typeless for writers," SGLang-Omni is the engine; something like Superwhisper / Voibe / a Swift overlay is the shell.

## Local MacBook inference demand beyond fast / stable / correct

From the Typeless complaints, Apple Silicon RFC validation notes, and the dictation-app market:

| Need | Why it shows up on a MacBook |
| --- | --- |
| Privacy / air-gap | Cloud dictation is the #1 objection; healthcare/legal/company laptops cannot send audio or screen text to AWS. |
| Unified-memory fit | 8–16 GB machines; Qwen3-ASR 0.6B is the documented 8 GB row. 30B Omni is a different product. |
| Battery and thermals | Always-on mic + Metal will throttle; MPS caching allocator sawtooth (2–4 GB) matters for "leave it running." |
| Cold start | Fun-ASR-Nano MLX ~12 s process→healthy. Dictation apps are expected to be instant after login. |
| Background / sleep-wake | iOS Auto-Lock bug; Mac ScreenCaptureKit vs App Store. Local servers must survive lid close. |
| System-wide inject | Users compare to Typeless, not to `curl`. Accessibility + IME is a separate product. |
| Screen/app context *without* leaving the device | Typeless quality comes from focused-app + DOM + clipboard. Local equivalent is on-device RAG over Accessibility, not a cloud context channel. |
| Streaming + barge-in / full duplex | Roadmap #1909 / #2052. Voice agents need interruptibility, not file upload. |
| Personal lexicon / hotwords | Qwen3-ASR `prompt` biasing exists; Fun-ASR hotwords are being wired in mlx-audio. This is the "names and jargon" problem. |
| Concurrent pipelines | ASR cleanup LLM + TTS reply on one 16 GB die. Stage offload (#2008) is the Mac version of PD disaggregation. |
| Same OpenAI schema | Cursor, agents, and internal tools already speak `/v1/audio/*`. |
| Offline model swap | Users want Whisper today, Qwen3-ASR tomorrow, without a vendor lock-in. |
| Long-form without a 6-minute cap | Typeless session cap is a top complaint; omni chunking at 30 s / 1200 s native is the serving answer. |

## Typeless problems on X (sampled 2026-08-12 → 2026-09-10)

X recent counts for `Typeless (dictation OR transcri OR privacy OR "high demand" OR whisper OR Mac OR ASR)`: **10 posts in 7 days** — a small, noisy stream, not a firehose. Relevancy search (25 posts across two pages) mixed user complaints, competitor promo, and Grok replies. Signal that is actually about using Typeless:

| Date | Post | Complaint |
| --- | --- | --- |
| 2026-08-13 | [ddflj3310](https://x.com/ddflj3310/status/2087769023849087219) | Free tier: ad/upgrade popup on every use; considering switching. |
| 2026-08-17 | [insecurejezza](https://x.com/insecurejezza/status/2089172505491837278) | After ~250k words, transcription failures; **audio not recoverable** after a 5-minute dictation. |
| 2026-08-21 | [morphglyph](https://x.com/morphglyph/status/2090645514010370304) | Annual member: "BUG 好多". |
| 2026-08-23 | [whyeszhu](https://x.com/whyeszhu/status/2091315964944207923) (22.9k followers) | iOS update **blocks screen sleep**; still unfixed. |
| 2026-08-27 | [ox40404](https://x.com/ox40404/status/2092945468309270875) | iOS Auto-Lock disabled while Typeless is backgrounded — office/café privacy issue. |
| 2026-08-31 | [GetAskClaw](https://x.com/GetAskClaw/status/2094260702739632140) | Relentless "Upgrade to Typeless Pro / high demand" popups. |
| 2026-09-01 | [zeroowl](https://x.com/zeroowl/status/2094786433575723438) | Typeless user who wants **everything on-device** (M4 Air). |
| 2026-09-04 | [herbertyang](https://x.com/herbertyang/status/2095901755653468660) | Switched entire STT pipeline local; "no more cloud." |
| 2026-09-05 | [AIStarsGo](https://x.com/AIStarsGo/status/2096144320302551126) | ScreenCaptureKit held open → **macOS App Store hides Install** (anti-overlay fraud heuristic). |
| 2026-09-05 | [caiyue5](https://x.com/caiyue5/status/2096263055701406073) (9.9k) | Ad popup frequency up; price high for CN users. |
| 2026-09-09 | [dragon2049fly](https://x.com/dragon2049fly/status/2097581705192444106) | Upgrade / High Demand popups cover the ChatGPT composer. |

Independent of X, a November 2025 reverse-engineering thread (widely cited, e.g. [Voibe's summary](https://www.getvoibe.com/resources/typeless-privacy-issues/)) claimed 100% cloud ASR to AWS `us-east-2`, plus Accessibility/DOM/clipboard/CGEventTap collection. Treat that as an allegation unless independently re-verified; it matches Typeless's later public language that audio is processed on cloud servers.

Product-review consensus (BossAI, Voibe, buildersos — secondary): 6-minute session cap, "High demand right now, couldn't finish writing" cloud errors, noisy-room degradation, no offline mode. Accuracy and formatting are often praised.

Implication for an MLX squad: **do not compete with Typeless on popup copy.** Compete on local, recoverable audio, no session cap, no ScreenCaptureKit, OpenAI API, and model choice.

## Current roadmaps (read these before opening a PR)

### Project roadmap — [#1081](https://github.com/sgl-project/sglang-omni/issues/1081) (updated 2026-09-10)

Focus models: Qwen3-TTS 1.7B, Qwen3-ASR 1.7B, Nemotron ASR, Whisper, Qwen3-Omni, Cosmos. Hardware track now **explicitly includes Apple Silicon via #1967**. Other hardware: XPU, ROCm, Ascend NPU, MUSA, Intel CPU. Runtime themes that will eventually matter on Mac: ASR streaming (#1837), full duplex (#1909, #2052), stage offload (#1619, #2008), consumer GPU serving (#1120).

Community: Slack `#sglang-omni-dev`; WeChat via `LoveDeathAndLLM` (full name + org). Org contact: Chenyang Zhao `<zhaochenyang@lmsys.org>`.

### Apple Silicon RFC — [#1967](https://github.com/sgl-project/sglang-omni/issues/1967)

This **is** the MLX squad tracker. Status as of 2026-09-09:

| Model | Status | People |
| --- | --- | --- |
| Qwen3-ASR | Merged (#1730) | @yeahdongcn |
| Fun-CosyVoice3 | WIP #1964 | @yeahdongcn |
| Qwen3-TTS | WIP #1960 | @AkazaAkane |
| Qwen3-Omni | WIP #2006 | @adityavaid |
| Whisper ASR | WIP #1977 (merge pending) | @LijuanTang94 |
| Fun-ASR-Nano | WIP #1981→#1982→#1983 | @taylorty |
| MOSS-Transcribe-Diarize | WIP #1989 | @wirybeaver |
| MiniMax Music 3 | WIP #1990 | @wirybeaver |
| MOSS-TTS Local | WIP #1991 (hybrid MLX AR + MPS codec) | @wirybeaver |
| Fish S2-Pro | WIP #2017 | @taylorty |
| Nemotron-3.5 ASR streaming | claimed, PR TBD | @BruceLoveDecimal |
| dots.tts | claimed #2055 | @guozhihao-224 |
| Higgs Audio v3 TTS | claimed #2085 | @ErliCai |
| Shared Apple ASR runners | WIP #1981 | @taylorty |
| Support matrix docs | WIP #1992 | @Sheehan20 |
| macOS arm64 CI | WIP #1984 | @zero-piB |
| Test-collection blockers | WIP #1993 | @Sheehan20 |

Policy that a new contributor must follow:

1. **Comment on #1967 before claiming work.** Maintainers reassigned Higgs → Nemotron when someone jumped the queue.
2. **Ship both MLX and Torch/MPS** unless the RFC row says hybrid.
3. **No `mlx-audio` runtime dependency.**
4. Validate HTTP/SSE, a real corpus (Fun-ASR used 1000 utterances, WER 3.73% MLX vs 3.76% MPS), sustained RSS, and cold start. Smoke E2E is not enough for merge.
5. Keep shared semantics in common code; backend-specific execution in the runner.
6. Document `SGLANG_USE_MLX=1`, concurrency, streaming, and audio-duration limits per model.

Open contribution surfaces that are **not** already a named model claim:

- Qualification numbers (WER, TTFA, RSS, 8 GB vs 64 GB) for WIP models.
- Shared runners, capacity clamps, malformed-audio HTTP 400 — these already fixed CUDA bugs found on Mac.
- macOS CI on a Metal runner (unit CI is not HTTP/SSE).
- Stage offload on M5 (#2008).
- Streaming ASR / Nemotron if Bruce's PR stalls.
- A thin local client (hotkey → `/v1/audio/transcriptions`) is product work; it is not in the RFC, so do not surprise the issue with an Electron app.

### How to actually join

1. Clone, run `./install.sh`, serve Qwen3-ASR with `SGLANG_USE_MLX=1`, transcribe a compressed M4A (not just WAV).
2. Read `docs/design/refactor_rfc.md` and the developer reference (stage graph).
3. Join Slack `#sglang-omni-dev`.
4. Pick a **gap** on #1967 (docs, CI, qualification, unclaimed follow-up) rather than a starred model.
5. Open a stacked PR the way Fun-ASR did: shared code first, MPS second, MLX third.
6. Include hardware + mlx/sglang/omni versions + commit SHA in the PR body. The RFC asks for this.

## MLX is changing — verified

Yes. 2025–2026 is a real break in the Apple inference stack, on three layers at once.

### Core `mlx` (ml-explore/mlx)

- **Latest stable at time of writing:** [v0.32.2](https://github.com/ml-explore/mlx/releases/tag/v0.32.2) (2026-08-25). Docs still say **macOS ≥ 14.0**.
- **De facto Sonoma break:** from ~0.29.2, Metal `newResidencySetWithDescriptor` (macOS 15+) is used. oMLX issue [#125](https://github.com/jundot/omlx/issues/125) documents PyPI still shipping `macosx_14_0_arm64` wheels that crash at import on 14.x. Treat 15.0 as the real floor for current wheels.
- **M5 Neural Accelerator (NAX):** [v0.30.0](https://github.com/ml-explore/mlx/releases/tag/v0.30.0) adds NAX; later notes require `MACOSX_DEPLOYMENT_TARGET=26.2`. This splits the fleet: M1–M4 Metal vs M5 NAX.
- **MLX is no longer Apple-only.** CUDA backend (`pip install mlx[cuda12]`), Thunderbolt RDMA / JACCL ([v0.30.1](https://github.com/ml-explore/mlx/releases/tag/v0.30.1)), iOS Metal default on, Windows CUDA JIT. The library is becoming a cross-vendor array runtime with Apple as the original backend.
- **v0.31–0.32:** quantized matmul (QMM / QQMM), fused SDPA options, Metal/CUDA kernel work, compile-cache thread-safety. Serving code that assumed single-threaded generate graphs is now wrong.

### `mlx-lm`

[v0.31.0](https://github.com/ml-explore/mlx-lm/releases/tag/v0.31.0) (2026-03) through [v0.31.3](https://github.com/ml-explore/mlx-lm/releases/tag/v0.31.3) (2026-04):

- `Batch` removed from `mlx_lm.generate`; `BatchGenerator` dropped `prompt_progress_callback`. Rapid-MLX [#105](https://github.com/raullenchai/Rapid-MLX/issues/105) broke continuous batching until they adapted.
- Batch KV-cache broadcast bugs in 0.31.2 ([#1139](https://github.com/ml-explore/mlx-lm/issues/1139), fixed in 0.31.3).
- Thread-local generation streams to match MLX 0.31.2.

### `mlx-audio` (Blaizzy)

v0.4.x restructured STT/TTS into `mlx_audio.stt` / `mlx_audio.tts`, moved helpers (`resample_audio` → `stt.utils`), added Qwen3-ASR/TTS, continuous batching, Nemotron streaming, MOSS-TD. Downstream pins (oMLX [#1688](https://github.com/jundot/omlx/issues/1688)) broke audio input when mlx-vlm and mlx-audio disagreed on import paths.

### What this means for the MLX squad

SGLang-Omni's "no mlx-audio dependency + dual MLX/MPS runners" policy is the correct response to this churn. Pin **mlx** versions in the Apple installer, test on both macOS 15 (M-series Metal) and macOS 26.2+ (M5 NAX) if you claim M5 numbers, and never import `mlx_audio` from serving code. Use mlx-audio only as a reading reference for kernels and model graphs.

## Suggested first 30 days

1. Reproduce Qwen3-ASR MLX vs MPS on your machine; publish the #1992-style table (load, KV, warm latency, RSS).
2. Claim a **non-model** row on #1967 (qualification, CI Metal runner, docs) or help land a merge-pending PR (#1977 Whisper).
3. If you want product-shaped work, keep it out of the runtime: a local client that talks OpenAI audio APIs, with no ScreenCaptureKit, recoverable WAV, and no 6-minute cap — positioned against Typeless's actual X complaints.
4. Track mlx 0.32 + mlx-lm 0.31 APIs so the next SGLang bump does not silently break `SGLANG_USE_MLX=1`.
