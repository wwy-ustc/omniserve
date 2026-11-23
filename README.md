# OmniServe

OmniServe 是一个本地并发语音推理服务。网页接收同时提交的多条录音或音频文件，显示 Whisper 转写、Qwen 流式文本回答，以及每个请求的转写、排队和生成时间线。

项目重点是推理服务调度：CPU ASR 与 GPU LLM 流水线、请求独立 KV Cache、异长上下文共享 decode，以及有原始时间戳和 token 摘要的评测。CUDA 通过 PyTorch 后端执行，当前没有自写 CUDA kernel、分页 KV Cache 或多 GPU 支持。

| 阅读入口 | 内容 |
| --- | --- |
| [CUDA 安装与验收](docs/CUDA.md) | Windows/Linux 配置、离线 GPU 自检、三种批处理对照、显存指标 |
| [项目讲解与简历](docs/INTERVIEW.md) | 90 秒介绍、代码入口、简历表述、面试问答、演示步骤 |
| [本次验证记录](docs/VALIDATION.md) | 已通过检查、跳过项和 GPU 实测状态 |

## Windows CUDA 快速启动

在项目根目录使用 Python 3.12：

```powershell
powershell -File tools/setup_windows.ps1 -Python python -Device cuda
.\.venv\Scripts\python.exe voice_server.py --mode continuous --device cuda --max-batch-size 4
```

安装脚本先从 PyTorch CUDA wheel 源安装，再装项目依赖，最后用无需下载权重的微型 Qwen 检查 GPU 和缓存调度。若 Python 没加入 PATH，用 `-Python` 传入 Python 3.12 的完整路径。更详细步骤见 [CUDA 文档](docs/CUDA.md)。

## 启动

使用 Python 3.12 和支持目标设备的 PyTorch。在 NVIDIA 机器上先使用适配驱动的 CUDA 版 PyTorch，再安装其余依赖。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-voice.txt
python voice_server.py
```

打开 http://127.0.0.1:8765。可一次选中多条音频，或在已有请求运行时继续录音。麦克风需要浏览器授权，上传支持 WAV、MP3、M4A、WebM、OGG、FLAC 和 AIFF。每条音频最多 30 秒、12 MiB。请求历史保存在服务内存中，重启会清空。

首次启动会从 Hugging Face 下载固定版本的 Whisper Tiny 与 Qwen2.5-0.5B-Instruct。下载后可加 `--local-files-only` 离线运行。音频在本机解码和推理。服务只监听回环地址。默认自动选择 CUDA、MPS 或 CPU，也可通过 `--device` 指定。

## 后端和并发

- Whisper 在 CPU 转写。Qwen 使用独立生成 Worker 并保有每条请求自己的 KV Cache。默认模式在转写 B 的同时生成 A，每次 GPU 前向属于单条请求。
- `--mode sequential` 依次完成每条请求的两个阶段，作为端到端基线。
- `--mode pipeline` 启用两个独立 Worker，跨请求重叠转写与生成。
- `--mode continuous --max-batch-size 4` 在同一个 Qwen worker 中逐轮接纳已转写请求。每条新请求单独 prefill，已就绪的请求共享 decode 前向。EOS 或达到生成上限后释放该请求的缓存和批次槽位。网页沿用同一套增量文本接口。
- `--mode budget --device cuda` 在流水线基础上增加 CUDA 显存准入检查。它根据 Prompt 长度、输出 token 上限与实时可用显存估算生成空间，超预算时拒绝该请求。当前生成 Worker 串行执行请求，显存控制按请求准入。

## 压测

对每种模式使用同一批音频、相同设备和生成参数。按模式分别启动服务，再执行压测：

```bash
python benchmarks/voice_benchmark.py --url http://127.0.0.1:8765 \
  --mode pipeline --concurrency 4 --warmup 1 --bursts 5 one.wav two.wav
```

输出包含请求吞吐量、端到端和各阶段延迟、真实跨请求重叠时长以及原始时间戳。CUDA 模式通过 `/health` 提供 PyTorch 进程的峰值已分配显存。网页时间线从服务端时间戳绘制。网页以轮询显示增量输出，因此浏览器实际观察到的首段文本会额外受轮询间隔影响。

现有的 Qwen 单请求低层基准仍可单独使用：

```bash
python workers/llm_worker.py --device mps --local-files-only
```

## 连续批处理实测

在 Apple Silicon 上启动服务。Whisper 用 CPU，Qwen 的 prefill 和每步 decode 用 Apple GPU（MPS）。NVIDIA 主机上可将 `--device mps` 换为 `--device cuda` 并使用支持 CUDA 的 PyTorch。只用 CPU 时传 `--device cpu`。

```bash
python voice_server.py --mode continuous --device mps --local-files-only
python benchmarks/voice_benchmark.py --mode continuous --concurrency 4 \
  --arrival-gap-ms 200 --warmup 1 --bursts 5 one.wav two.wav three.wav four.wav
```

将服务和压测的模式都换成 `pipeline`，保持模型、设备、输出上限、音频和提交间隔相同，即可对照。将并发数改为 1 验证单请求开销，将提交间隔设为 0 测集中提交。实际 LLM 到达时间是 ASR 完成时间。重点比较 LLM 排队、ASR 完成到首次可见文本、端到端 p95 和请求吞吐。

连续模式的 `/api/decode-trace` 保留最近 2048 次成功 decode 前向的时间、批内请求 ID 和有效 KV Cache 长度。多个请求 ID 表示一次共享模型前向。压测 JSON 收录相关记录，以及每条任务的生成 token 数和 token ID 序列摘要，可跨模式核对输出。现有 `first_token_ms` 字段记录服务端首次可见文本的时间，浏览器还受轮询间隔影响。当前批组变化和多请求 decode 会按层拼装、拆分 KV Cache，单请求 decode 直接复用自身缓存。

### Apple M5 端到端结果

以下是原仓库保留的历史结果，仓库中没有附对应原始压测 JSON。本次 Windows 迁移未复测这些数字；它们不代表 RTX 3050 Ti 的性能。新增 CUDA 实验请保存 `tools/run_cuda_benchmarks.py` 和语音压测输出，再据实际数据填写简历。

使用本机合成的四条英语语音，Whisper Tiny 在 CPU 转写，Qwen2.5-0.5B-Instruct 在 MPS 以 FP16 生成，每条最多 96 token。每个场景预热 1 轮、测量 5 轮。数值按 `pipeline → continuous` 列出。

| 提交方式 | LLM 排队 p95 | ASR 完成至首次可见文本 p95 | 端到端 p95 | 请求/秒 |
| --- | ---: | ---: | ---: | ---: |
| 单请求 | 0 → 0 ms | 33 → 31 ms | 1.37 → 1.45 s | 0.87 → 0.83 |
| 四路集中 | 1975 → 14 ms | 1996 → 45 ms | 3.98 → 2.94 s | 1.01 → 1.37 |
| 四路错峰，每次间隔 200 ms | 1941 → 16 ms | 1962 → 38 ms | 3.37 → 2.42 s | 1.01 → 1.37 |

集中提交中，示例长请求的 LLM 生成耗时中位数从 1.30 s 升到 2.11 s。整体吞吐和尾延迟改善主要来自排队时间下降。单请求场景没有共享 decode。两条长音频的专门验证中，两轮都观察到 `A → A+B → B`。

同一组请求按转写和生成 token ID 摘要对比，MPS 上 45 条中有 44 条逐 token 一致。其中一条在第 75 个 token 出现分歧。按记录的批次成员重放时，FP16 批量路径中两个候选的 logit 同为 16.890625，单条路径中分别为 16.921875 和 16.875。CPU FP32 按同一拓扑重放，这一条的 token ID 全序列一致。跨批形状的浮点结果可能产生细微差异。
