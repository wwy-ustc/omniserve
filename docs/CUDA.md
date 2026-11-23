# OmniServe CUDA 安装与验收

CUDA 路径覆盖 Qwen 单请求、静态批处理、连续批处理、语音服务和串行显存准入。Whisper 仍在 CPU 上执行。代码使用 PyTorch 提供的 CUDA 算子，运行 wheel 无需编译项目或自写 kernel。

## 安装

建议 Python 3.12。项目固定 `torch==2.15.0`、`transformers==5.23.0`、`imageio-ffmpeg==0.8.1`；调度器依赖当前版本的 `DynamicCache`/`DynamicLayer` 接口，不能随意降为 Transformers 4.x。

Windows，在项目根目录运行：

```powershell
powershell -File tools/setup_windows.ps1 -Python python -Device cuda
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tools/validate_cuda.py
.\.venv\Scripts\python.exe tools/run_cuda_benchmarks.py
.\.venv\Scripts\python.exe voice_server.py --mode continuous --device cuda --max-batch-size 4
```

脚本默认选择官方 `cu128` wheel 源，可用 `-CudaWheel cu132` 或 `cu134` 切换。应先根据 NVIDIA 驱动和平台核对 [PyTorch 安装选择器](https://docs.pytorch.org/get-started/)，再确定 wheel。`nvidia-smi` 显示的 CUDA 上限与 `torch.version.cuda` 的 wheel runtime 是不同信息。

Linux/WSL 的等价安装与启动：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-voice.txt
python tools/validate_cuda.py
python tools/run_cuda_benchmarks.py
python voice_server.py --mode continuous --device cuda --max-batch-size 4
```

打开 `http://127.0.0.1:8765`。首次真实推理会下载固定 revision 的 Whisper Tiny 和 Qwen2.5-0.5B-Instruct；后续给服务与基准加 `--local-files-only`。所有脚本都应使用同一个虚拟环境。

## 分层验收

| 检查 | 命令或观察 | 能证明的范围 |
| --- | --- | --- |
| 无模型逻辑测试 | `python -m unittest discover -s tests -v` | API、并发、错误恢复、trace、设备选择和显存计算；缺模型依赖时明确跳过调度器测试 |
| 微型 Qwen GPU 自检 | `python tools/validate_cuda.py` | 实际 CUDA FP16 运算、异长共享 decode、迟到请求与槽位回收、结束释放缓存；不下载权重 |
| 真实 Qwen 性能 | `python tools/run_cuda_benchmarks.py` | prefill/decode、三种批处理、到达拓扑；保存版本、GPU、时间、token 和显存 |
| 语音端到端 | 启动服务，上传或录制音频 | 音频解码→Whisper→Qwen→网页显示 |
| GPU 调度回归 | PowerShell 设 `$env:OMNISERVE_TEST_DEVICE='cuda'` 后运行测试 | 用 FP32 tiny Qwen 对照独立生成及 KV 拆装；完成后移除该环境变量 |

GPU 自检写入 `results/cuda-validation.json`，包括 `status`、GPU/版本、共享 decode 次数、token ID 和 trace。失败也会保存原因并返回非零退出码；随机小模型通过不能替代预训练模型或语音性能测试。

真实基准输出在 `results/cuda/`：`single-request.json`、`batch-comparison.json`、`arrival-topology.json`，并附每条命令的日志与 `run-summary.json`。`batch-comparison` 比较 sequential/static/continuous；各模式轮换顺序、分别预热，计时边界执行 CUDA synchronize，报告峰值 allocated/reserved。单请求脚本采用固定 decode 步数，包含 EOS 后的固定步运算，不与语音服务的提前停止吞吐直接混用。

FP16 下改变 batch 形状可能改变近似并列的 logits，从而产生 token 分歧。默认保留各模式输出及一致性字段；若要把逐 token 一致作为硬条件，给 `run_cuda_benchmarks.py` 加 `--strict-token-match`。数值分歧需要检查首次差异，并用 CPU FP32/相同到达拓扑复核；不能仅凭 FP16 差异断言缓存有错，也不能据此忽略错误。

## 端到端压测

准备同一组真实、含语音且不超过 30 秒的音频。分别用 `sequential`、`pipeline`、`continuous` 启动服务，在另一个终端运行同模式命令：

```powershell
.\.venv\Scripts\python.exe benchmarks/voice_benchmark.py --mode continuous --concurrency 4 --warmup 1 --bursts 5 --arrival-gap-ms 200 --output results/voice-continuous.json one.wav two.wav three.wav four.wav
```

将模式改为 `pipeline` 并另存输出，保持音频、输出上限、设备、提交间隔、CPU 线程数不变；再做并发 1 和间隔 0 两种对照。短音频或短答案可能没有足够重叠，先检查 `shared_decode_forwards_observed` 和实际 ASR 完成时间，再解释吞吐。

重点读取 `llm_queue_ms`、`asr_to_first_visible_ms`、`end_to_end_ms` 的 p95、请求/秒，以及原始 `jobs/decodes`。`first_token_ms` 的真实含义是服务端首次可见文本，浏览器另外有约 400 ms 轮询延迟。共享前向的直接证据是同一条 decode 记录中包含多个请求 ID。

trace 端点提供递增游标和服务实例 ID；压测在上传、等待任务完成期间持续收集。若 2048 条环形缓冲仍发生溢出或服务重启，压测返回失败，避免把不完整 trace 当成完整证据。溢出时减小 `--poll-ms` 后重测。

## 显存准入与实现边界

对 batch=1，KV Cache 字节数为 `2 × 层数 × KV头数 × head_dim × 总token数 × 每元素字节数`。Qwen2.5-0.5B 的 24 层、2 KV 头、head_dim=64 在 FP16 下每 token 约 12 KiB。注意使用 GQA 的 KV 头数，而非 query 头数。

`budget` 在每条请求生成前检查：两份 KV 的增长/复制空间、保守 prefill 工作区和至少 512 MiB 余量。可用空间包括驱动空闲显存及本进程 `reserved - allocated` 中可复用的缓存。它是启发式估算，其他进程、碎片和算子工作区仍可能导致 OOM；不会通过 `empty_cache()` 清空缓存来伪造较低显存。

```powershell
.\.venv\Scripts\python.exe voice_server.py --mode budget --device cuda
```

此模式的 LLM worker 仍串行执行，未将预算器接入连续批处理总量控制。`/health` 的峰值 allocated 是进程启动以来的累计峰值；比较服务模式时重启服务。低层基准另有每轮峰值 reserved，二者不是同一个指标。

连续批处理采用完整独立 prefill，长 prefill 会阻塞既有 decode；每轮按层拼接并拆回 KV Cache 有复制成本。当前不包含 chunked prefill、PagedAttention、CUDA Graph 或多流执行。[Transformers 缓存说明](https://huggingface.co/docs/transformers/cache_explanation)给出了 KV 的维度和 attention mask 与历史长度的对应关系。
