# OmniServe：项目讲解与面试演示

本文依据仓库当前实现整理。项目定位是**本地语音推理服务与连续批处理实验**，核心贡献集中在请求状态、阶段流水线、共享 decode 和可复现评测。模型权重与底层算子由 Transformers、PyTorch 及设备后端提供。

## 90 秒项目介绍

> 我做的 OmniServe 是一个本地语音推理服务。用户可以并发上传或录制多条音频，系统先用 CPU 上的 Whisper Tiny 转写，再用 Qwen2.5-0.5B-Instruct 生成文本回答，网页展示每条请求的排队、转写、生成时间线和增量回答。
>
> 我重点实现和验证了两层调度。第一层把转写和生成拆成独立 worker，让一个请求的转写可以和另一个请求的生成重叠。第二层在 LLM worker 内维护每条请求的 token 历史和 KV Cache：新请求单独 prefill，已有请求在同一步共享一次 decode 前向；完成的请求释放自己的缓存和槽位，后续请求可以加入。
>
> 难点是不同请求的上下文长度不一样。我按层把 KV 左填充后拼成批次，配合 attention mask 和每条请求的 position IDs，计算后再去掉填充并拆回独立缓存。评测不只看网页有没有多个任务，而是记录每次 decode 的请求 ID、有效缓存长度、各阶段时间戳和输出 token 摘要，验证是否发生共享前向以及输出是否一致。
>
> 当前实现的边界也很明确：prefill 仍是完整串行执行，缓存拼装有复制开销；没有实现分页 KV、自写 CUDA kernel 或 GPU 多流并发。CUDA 的 budget 模式是单请求显存准入检查，还没有接入连续批处理的整体显存调度。

介绍完可根据提问展开具体代码，不需要在开场背诵所有文件和指标。

## 真实架构与代码入口

```text
浏览器：录音/上传、轮询任务、绘制时间线
    │ POST /api/jobs；GET /api/jobs
    ▼
ThreadingHTTPServer / VoicePipeline
    │ 音频队列
    ▼
CPU ASR worker：FFmpeg 解码 → Whisper 转写
    │ 已转写请求队列
    ▼
LLM worker
    ├─ pipeline / budget：逐请求生成
    └─ continuous：逐请求 prefill + 多请求共享 decode
    │ 更新 Job、记录首次可见文本与结束时间
    ▼
浏览器轮询增量结果；压测端收集原始记录和 decode trace
```

| 代码位置 | 应能指着代码解释的内容 |
| --- | --- |
| `voice_server.py` / `Job` | 请求状态、各阶段时间戳、输出 token 数和 token ID 摘要 |
| `voice_server.py` / `VoicePipeline` | 两级队列、任务状态锁、活动任务限制、四种执行模式 |
| `voice_server.py` / `run_continuous()` | 轮次开始时确定已有 decode 组，空位接纳新请求，新请求下一轮加入 decode |
| `voice_asr.py` / `VoiceRecognizer` | 固定 Whisper revision，CPU 推理，FFmpeg 统一解码为单声道 16 kHz |
| `voice_llm.py` / `VoiceLanguageModel` | 固定 Qwen revision、聊天模板、贪心生成、Unicode 文本发布与显存估算 |
| `workers/decode_scheduler.py` / `Scheduler` | 每请求 KV、独立 prefill、异长缓存拼装、共享 decode、完成后释放 |
| `workers/llm_worker.py` | 单请求低层 prefill/decode 基准、设备同步、峰值 allocated/reserved 指标 |
| `batching_baseline.py` | 顺序与静态批量生成对照、预热、多轮测试、token ID 对照 |
| `continuous_batching.py` | 用逻辑 tick 控制到达，验证迟到请求、异长共享 decode 和输出正确性 |
| `benchmarks/voice_benchmark.py` | 同一组音频的并发回放、到达间隔、阶段延迟、重叠时间和原始记录 |
| `web/app.js` | 录音/上传、任务轮询和服务端时间戳可视化；传输方式是轮询 |

四种模式的区别：

| 模式 | 当前行为 | 用途 |
| --- | --- | --- |
| `sequential` | 一条请求转写和生成都完成后处理下一条 | 端到端顺序基线 |
| `pipeline` | ASR 与 LLM 独立 worker；LLM 内逐请求生成 | 检验跨阶段重叠 |
| `continuous` | ASR 与 LLM 独立；LLM 内动态接纳、独立 prefill、共享 decode | 检验批内请求变化和排队影响 |
| `budget` | 流水线 + CUDA 生成前显存估算与超预算拒绝；LLM 内仍串行 | 检验单请求准入；不代表连续批整体显存管理 |

## 可直接使用的简历表述

以下三条不包含性能提升数字。使用前，应确保能够现场解释相应代码；若为已有仓库的续作，请把“实现”改成符合本人改动范围的表述。

- 构建本地并发语音推理服务，集成 Whisper Tiny 转写与 Qwen 文本生成，实现 ASR/LLM 双 worker 流水线、请求状态管理、增量回答和阶段时间线。
- 实现基于每请求 KV Cache 的连续批处理调度：独立 prefill、异长缓存左填充、共享 decode 前向及请求完成后的缓存释放，并记录批次成员与有效缓存长度以验证执行行为。
- 建立顺序、流水线和连续批处理的可复现对照工具，记录模型 revision、音频摘要、阶段延迟、吞吐、decode trace 和输出 token 摘要；支持 CUDA、MPS 与 CPU 设备路径及 CUDA 单请求显存准入检查。

需要加入性能数字时，应从本次目标设备的原始 JSON 中提取，同时写明负载、并发、精度、预热与测量轮次。README 中的 Apple M5 记录只适用于其记录的实验，不能替换为 NVIDIA 实测结果，也不能把服务排队改善表述为自写算子加速。

## 常见面试问答

### 1. Prefill 与 decode 有什么区别？

Prefill 处理整段提示词，生成提示词各位置的 KV，并根据最后一个位置的 logits 选出首个输出 token。Decode 把上一步输出的 token 作为新输入，读取历史 KV，追加一个 KV 位置，再选下一个 token。

本项目的 `Scheduler.start()` 执行整段 prefill，并把首个输出 token 放进 `state.generated`。这个 token 刚被预测出来，还没有自己的 KV。因此 decode 之前的不变量是：

```text
cache_length = prompt_length + generated_length - 1
```

`decode()` 会检查这个关系。一次 decode 前向使缓存增长一个位置，同时新选出的 token 又成为下一步的待处理输入。

### 2. KV Cache 是什么布局？为什么不同长度也能合批？

这里使用 Transformers 的 `DynamicCache` 与 `DynamicLayer`，只支持当前代码检查通过的 full-attention 层。每层有 K、V 两个张量，其形状为：

```text
[batch_size, num_key_value_heads, sequence_length, head_dim]
```

每条请求平时持有 batch 维为 1 的独立缓存。共享 decode 时取当前最长有效长度 `width`，较短请求在序列维左填充，然后沿 batch 维拼接。attention mask 遮住填充位置；`position_ids` 取各请求自己的有效缓存长度，避免把填充长度当作真实 token 位置。

模型执行后，按请求裁掉左侧填充，并用 `clone()` 保存独立的缓存张量。这样请求 A 完成后，请求 B 不会通过一片切片存储继续持有整个批次的内存。单请求路径直接复用自己的缓存，省去这一轮拼装与拆分。

### 3. 为什么要共享 decode？多个线程调用模型不行吗？

只增加 HTTP 线程可以接收多个请求，但不意味着一次 GPU 前向处理了多个序列。这里共享 decode 的具体证据是：`Scheduler.decode()` 用 batch 维为活动请求数的输入，调用一次 `self.model(...)`，同时推进这些请求。

合批有机会提高权重读取和设备执行的效率，并让后来的请求更早获得进展。但 padding、KV 拼装与拆分、更多活动缓存都会增加成本，收益要由负载和设备实测决定。当前 LLM worker 只有一个，没有实现多个 GPU stream 同时运行独立模型前向。

### 4. 它为什么叫连续批处理？与静态 batching 有什么区别？

静态 batching 通常先固定一组请求，再一起生成；本项目在生成轮次之间继续查看待处理队列，空位可以接纳新请求，已结束请求从活动集合移除。批内成员会随执行改变，例如 `A → A+B → B`。

这里的“连续”指调度过程中的动态接纳与退出。新请求单独完成 prefill，下一轮才加入已有 decode 组。它没有把 prefill 与 decode 混成一次模型前向，也没有实现 chunked prefill：长 prefill 仍可能让已有请求暂时等候。

### 5. CUDA 是异步的，时间要怎么测？

单请求低层基准在计时边界调用 `torch.cuda.synchronize()`，再读主机时钟，避免只测到异步提交开销。prefill、首 token 和整段 decode 的边界按各自代码分别同步；MPS 对应使用 `torch.mps.synchronize()`。

服务端阶段延迟则面向请求体验，包含排队、CPU 处理、设备执行和回传 token 的开销。生成过程中的 `.item()` / `.tolist()` 把 token 取回主机，会等待相应设备结果。不要为服务的每个算子随意添加全设备同步，否则可能改变待测行为。需要区分模型执行与 Python/KV 复制成本时，再另做 profiler 或设备事件分析。

### 6. 显存 allocated、reserved、free 有什么区别？

- `max_memory_allocated()`：PyTorch 管理的活跃张量内存的历史峰值。
- `max_memory_reserved()`：PyTorch 缓存分配器保留的内存的历史峰值，包含可复用空间。
- `mem_get_info()`：CUDA 运行时报告的设备空闲和总内存；空闲量还受其他进程与设备运行时影响。

`workers/llm_worker.py` 的低层测试每次计时前重置峰值，报告 allocated 与 reserved。服务 `/health` 当前只返回进程中 PyTorch 的峰值 allocated，没有按压测阶段自动重置；它可能包含模型加载、预热和之前请求的峰值。它不等于整张卡的使用量，也不等于本轮新增 KV 占用。比较时应固定实验窗口或重启服务，并写清口径。

### 7. budget 模式怎么估算？能保证不 OOM 吗？

生成前根据模型层数、KV 头数、head dimension、输入与最大输出长度、dtype 字节数估算 KV：

```text
KV bytes ≈ 2 × layers × kv_heads × head_dim
           × (input_tokens + max_new_tokens) × dtype_bytes
```

CUDA 路径在这个估算上再留出缓存增长/复制、prefill 工作空间及运行时余量，与实时空闲显存比较。它是启发式准入检查，不是分配器的精确内存承诺；执行算法、临时张量、分配器碎片及其他进程变化仍会影响实际占用。

当前 budget 使用逐请求 LLM worker，在请求生成前检查。它没有对 continuous 模式中所有活动缓存、拼装后的临时批缓存和候选请求统一记账。要扩展为连续批显存调度，需要把这些项目都纳入状态与准入策略，并在实际设备上验证边界。

### 8. 贪心生成为什么也可能出现 token 不一致？

固定权重和 greedy 只固定了“从本次 logits 取 argmax”的规则，并不保证不同 batch 形状的浮点运算逐位相同。FP16 的舍入、不同运算路径及候选 logits 很接近时的并列，都可能改变 argmax，后面的自回归序列随之分叉。

检查时先确认模型 revision、聊天模板、position IDs、attention mask、EOS 与输出上限一致，再找出首次不同的 token，检查附近候选 logits，并在更高精度路径或相同批次拓扑下复现。CPU FP32 在某个样本上通过，是该验证的证据，不能当作所有 FP16 不一致都合理的证明。

仓库的基准会做 token ID 对照；端到端任务记录 token 数和 token ID 序列摘要。摘要适合判断是否相同，定位分歧仍需要实际 token 序列或重放。当前文档中的 MPS 差异分析属于已有实验记录，需要保留其设备与样本范围。

### 9. 首 token 延迟是怎么算的？

当前端到端任务字段 `first_token_ms` 实际记录的是服务端**首次非空可见文本**发布时间。特殊 token、EOS、组成 Unicode 字符的中间字节 token，可能不会立即形成可见文字；浏览器还会增加轮询等待。

因此要把三个口径分开：低层基准的模型首 token 时间、服务端首次可见文本时间、浏览器实际显示时间。压测里的 ASR 完成至首次可见文本还包含 LLM 排队，不能直接称为纯模型 prefill 耗时。

### 10. 流水线重叠怎样证明？

从不同请求的 ASR 区间与 LLM 区间求交集，计算它们在墙钟时间上是否重叠。压测工具还把多个交集取并集，避免同一时间被重复累加。这个证据说明不同阶段重叠，不证明 GPU 多流并行。

共享 decode 则看 `/api/decode-trace`：一次成功 decode 记录中的 `job_ids` 多于一个，并且对应一次模型调用；`cache_lengths` 给出该次调用前各请求的有效缓存长度。它与网页中同时显示多个“生成中”是不同层面的证据。

### 11. 和 vLLM 的区别是什么？有没有 PagedAttention 或自己写 CUDA？

这是基于 PyTorch/Transformers 的小型推理调度实验。当前 KV 是动态连续张量，合批依靠 padding、拼接和复制；没有块表、分页 KV 管理或 PagedAttention，也没有自写 CUDA/Triton kernel。使用 CUDA 设备只代表调用已有后端执行模型。

本项目可讲清的工作是请求生命周期、跨阶段流水线、每请求缓存、共享 decode、指标与对照实验。后续优化可先 profile 缓存复制、padding 和长 prefill 的影响，再决定是否引入分块 prefill、缓存预分配或分页布局。

### 12. 并发、背压和错误怎么处理？

HTTP 层用独立处理线程接收请求，任务状态通过锁保护，ASR 与 LLM 各有任务队列。当前以活动任务数量限制形成准入边界，满额返回 HTTP 429；单个文件还有上传大小与音频时长限制。任务历史是有限的内存记录，服务重启后清空。

单请求失败会进入 error 状态。共享 decode 前向失败时，相关整个组会失败，因为模型可能已经修改过其中的缓存，不能假定继续执行仍安全。当前未实现持久化任务、客户端取消后停止推理或生产级优雅排空；不要把本地演示服务描述成完整生产平台。

## 五分钟演示脚本

演示前完成依赖安装和权重下载，准备两到四条不同长度、包含可识别语音的音频。不要把首次下载与编译算进演示；若现场没有 GPU，使用 CPU 并把请求数与输出上限降低。以下命令在仓库根目录执行。

### 0:00–0:45：确认模型、设备和模式

在终端 A 启动：

```bash
python voice_server.py --mode continuous --device cuda --max-batch-size 4 --max-new-tokens 96 --local-files-only
```

Apple Silicon 改成 `--device mps`，无 GPU 改成 `--device cpu`。只有已缓存权重时才使用 `--local-files-only`。

打开 `http://127.0.0.1:8765/health`，展示设备、dtype、模型 revision、模式与最大批大小；说明 ASR 固定在 CPU。

### 0:45–1:45：完成一条音频请求

打开首页，上传一条音频，展示转写、增量回答和各阶段时间线。指明 `queued → transcribing → waiting → generating → done` 的状态变化；解释网页通过轮询读取已生成文本。

### 1:45–3:00：展示多个请求与共享 decode

一次上传数条音频，或在长回答生成时继续上传。打开：

```text
http://127.0.0.1:8765/api/decode-trace
```

找一条有多个 `job_ids` 的记录，解释“这次模型前向推进了多个请求”，并指出每条请求有效缓存长度可以不同。若本次全部请求过短、ASR 完成太晚或生成早已结束，可能观察不到共享组；使用预演过的音频重试，或展示已保存的真实 trace，明确它是先前运行记录。

需要与音频转写无关、稳定控制迟到请求的演示时，可提前运行并保存以下低层验证输出。它检查对应负载上的迟到接纳、异长共享 decode 和独立生成对照；不应保证未经目标设备验证的 FP16 token 对照一定通过。

```bash
python continuous_batching.py --device cuda --local-files-only
```

### 3:00–4:15：展示可复现的对照方式

在终端 B 运行小规模回放，音频路径替换为演示文件：

```bash
python benchmarks/voice_benchmark.py --mode continuous --concurrency 4 --arrival-gap-ms 200 --warmup 1 --bursts 2 --output continuous-demo.json one.wav two.wav three.wav four.wav
```

展示输出中的完成请求数、端到端和 LLM 排队 p95、共享 decode 记录和音频摘要。这个短演示只验证链路；稳定性能结论需要更多测量。正式比较时将服务与压测都切换为 `pipeline`，固定设备、模型、音频、到达间隔、精度及输出上限，并分别保存原始 JSON。可在演示时展示提前跑完的两份真实结果，清楚标明环境和时间。

### 4:15–5:00：解释一段核心代码和取舍

打开 `workers/decode_scheduler.py` 的 `decode()`，依次指出：有效缓存长度检查、左填充、mask/position IDs、一次共享模型调用、裁剪与 `clone()`。最后说明下一步根据 profiling 优化 KV 复制和长 prefill，而 budget 目前仍是另一个模式中的单请求准入检查。

## 讲述前自检

- 能从一次请求追踪到 ASR 队列、LLM 队列、KV 建立、decode、EOS/长度结束和缓存释放。
- 能说明已有模型与框架提供的部分，以及自己实际新增或修改的代码。
- 不把轮询称作 SSE，不把多请求活动称作 GPU 多流并行，不把 padding 缓存称作 PagedAttention。
- 性能数字有对应原始记录；Apple MPS、NVIDIA CUDA 与 CPU 的结果分别标注。
- 能解释正确性失败时如何定位，而非直接把 token 差异全部归因于 FP16。
