# OmniServe 本次验证记录

本次在 Windows 11 / Python 3.12.14 上整理。`nvidia-smi` 识别到 NVIDIA GeForce RTX 3050 Ti Laptop GPU，4096 MiB 显存，驱动 617.42。GPU 识别与实际模型推理是两个不同的验收阶段。

## 已执行检查

| 检查 | 结果 |
| --- | --- |
| `python tools/run_tests.py` | 30 项中 25 项通过，5 项明确跳过，无失败 |
| `node --check web/app.js` | 通过 |
| 全部 Python 源码语法编译 | 通过 |
| PowerShell 安装脚本语法解析 | 通过 |
| `git diff --check` | 通过 |
| `python tools/validate_cuda.py` | 未通过：当前隔离环境缺 PyTorch/Transformers，尚未执行 GPU 运算 |

已通过测试使用真实本地 HTTP 服务及轻量替身模型，覆盖上传与查询、Host/Origin、音频大小、24 个活动任务并发准入、错误后的 worker 恢复、ASR 与 LLM 阶段重叠、历史回收、trace 去重与溢出、服务重启、CUDA 设备选择/计时 API、显存估算和 allocator 缓存复用。

5 项跳过的是 tiny Qwen FP32 的模型级调度测试：异长共享 decode 对照独立输出、KV 内存隔离、迟到请求和槽位复用、单 token 上限、prefill/decode EOS 缓存释放。它们需要 PyTorch 与 Transformers，但无需联网下载模型。

## 尚需实际执行

当前会话的网络沙箱阻止 pip 建立下载连接，因此未安装 GPU wheel 或模型权重。CUDA 数值正确性、真实 Qwen 性能、Whisper 语音端到端、本机 GPU 显存峰值与吞吐均未实测。代码适配和 mock API 测试不能替代这些结果。

在可下载依赖的本机终端，按 [CUDA 安装与验收](CUDA.md) 依次运行安装、自检、模型级测试、低层基准和同音频端到端压测。完成后以实际输出更新本记录及简历数字。`results/unit-tests.json` 保存本次计数与跳过原因，`results/cuda-validation.json` 保存本次 GPU 自检未通过的具体原因；`results/` 默认不提交 Git。

## 简历使用范围

可以描述已存在的架构与实现：两级队列、跨请求 ASR/LLM 流水线、独立 KV、异长共享 decode、CUDA 适配和评测工具。性能提升百分比、GPU 吞吐和峰值显存应等实际原始 JSON 产生后再填写。README 的 Apple M5 表格保留为历史记录，本仓库未附其原始数据，本次未复测。

GitHub Actions 工作流已添加，包含无模型测试与安装 CPU PyTorch 后的离线 tiny Qwen 测试。上述结果来自本机检查；远端 CI 的状态以仓库 Actions 页面为准。
