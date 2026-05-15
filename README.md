# OPD Papers · 多模态在线策略蒸馏文献集

> **On-Policy Distillation (OPD)** 是 2025 年下半年开始迅速崛起的大模型后训练新范式，由 Thinking Machines Lab 系统提出并被 Qwen3 / GLM-5 / DeepSeek-V4 / Xiaomi-MiMo 等业界一线模型采用。它把强化学习的 on-policy 采样和知识蒸馏的密集 token 监督结合起来，做到 **以 1/10 算力达到 RL 同等效果**。
>
> 本仓库聚焦 **多模态方向** 的 OPD 论文，并广泛收集了中英文社区中高质量的技术解读，已抓取的解读放在 [`references/`](references/) 下可离线阅读。

---

## 📚 论文（PDF 在 [`pdfs/`](pdfs/)）

### 1. Lightning OPD · NVIDIA

> **Lightning OPD: Efficient Post-Training for Large Reasoning Models with Offline On-Policy Distillation**
> *Yecheng Wu, Song Han, Han Cai · NVIDIA · 2026*

在 SFT 和 OPD 使用 **同一个教师 (teacher consistency)** 的前提下，把 rollout 和教师 log-prob **离线收集**，训练时所有 GPU 全部跑学生。在保持与在线 OPD 相同最优解的同时，**训练效率提升 4.0×**——从 Qwen3-8B-Base 起步，30 GPU-hour 拿到 AIME 2024 69.9%。

- arXiv：<https://arxiv.org/abs/2604.13010> · [PDF](https://arxiv.org/pdf/2604.13010)
- HuggingFace papers：<https://huggingface.co/papers/2604.13010>
- 本地 PDF：[`pdfs/01_Lightning-OPD_2604.13010.pdf`](pdfs/01_Lightning-OPD_2604.13010.pdf)

### 2. VOLD · 把 LLM 推理迁移到 VLM

> **VOLD: Reasoning Transfer from LLMs to Vision-Language Models via On-Policy Distillation**
> *Walid Bousselham, Hilde Kuehne, Cordelia Schmid · 2025–2026*

文本侧高质量推理数据丰富但视觉侧稀缺。VOLD 把 **GRPO + OPD 组合**，让纯文本 LLM 教师指导 VLM 学生的推理 trace。一个关键洞察：**冷启动对齐 (cold-start alignment) 必不可少**——师生分布差太大时 OPD 完全失效。在 MMMU-Pro / MathVision / MathVista / LogicVista 上显著超越 baseline 和 SOTA。

- arXiv：<https://arxiv.org/abs/2510.23497> · [PDF](https://arxiv.org/pdf/2510.23497)
- OpenReview：<https://openreview.net/forum?id=lkv7sOGtfk>
- 本地 PDF：[`pdfs/02_VOLD_2510.23497.pdf`](pdfs/02_VOLD_2510.23497.pdf)

### 3. Uni-OPD · LLM + MLLM 双视角统一配方

> **Uni-OPD: Unifying On-Policy Distillation with a Dual-Perspective Recipe**
> *2026*

跨 LLM/MLLM 的统一 OPD 框架，**双视角优化**：
- **学生视角**：数据平衡策略，鼓励对"高信息量学生状态"的探索
- **教师视角**：outcome-guided margin calibration，恢复 token 级聚合监督和 outcome reward 的顺序一致性

在 **5 个领域、16 个 benchmark** 上覆盖单/多教师、强到弱、跨模态蒸馏。

- arXiv：<https://arxiv.org/abs/2605.03677> · [PDF](https://arxiv.org/pdf/2605.03677) · [HTML](https://arxiv.org/html/2605.03677v1)
- 本地 PDF：[`pdfs/03_Uni-OPD_2605.03677.pdf`](pdfs/03_Uni-OPD_2605.03677.pdf)

### 4. Video-OPD · 时序视频定位 (TVG)

> **Video-OPD: Efficient Post-Training of Multimodal Large Language Models for Temporal Video Grounding via On-Policy Distillation**
> *2026*

把 OPD 落到 **TVG (Temporal Video Grounding)** 任务。32B teacher 给 8B student 的轨迹做 token 级评分，reward 综合考虑正确性与时序邻近性；teacher 只评分不生成，保证 on-policy。引入两个训练策略：
- **TRPV**：用 ground-truth IoU 过滤 teacher 不可靠的预测
- **DBTP**：优先训练 teacher-student 分歧最大的样本

平均提升 **17%**，超过 GPT-4o / GPT-5 / Gemini-2.0-Flash，逼近 Gemini-2.5-Flash。

- arXiv：<https://arxiv.org/abs/2602.02994> · [PDF](https://arxiv.org/pdf/2602.02994) · [HTML](https://arxiv.org/html/2602.02994)
- 本地 PDF：[`pdfs/04_Video-OPD_2602.02994.pdf`](pdfs/04_Video-OPD_2602.02994.pdf)

### 5. AnyFlow · NVIDIA · 任意步视频扩散

> **AnyFlow: Any-Step Video Diffusion Model with On-Policy Flow Map Distillation**
> *Yuchao Gu, Guian Fang, Yuxin Jiang, Weijia Mao, Song Han, Han Cai, Mike Zheng Shou*
> *NVIDIA · Show Lab @ NUS · MIT · 2026*

第一个 **基于 Flow Map 的任意步视频扩散蒸馏框架**，优化整条 ODE 采样轨迹而不是固定步数。核心创新 **Flow Map Backward Simulation**——把完整 Euler rollout 分解为 shortcut flow-map transitions，实现高效 on-policy 蒸馏，同时降低少步采样的离散化误差和因果生成的 exposure bias。在 1.3B → 14B 双向/因果架构上全部验证。

- arXiv：<https://arxiv.org/abs/2605.13724> · [PDF](https://arxiv.org/pdf/2605.13724)
- 项目主页：<https://nvlabs.github.io/AnyFlow/>
- 代码：<https://github.com/NVlabs/AnyFlow>
- 本地 PDF：[`pdfs/05_AnyFlow_2605.13724.pdf`](pdfs/05_AnyFlow_2605.13724.pdf)

---

## 📝 技术解读（[`references/`](references/) 下为已抓取的完整 markdown）

### A · 入门 / 概览

| 解读 | 来源 | 本地存档 |
|------|------|--------|
| **On-Policy Distillation**（官方原始博客）| [Thinking Machines Lab — Kevin Lu, 2025-10-27](https://thinkingmachines.ai/blog/on-policy-distillation/) | [`05_thinking-machines_on-policy-distillation.md`](references/05_thinking-machines_on-policy-distillation.md) |
| **On-Policy Distillation 是什么？如何做？**（最易上手）| [青稞AI](https://qingkeai.online/archives/How%20do%20On-Policy%20Distillation) · [知乎-kxzxvbk](https://zhuanlan.zhihu.com/p/2000612721868177979) | [`04_qingkeai_how-do-opd.md`](references/04_qingkeai_how-do-opd.md) |
| Aikipedia: On-Policy Distillation | [Champaign Magazine](https://champaignmagazine.com/2025/10/29/aikipedia-on-policy-distillation/) | – |
| On-Policy Distillation by Thinking Machines Lab | [Medium — ML Point](https://medium.com/@ml-point/on-policy-distillation-by-thinking-machines-lab-13028e770c4f) | – |
| Mira Murati — On-Policy Distillation: Cheap Accuracy, Real Gains | [Medium — Mahesh Lambe](https://medium.com/@maheshlambe/mira-murati-thiking-machines-on-policy-distillation-cheap-accuracy-real-gains-85ac523b20f7) | – |
| LLM Knowledge Distillation Explained for On-Device AI | [Enclave AI](https://enclaveai.app/blog/2026/03/29/llm-knowledge-distillation-on-device-explained/) | – |
| 告别昂贵的 RL？OPD：1/10 成本实现更强 Post-Training | [LittleFish'Blog](https://www.xiaoiluo.com/article/on-policy-distillation) | – |
| On-Policy Distillation 解读 | [知乎](https://zhuanlan.zhihu.com/p/1966521167360790856) · [机器学习POD 转载](https://www.mlpod.com/1217.html) | – |
| Thinking Machines Lab: On-Policy Distillation | [知乎](https://zhuanlan.zhihu.com/p/1966866454017209256) | – |
| On-Policy Distillation 核心解读 | [知乎](https://zhuanlan.zhihu.com/p/1989738134242603245) | – |

### B · 深度原理 / Self-Distillation

| 解读 | 来源 | 本地存档 |
|------|------|--------|
| **On-Policy/Self-Distillation 深度解读**（公式 + OPSD）| [青稞AI](https://qingkeai.online/archives/On-Policy-Distillation) · [知乎-一木不](https://zhuanlan.zhihu.com/p/2004306938188537902) | [`01_qingkeai_on-policy-distillation-deep-dive.md`](references/01_qingkeai_on-policy-distillation-deep-dive.md) |
| **七篇论文！深度理解 OPD 与 Self-Distillation** | [青稞AI](https://qingkeai.online/archives/20260304) · [知乎-Escapist](https://zhuanlan.zhihu.com/p/2009410859999437740) | [`06_qingkeai_seven-papers-opd-deep-dive.md`](references/06_qingkeai_seven-papers-opd-deep-dive.md) |
| **三大流派！9 篇近期工作综述** | [青稞AI](https://qingkeai.online/archives/On-Policy%20Distillation) | [`07_qingkeai_three-schools-9-papers.md`](references/07_qingkeai_three-schools-9-papers.md) |
| 大模型知识蒸馏：OPD（原理篇）| [知乎](https://zhuanlan.zhihu.com/p/2023108953026883794) | – |
| 同一个模型，两种身份：OPSD 的三种实践 | [知乎](https://zhuanlan.zhihu.com/p/2008270892493473005) | – |
| OPSD：通过自蒸馏提升 LLM 推理能力 | [知乎](https://zhuanlan.zhihu.com/p/2003773186135831026) | – |
| Self-Distilled Reasoner（作者博客）| [Siyan Zhao](https://siyan-zhao.github.io/blog/2026/opsd/) | – |
| MIT 提出 SDFT：作为逆 RL 的在线自蒸馏 | [知乎](https://zhuanlan.zhihu.com/p/2001428154242320351) | – |
| OPSD: On-Policy Self-Distillation 主题页 | [Emergent Mind](https://www.emergentmind.com/topics/on-policy-self-distillation-opsd) | – |

### C · 失败模式 / 改进 / 机制分析

| 解读 | 来源 | 本地存档 |
|------|------|--------|
| **Rethinking OPD：现象、机制与配方**（最深入的失败模式分析）| [青稞AI](https://qingkeai.online/archives/Rethinking-OPD) · [代码 thunlp/OPD](https://github.com/thunlp/OPD) | [`02_qingkeai_rethinking-opd.md`](references/02_qingkeai_rethinking-opd.md) |
| 重新思考 On-Policy 蒸馏：训练动态、内在机制及工程实践 | [知乎](https://zhuanlan.zhihu.com/p/2033318685851562798) | – |
| 用强化学习做知识蒸馏，方差太大怎么办？(KETCHUP) | [青稞AI](https://qingkeai.online/archives/KETCHUP) | – |
| 从"手推策略梯度定理"开始：基于公式推导理解 RL 的创新本质 | [青稞AI](https://qingkeai.online/archives/%E6%89%8B%E6%8E%A8%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E5%AE%9A%E7%90%86) | – |
| 大模型 RL 算法梳理：从全量词元到部分词元的路径演化 | [青稞AI](https://qingkeai.online/archives/LLM-RL) | – |

### D · 工业实践（Qwen3 / GLM-5 / DeepSeek-V4 / Xiaomi-MiMo）

| 解读 | 来源 | 本地存档 |
|------|------|--------|
| **从 DeepSeek-V4 的多专家 OPD 反观人类学习** | [青稞AI](https://qingkeai.online/archives/DeepSeek-V4-OPD) | [`03_qingkeai_deepseek-v4-opd.md`](references/03_qingkeai_deepseek-v4-opd.md) |
| 从技术报告看 OPD 的崛起：后训练新范式 | [知乎](https://zhuanlan.zhihu.com/p/2031101471563962191) | – |
| OPD 深度研究：后训练的新范式 + 四家大厂的工程博弈 | [智柴网](https://zhichai.net/topic/177619070) | – |
| Thinking Machines Lab 最新研究如何复现？OPD 让训练成本直降 10 倍 | [知乎](https://zhuanlan.zhihu.com/p/1967547671804875374) · [魔搭转载](https://modelscope.csdn.net/69042ade0e4c466a32e33126.html) | – |
| OPD：强化学习与蒸馏的完美融合，LLM 训练效率提升 30 倍 | [知乎](https://zhuanlan.zhihu.com/p/1967387150015259097) | – |
| Qwen3 技术报告：Strong-to-Weak Distillation 强到弱蒸馏 + 代码实现 | [CSDN](https://blog.csdn.net/flyfish1986/article/details/150338197) | – |
| Qwen3 技术报告解读 | [知乎](https://zhuanlan.zhihu.com/p/1906842277260821306) | – |
| OPD：让小模型在实战中成长 | [知乎](https://zhuanlan.zhihu.com/p/1974209478040757587) | – |
| 大模型的 OPD（在线蒸馏策略）| [CSDN](https://blog.csdn.net/qq_16763983/article/details/154883404) | – |
| 只要 RL 1/10 成本！翁荔的 Thinking Machines 盯上 Qwen 的黑科技 | [智源社区](https://hub.baai.ac.cn/view/49888) | – |

### E · 工具 / 代码 / 框架

| 资源 | 类型 |
|------|------|
| [Thinking Machines — Tinker Cookbook (distillation)](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation) | 官方 OPD 训练实现，~100 行 |
| [Tinker — On-Policy Context Distillation 项目](https://github.com/thinking-machines-lab/tinker-project-ideas/blob/main/on-policy-context-distillation.md) | Tinker 社区项目模板 |
| [thunlp/OPD](https://github.com/thunlp/OPD) | Rethinking OPD 论文配套代码 |
| [shawnli/on-policy-distillation-research](https://github.com/shawnli/on-policy-distillation-research) | 中文社区研究笔记 + 复现 |
| [HuggingFace TRL — General On-Policy Logit Distillation](https://huggingfaceh4-on-policy-distillation.hf.space/) | TRL 库支持任意 tokenizer 跨族蒸馏 |
| [HuggingFace Space — OPD 演示](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation) | 可视化交互演示 |
| [NVlabs/AnyFlow](https://github.com/NVlabs/AnyFlow) | 视频扩散 OPD 官方实现 |

### F · 相关 Survey / Benchmark

| 资源 | 链接 |
|------|------|
| **A Survey of On-Policy Distillation for Large Language Models** | <https://arxiv.org/abs/2604.00626> · [PDF](https://arxiv.org/pdf/2604.00626) |
| **Rethinking On-Policy Distillation of LLMs: Phenomenology, Mechanism, and Recipe** | <https://arxiv.org/abs/2604.13016> |
| **Unmasking On-Policy Distillation: Where It Helps, Where It Hurts, and Why** | <https://arxiv.org/abs/2605.10889v1> |
| **Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes** | <https://arxiv.org/abs/2603.25562> |
| **KL for a KL: On-Policy Distillation with Control Variate Baseline** | <https://arxiv.org/abs/2605.07865> |
| **Asymmetric On-Policy Distillation: Bridging Exploitation and Imitation** | <https://arxiv.org/abs/2605.06387> |
| **Black-Box On-Policy Distillation of LLMs** | <https://arxiv.org/abs/2511.10643> |
| **MiniLLM** (开山之作之一) | <https://arxiv.org/abs/2306.08543> |
| **GKD — On-Policy Distillation of Language Models** (Agarwal+, 2023) | <https://arxiv.org/abs/2306.13649> |

---

## 🧭 推荐阅读顺序

**最短路径（30 分钟入门）**：
1. [Thinking Machines 官方博客](references/05_thinking-machines_on-policy-distillation.md) — 一手定义，配代码
2. [青稞AI: OPD 是什么？如何做？](references/04_qingkeai_how-do-opd.md) — 中文最易上手
3. 看 [`pdfs/01_Lightning-OPD`](pdfs/01_Lightning-OPD_2604.13010.pdf) — 工业级最快配方

**完整路径（深度理解）**：
1. 上面三个
2. [青稞AI: OPD / Self-Distillation 深度解读](references/01_qingkeai_on-policy-distillation-deep-dive.md) — 公式 + OPSD 流派
3. [青稞AI: 七篇论文深度理解](references/06_qingkeai_seven-papers-opd-deep-dive.md) — GKD → MiniLLM → G-OPD → SDPO → TML-OPD 串讲
4. [青稞AI: 三大流派 9 篇近期工作](references/07_qingkeai_three-schools-9-papers.md) — 涵盖 Video-OPD
5. [青稞AI: Rethinking OPD](references/02_qingkeai_rethinking-opd.md) — 给社区"泼冷水"：什么时候 OPD 会失败
6. [青稞AI: DeepSeek-V4 多专家 OPD](references/03_qingkeai_deepseek-v4-opd.md) — 工业级多专家融合
7. 读完整 5 篇 PDF

---

## 🗂 仓库结构

```
opd-papers/
├── README.md
├── pdfs/                                          # 5 篇原始论文 PDF
│   ├── 01_Lightning-OPD_2604.13010.pdf
│   ├── 02_VOLD_2510.23497.pdf
│   ├── 03_Uni-OPD_2605.03677.pdf
│   ├── 04_Video-OPD_2602.02994.pdf
│   └── 05_AnyFlow_2605.13724.pdf
└── references/                                    # 7 篇高质量中英文解读（已抓取为 markdown）
    ├── 01_qingkeai_on-policy-distillation-deep-dive.md
    ├── 02_qingkeai_rethinking-opd.md
    ├── 03_qingkeai_deepseek-v4-opd.md
    ├── 04_qingkeai_how-do-opd.md
    ├── 05_thinking-machines_on-policy-distillation.md
    ├── 06_qingkeai_seven-papers-opd-deep-dive.md
    └── 07_qingkeai_three-schools-9-papers.md
```

## 📝 说明

- **PDF**：均从 [arXiv](https://arxiv.org/) 直接下载，遵循作者原始许可，本仓库仅用于个人学习再托管。引用请参考各论文 arXiv 页面。
- **解读**：`references/` 下的 markdown 都标注了原始来源链接和作者；如需正式引用请回到原始链接。知乎文章因为需要登录无法直接抓取，README 中给出链接但未做本地存档。
- **更新时间**：2026-05-15。
