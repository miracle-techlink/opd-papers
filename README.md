# OPD Papers · 多模态在线策略蒸馏文献集

> **On-Policy Distillation (OPD)** 是 2025 年下半年开始迅速崛起的大模型后训练新范式，由 Thinking Machines Lab 系统提出并被 Qwen3 / GLM-5 / DeepSeek-V4 / Xiaomi-MiMo 等业界一线模型采用。它把强化学习的 on-policy 采样和知识蒸馏的密集 token 监督结合起来，做到 **以 1/10 算力达到 RL 同等效果**。
>
> 本仓库聚焦 **多模态方向**（MLLM / VLM / VLA / 扩散与流匹配等非自回归生成模型）的 OPD 论文，并广泛收集了中英文社区中高质量的技术解读，已抓取的解读放在 [`references/`](references/) 下可离线阅读。
>
> 📊 **目前收录 17 篇 PDF + 7 篇深度解读**，按主题分类。

---

## 📚 论文（[`pdfs/`](pdfs/) 按主题分子目录）

### 🧠 01 · LLM 推理（OPD 的源头）

#### Lightning OPD · NVIDIA · `2604.13010`

> **Lightning OPD: Efficient Post-Training for Large Reasoning Models with Offline On-Policy Distillation**
> *Yecheng Wu, Song Han, Han Cai · NVIDIA · 2026*

在 SFT 和 OPD 使用 **同一个教师 (teacher consistency)** 的前提下，把 rollout 和教师 log-prob **离线收集**，训练时所有 GPU 全部跑学生。在保持与在线 OPD 相同最优解的同时，**训练效率提升 4.0×**——从 Qwen3-8B-Base 起步，30 GPU-hour 拿到 AIME 2024 69.9%。

- 📄 [arXiv](https://arxiv.org/abs/2604.13010) · [PDF](https://arxiv.org/pdf/2604.13010) · [HuggingFace](https://huggingface.co/papers/2604.13010)
- 💾 [`pdfs/01-llm-reasoning/Lightning-OPD_2604.13010.pdf`](pdfs/01-llm-reasoning/Lightning-OPD_2604.13010.pdf)

---

### 👁️ 02 · VLM / MLLM 推理与跨模态

#### VOLD · LLM→VLM 推理迁移 · `2510.23497`

> **VOLD: Reasoning Transfer from LLMs to Vision-Language Models via On-Policy Distillation**
> *Walid Bousselham, Hilde Kuehne, Cordelia Schmid · 2025–2026*

文本侧高质量推理数据丰富但视觉侧稀缺。VOLD 把 **GRPO + OPD 组合**，让纯文本 LLM 教师指导 VLM 学生的推理 trace。关键洞察：**冷启动对齐 (cold-start alignment) 必不可少**——师生分布差太大时 OPD 完全失效。MMMU-Pro / MathVision / MathVista / LogicVista 全面领先 SOTA。

- 📄 [arXiv](https://arxiv.org/abs/2510.23497) · [PDF](https://arxiv.org/pdf/2510.23497) · [OpenReview](https://openreview.net/forum?id=lkv7sOGtfk)
- 💾 [`pdfs/02-vlm-mllm/VOLD_2510.23497.pdf`](pdfs/02-vlm-mllm/VOLD_2510.23497.pdf)

#### Uni-OPD · LLM + MLLM 双视角统一配方 · `2605.03677`

> **Uni-OPD: Unifying On-Policy Distillation with a Dual-Perspective Recipe** · *2026*

跨 LLM/MLLM 的统一 OPD 框架。**学生视角**做数据平衡探索 informative 状态；**教师视角**用 outcome-guided margin calibration 恢复 token-level 监督与 outcome reward 的顺序一致性。**5 个领域、16 个 benchmark** 上覆盖单/多教师、强到弱、跨模态蒸馏。

- 📄 [arXiv](https://arxiv.org/abs/2605.03677) · [PDF](https://arxiv.org/pdf/2605.03677) · [HTML](https://arxiv.org/html/2605.03677v1)
- 💾 [`pdfs/02-vlm-mllm/Uni-OPD_2605.03677.pdf`](pdfs/02-vlm-mllm/Uni-OPD_2605.03677.pdf)

#### Video-OPD · 时序视频定位 (TVG) · `2602.02994`

> **Video-OPD: Efficient Post-Training of MLLMs for Temporal Video Grounding via OPD** · *2026*

把 OPD 落到 TVG 任务。32B teacher 给 8B student 的轨迹做 token 级评分，reward 综合考虑正确性与时序邻近性。两个训练策略：**TRPV**（用 GT IoU 过滤教师不可靠预测）+ **DBTP**（优先训练分歧最大样本）。平均提升 **17%**，超过 GPT-4o / GPT-5 / Gemini-2.0-Flash。

- 📄 [arXiv](https://arxiv.org/abs/2602.02994) · [PDF](https://arxiv.org/pdf/2602.02994)
- 💾 [`pdfs/02-vlm-mllm/Video-OPD_2602.02994.pdf`](pdfs/02-vlm-mllm/Video-OPD_2602.02994.pdf)

#### RAL · On-Policy Attention Distillation · `2602.04884`

> **Reinforced Attention Learning** · *2026*

跳出"对齐输出 token"的传统思路：**直接对内部注意力分布做 policy gradient 优化**。引入 *On-Policy Attention Distillation*——蒸馏的不是输出，是隐藏的注意力行为。证明在跨模态对齐上比标准 KD 更强。

- 📄 [arXiv](https://arxiv.org/abs/2602.04884) · [PDF](https://arxiv.org/pdf/2602.04884) · [HTML](https://arxiv.org/html/2602.04884)
- 💾 [`pdfs/02-vlm-mllm/RAL_2602.04884.pdf`](pdfs/02-vlm-mllm/RAL_2602.04884.pdf)

---

### 🤖 03 · VLA / Robotics 视觉-语言-动作

#### VLA-OPD · Bridging SFT 与 Online RL · `2603.26666`

> **VLA-OPD: Bridging Offline SFT and Online RL for VLA Models via On-Policy Distillation** · *2026*

在 VLA 模型上落地 OPD：用专家 teacher 在 student 自生成轨迹上提供 **dense token-level 监督**，避开稀疏环境 reward。框架基于 **Reverse-KL**，避免 Forward-KL 的 mode-covering 熵爆炸和 Hard-CE 的过早熵坍塌。**LIBERO / RoboTwin2.0** 上同时提升 RL 的样本效率和 SFT 的鲁棒性，缓解灾难性遗忘。

- 📄 [arXiv](https://arxiv.org/abs/2603.26666) · [PDF](https://arxiv.org/pdf/2603.26666)
- 💾 [`pdfs/03-vla-robotics/VLA-OPD_2603.26666.pdf`](pdfs/03-vla-robotics/VLA-OPD_2603.26666.pdf)

#### Refined Policy Distillation · VLA Generalists → RL Experts · `2503.05833`

> **Refined Policy Distillation: From VLA Generalists to RL Experts** · *2025*

VLA 通用模型在具体任务上往往落后 task-specific 模型。RPD 用 RL 训出的 expert 反向蒸馏回 VLA generalist，做到"通用基座 + 专家精修"的组合。早期 VLA + 策略蒸馏代表作。

- 📄 [arXiv](https://arxiv.org/abs/2503.05833) · [PDF](https://arxiv.org/pdf/2503.05833)
- 💾 [`pdfs/03-vla-robotics/Refined-Policy-Distillation_2503.05833.pdf`](pdfs/03-vla-robotics/Refined-Policy-Distillation_2503.05833.pdf)

#### One-Step Flow Policy · 视觉运动策略自蒸馏 · `2603.12480`

> **One-Step Flow Policy: Self-Distillation for Fast Visuomotor Policies** · *2026*

把 flow-based visuomotor policy 通过 **自蒸馏**压成单步生成器，部署延迟显著下降。VLA / robot policy 领域的非自回归生成 + OPD 思想结合。

- 📄 [arXiv](https://arxiv.org/abs/2603.12480) · [PDF](https://arxiv.org/pdf/2603.12480) · [HTML](https://arxiv.org/html/2603.12480v1)
- 💾 [`pdfs/03-vla-robotics/One-Step-Flow-Policy_2603.12480.pdf`](pdfs/03-vla-robotics/One-Step-Flow-Policy_2603.12480.pdf)

#### SDM Policy · 分数与分布双匹配 · `2412.09265`

> **Score and Distribution Matching Policy: Advanced Accelerated Visuomotor Policies via Matched Distillation** · *2024*

把扩散视觉运动策略 (Diffusion Policy) 压成单步生成器：**score matching** 对齐动作分布 + **distribution matching** 最小化 KL 散度，是后续 VLA 蒸馏工作的理论起点之一。

- 📄 [arXiv](https://arxiv.org/abs/2412.09265) · [PDF](https://arxiv.org/pdf/2412.09265) · [HTML](https://arxiv.org/html/2412.09265v3)
- 💾 [`pdfs/03-vla-robotics/SDM-Policy_2412.09265.pdf`](pdfs/03-vla-robotics/SDM-Policy_2412.09265.pdf)

---

### 🎨 04 · 扩散模型 / 流匹配（非自回归生成）

#### AnyFlow · NVIDIA · 任意步视频扩散 · `2605.13724`

> **AnyFlow: Any-Step Video Diffusion Model with On-Policy Flow Map Distillation**
> *Yuchao Gu, Guian Fang, Yuxin Jiang, Weijia Mao, Song Han, Han Cai, Mike Zheng Shou*
> *NVIDIA · Show Lab @ NUS · MIT · 2026*

第一个 **基于 Flow Map 的任意步视频扩散蒸馏框架**，优化整条 ODE 采样轨迹。核心创新 **Flow Map Backward Simulation**——把完整 Euler rollout 分解为 shortcut flow-map transitions，实现高效 on-policy 蒸馏。1.3B → 14B 双向/因果架构全部验证。

- 📄 [arXiv](https://arxiv.org/abs/2605.13724) · [PDF](https://arxiv.org/pdf/2605.13724) · [项目主页](https://nvlabs.github.io/AnyFlow/) · [代码](https://github.com/NVlabs/AnyFlow)
- 💾 [`pdfs/04-diffusion-flow/AnyFlow_2605.13724.pdf`](pdfs/04-diffusion-flow/AnyFlow_2605.13724.pdf)

#### D-OPSD · 步蒸馏扩散模型的 On-Policy 自蒸馏 · `2605.05204`

> **D-OPSD: On-Policy Self-Distillation for Continuously Tuning Step-Distilled Diffusion Models** · *2026*

把 OPSD（On-Policy Self-Distillation）思想从自回归 LLM **首次完整迁移到扩散模型**。同一模型扮演 teacher (条件 = 文本 + 目标图像多模态特征) 与 student (条件 = 仅文本)。专门解决：步蒸馏模型用普通 SFT 微调会破坏 few-step 生成能力的难题。

- 📄 [arXiv](https://arxiv.org/abs/2605.05204) · [PDF](https://arxiv.org/pdf/2605.05204) · [HTML](https://arxiv.org/html/2605.05204v1) · [HuggingFace](https://huggingface.co/papers/2605.05204)
- 💾 [`pdfs/04-diffusion-flow/D-OPSD_2605.05204.pdf`](pdfs/04-diffusion-flow/D-OPSD_2605.05204.pdf)

#### Flow-OPD · Flow Matching 模型的 OPD 框架 · `2605.08063`

> **Flow-OPD: On-Policy Distillation for Flow Matching Models** · *2026*

第一个把 OPD 整合进 **Flow Matching** 的统一后训练框架。两阶段：(1) 单任务 GRPO 微调出领域专家 teacher → (2) Flow-based Cold-Start 建立鲁棒初始策略，三步编排"on-policy 采样 → 任务路由 → 密集轨迹监督"融合多专家。再加 **Manifold Anchor Regularization (MAR)**——任务无关 teacher 锚定生成在高质量流形上。

- 📄 [arXiv](https://arxiv.org/abs/2605.08063) · [PDF](https://arxiv.org/pdf/2605.08063) · [HTML](https://arxiv.org/html/2605.08063v1) · [代码](https://github.com/CostaliyA/Flow-OPD) · [HuggingFace](https://huggingface.co/papers/2605.08063)
- 💾 [`pdfs/04-diffusion-flow/Flow-OPD_2605.08063.pdf`](pdfs/04-diffusion-flow/Flow-OPD_2605.08063.pdf)

#### π-Flow · 基于策略的少步生成 · `2510.14974`

> **pi-Flow: Policy-Based Few-Step Generation via Imitation Distillation** · *2025–2026*

修改 flow 模型输出层，在单个时间步预测一个 **network-free policy**。在 FLUX.1-12B 和 Qwen-Image-20B (4 NFEs) 上，比 SOTA DMD 模型 **多样性显著提升**，同时保持教师级质量。

- 📄 [arXiv](https://arxiv.org/abs/2510.14974) · [PDF](https://arxiv.org/pdf/2510.14974)
- 💾 [`pdfs/04-diffusion-flow/pi-Flow_2510.14974.pdf`](pdfs/04-diffusion-flow/pi-Flow_2510.14974.pdf)

#### Align Your Flow · NVIDIA · 连续时间 Flow Map 蒸馏 · `2506.14603`

> **Align Your Flow: Scaling Continuous-Time Flow Map Distillation** · *NVIDIA Toronto AI · 2025*

把 flow map 蒸馏目标推广到 **连续时间**，统一 consistency model 与 flow matching 的训练目标。是 AnyFlow 的理论先行工作之一。

- 📄 [arXiv](https://arxiv.org/abs/2506.14603) · [PDF](https://arxiv.org/pdf/2506.14603) · [项目主页](https://research.nvidia.com/labs/toronto-ai/AlignYourFlow/)
- 💾 [`pdfs/04-diffusion-flow/Align-Your-Flow_2506.14603.pdf`](pdfs/04-diffusion-flow/Align-Your-Flow_2506.14603.pdf)

#### DMDR · DMD 遇上 RL · `2511.13649`

> **Distribution Matching Distillation Meets Reinforcement Learning** · *2025–2026*

**DMDR** 框架：联合优化 RL 和 DMD 目标。RL 让蒸馏 preference-aware（不再均匀压缩全分布）；DMD 反过来作为 reward hacking 的正则化项。少步扩散模型对齐人类偏好的新做法。

- 📄 [arXiv](https://arxiv.org/abs/2511.13649) · [PDF](https://arxiv.org/pdf/2511.13649) · [HTML](https://arxiv.org/html/2511.13649v2) · [代码](https://github.com/vvvvvjdy/dmdr)
- 💾 [`pdfs/04-diffusion-flow/DMDR_2511.13649.pdf`](pdfs/04-diffusion-flow/DMDR_2511.13649.pdf)

---

### 🧩 05 · 离散扩散 / Diffusion LM

#### dUltra · 扩散语言模型的 OPD · `2512.21446`

> **dUltra: Ultra-Fast Diffusion Language Models via Reinforcement Learning** · *2025*

基于 GRPO 的 on-policy distillation 框架，**同时优化扩散模型与 unmasking 顺序规划器**。提出"on-policy RL + on-policy distillation reward"组合，学习最优 unmasking 策略。把 OPD 思想首次系统应用到离散扩散语言模型。

- 📄 [arXiv](https://arxiv.org/abs/2512.21446) · [PDF](https://arxiv.org/pdf/2512.21446) · [HTML](https://arxiv.org/html/2512.21446)
- 💾 [`pdfs/05-discrete-diffusion/dUltra_2512.21446.pdf`](pdfs/05-discrete-diffusion/dUltra_2512.21446.pdf)

---

### 📊 99 · Survey

#### A Survey of On-Policy Distillation for LLMs · `2604.00626`

> **A Survey of On-Policy Distillation for Large Language Models** · *2026*

全面综述：把 OPD 形式化为"在学生采样轨迹上做 f-divergence 最小化"，沿 **三条设计轴** 组织文献——优化什么、信号从哪来、如何稳定训练。

- 📄 [arXiv](https://arxiv.org/abs/2604.00626) · [PDF](https://arxiv.org/pdf/2604.00626) · [HTML](https://arxiv.org/html/2604.00626)
- 💾 [`pdfs/99-survey/OPD-Survey_2604.00626.pdf`](pdfs/99-survey/OPD-Survey_2604.00626.pdf)

---

## 📝 技术解读（[`references/`](references/) 下为已抓取的完整 markdown）

### A · 入门 / 概览

| 解读 | 来源 | 本地存档 |
|------|------|--------|
| **On-Policy Distillation**（官方原始博客）| [Thinking Machines Lab — Kevin Lu, 2025-10-27](https://thinkingmachines.ai/blog/on-policy-distillation/) | [`05_thinking-machines_on-policy-distillation.md`](references/05_thinking-machines_on-policy-distillation.md) |
| **OPD 是什么？如何做？**（最易上手）| [青稞AI](https://qingkeai.online/archives/How%20do%20On-Policy%20Distillation) · [知乎-kxzxvbk](https://zhuanlan.zhihu.com/p/2000612721868177979) | [`04_qingkeai_how-do-opd.md`](references/04_qingkeai_how-do-opd.md) |
| Aikipedia: On-Policy Distillation | [Champaign Magazine](https://champaignmagazine.com/2025/10/29/aikipedia-on-policy-distillation/) | – |
| On-Policy Distillation by Thinking Machines Lab | [Medium — ML Point](https://medium.com/@ml-point/on-policy-distillation-by-thinking-machines-lab-13028e770c4f) | – |
| Mira Murati — On-Policy Distillation: Cheap Accuracy, Real Gains | [Medium — Mahesh Lambe](https://medium.com/@maheshlambe/mira-murati-thiking-machines-on-policy-distillation-cheap-accuracy-real-gains-85ac523b20f7) | – |
| LLM Knowledge Distillation Explained for On-Device AI | [Enclave AI](https://enclaveai.app/blog/2026/03/29/llm-knowledge-distillation-on-device-explained/) | – |
| 告别昂贵的 RL？OPD：1/10 成本实现更强 Post-Training | [LittleFish'Blog](https://www.xiaoiluo.com/article/on-policy-distillation) | – |
| OPD 解读 | [知乎](https://zhuanlan.zhihu.com/p/1966521167360790856) · [机器学习POD 转载](https://www.mlpod.com/1217.html) | – |
| Thinking Machines Lab: On-Policy Distillation | [知乎](https://zhuanlan.zhihu.com/p/1966866454017209256) | – |
| OPD 核心解读 | [知乎](https://zhuanlan.zhihu.com/p/1989738134242603245) | – |

### B · 深度原理 / Self-Distillation

| 解读 | 来源 | 本地存档 |
|------|------|--------|
| **OPD / Self-Distillation 深度解读**（公式 + OPSD）| [青稞AI](https://qingkeai.online/archives/On-Policy-Distillation) · [知乎-一木不](https://zhuanlan.zhihu.com/p/2004306938188537902) | [`01_qingkeai_on-policy-distillation-deep-dive.md`](references/01_qingkeai_on-policy-distillation-deep-dive.md) |
| **七篇论文！深度理解 OPD 与 Self-Distillation** | [青稞AI](https://qingkeai.online/archives/20260304) · [知乎-Escapist](https://zhuanlan.zhihu.com/p/2009410859999437740) | [`06_qingkeai_seven-papers-opd-deep-dive.md`](references/06_qingkeai_seven-papers-opd-deep-dive.md) |
| **三大流派！9 篇近期工作综述** | [青稞AI](https://qingkeai.online/archives/On-Policy%20Distillation) | [`07_qingkeai_three-schools-9-papers.md`](references/07_qingkeai_three-schools-9-papers.md) |
| 大模型知识蒸馏：OPD（原理篇）| [知乎](https://zhuanlan.zhihu.com/p/2023108953026883794) | – |
| 同一个模型，两种身份：OPSD 的三种实践 | [知乎](https://zhuanlan.zhihu.com/p/2008270892493473005) | – |
| OPSD：通过自蒸馏提升 LLM 推理能力 | [知乎](https://zhuanlan.zhihu.com/p/2003773186135831026) | – |
| Self-Distilled Reasoner（作者博客）| [Siyan Zhao](https://siyan-zhao.github.io/blog/2026/opsd/) | – |
| MIT 提出 SDFT：作为逆 RL 的在线自蒸馏 | [知乎](https://zhuanlan.zhihu.com/p/2001428154242320351) | – |

### C · 失败模式 / 改进 / 机制分析

| 解读 | 来源 | 本地存档 |
|------|------|--------|
| **Rethinking OPD：现象、机制与配方**（最深入的失败模式分析）| [青稞AI](https://qingkeai.online/archives/Rethinking-OPD) · [代码 thunlp/OPD](https://github.com/thunlp/OPD) | [`02_qingkeai_rethinking-opd.md`](references/02_qingkeai_rethinking-opd.md) |
| 重新思考 On-Policy 蒸馏：训练动态、内在机制及工程实践 | [知乎](https://zhuanlan.zhihu.com/p/2033318685851562798) | – |
| 用 RL 做知识蒸馏，方差太大怎么办？(KETCHUP) | [青稞AI](https://qingkeai.online/archives/KETCHUP) | – |
| 从"手推策略梯度定理"开始：理解 RL 的创新本质 | [青稞AI](https://qingkeai.online/archives/%E6%89%8B%E6%8E%A8%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E5%AE%9A%E7%90%86) | – |
| 大模型 RL 算法梳理：从全量词元到部分词元的路径演化 | [青稞AI](https://qingkeai.online/archives/LLM-RL) | – |

### D · 工业实践（Qwen3 / GLM-5 / DeepSeek-V4 / Xiaomi-MiMo）

| 解读 | 来源 | 本地存档 |
|------|------|--------|
| **从 DeepSeek-V4 的多专家 OPD 反观人类学习** | [青稞AI](https://qingkeai.online/archives/DeepSeek-V4-OPD) | [`03_qingkeai_deepseek-v4-opd.md`](references/03_qingkeai_deepseek-v4-opd.md) |
| 从技术报告看 OPD 的崛起：后训练新范式 | [知乎](https://zhuanlan.zhihu.com/p/2031101471563962191) | – |
| OPD 深度研究：后训练的新范式 + 四家大厂的工程博弈 | [智柴网](https://zhichai.net/topic/177619070) | – |
| Thinking Machines 最新研究如何复现？OPD 让训练成本直降 10 倍 | [知乎](https://zhuanlan.zhihu.com/p/1967547671804875374) · [魔搭转载](https://modelscope.csdn.net/69042ade0e4c466a32e33126.html) | – |
| OPD：强化学习与蒸馏的完美融合，LLM 训练效率提升 30 倍 | [知乎](https://zhuanlan.zhihu.com/p/1967387150015259097) | – |
| Qwen3 技术报告：Strong-to-Weak Distillation + 代码 | [CSDN](https://blog.csdn.net/flyfish1986/article/details/150338197) | – |
| Qwen3 技术报告解读 | [知乎](https://zhuanlan.zhihu.com/p/1906842277260821306) | – |
| OPD：让小模型在实战中成长 | [知乎](https://zhuanlan.zhihu.com/p/1974209478040757587) | – |
| 大模型的 OPD（在线蒸馏策略）| [CSDN](https://blog.csdn.net/qq_16763983/article/details/154883404) | – |
| 只要 RL 1/10 成本！翁荔的 Thinking Machines 盯上 Qwen 的黑科技 | [智源社区](https://hub.baai.ac.cn/view/49888) | – |

### E · 工具 / 代码 / 框架

| 资源 | 类型 |
|------|------|
| [Thinking Machines — Tinker Cookbook (distillation)](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation) | 官方 OPD 训练实现，~100 行 |
| [HuggingFace TRL — General On-Policy Logit Distillation](https://huggingfaceh4-on-policy-distillation.hf.space/) | TRL 库支持任意 tokenizer 跨族蒸馏 |
| [HuggingFace Space — OPD 演示](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation) | 可视化交互演示 |
| [thunlp/OPD](https://github.com/thunlp/OPD) | Rethinking OPD 论文配套代码 |
| [shawnli/on-policy-distillation-research](https://github.com/shawnli/on-policy-distillation-research) | 中文社区研究笔记 + 复现 |
| [NVlabs/AnyFlow](https://github.com/NVlabs/AnyFlow) | 视频扩散 OPD 官方实现 |
| [CostaliyA/Flow-OPD](https://github.com/CostaliyA/Flow-OPD) | Flow Matching OPD 官方实现 |
| [vvvvvjdy/dmdr](https://github.com/vvvvvjdy/dmdr) | DMD + RL (DMDR) 官方实现 |

### F · 相关 Survey / 重要基础工作

| 资源 | 链接 |
|------|------|
| **A Survey of On-Policy Distillation for LLMs** | <https://arxiv.org/abs/2604.00626>（本仓库已收录）|
| **Rethinking OPD of LLMs: Phenomenology, Mechanism, and Recipe** | <https://arxiv.org/abs/2604.13016> |
| **Unmasking On-Policy Distillation: Where It Helps, Where It Hurts, and Why** | <https://arxiv.org/abs/2605.10889v1> |
| **Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes** | <https://arxiv.org/abs/2603.25562> |
| **KL for a KL: On-Policy Distillation with Control Variate Baseline** | <https://arxiv.org/abs/2605.07865> |
| **Asymmetric On-Policy Distillation: Bridging Exploitation and Imitation** | <https://arxiv.org/abs/2605.06387> |
| **Black-Box On-Policy Distillation of LLMs** | <https://arxiv.org/abs/2511.10643> |
| **MiniLLM** (开山之作之一) | <https://arxiv.org/abs/2306.08543> |
| **GKD — On-Policy Distillation of Language Models** (Agarwal+, 2023) | <https://arxiv.org/abs/2306.13649> |
| **One-step Diffusion with DMD** (DMD 原论文) | <https://arxiv.org/abs/2311.18828> |
| **Score Distillation of Flow Matching Models** | <https://arxiv.org/abs/2509.25127> |
| **SenseFlow: Scaling Distribution Matching for Flow-based T2I Distillation** | <https://arxiv.org/abs/2506.00523> |
| **Decoupled DMD: CFG Augmentation as the Spear** | <https://arxiv.org/abs/2511.22677> |
| **Adversarial Distribution Matching for Diffusion Distillation** | <https://arxiv.org/abs/2507.18569> |
| **Continuous-Time Distribution Matching for Few-Step Diffusion** | <https://arxiv.org/abs/2605.06376> |
| **How to build a consistency model: Learning flow maps via self-distillation** | <https://arxiv.org/abs/2505.18825> |
| **Stabilizing Consistency Training: A Flow Map Analysis** | <https://arxiv.org/abs/2601.22679> |
| **Hindsight Distillation Reasoning for KB-VQA** | <https://arxiv.org/abs/2511.11132> |

---

## 🧭 推荐阅读顺序

**最短路径（30 分钟入门）**：
1. [Thinking Machines 官方博客](references/05_thinking-machines_on-policy-distillation.md) — 一手定义 + 代码
2. [青稞AI: OPD 是什么？如何做？](references/04_qingkeai_how-do-opd.md) — 中文最易上手
3. 读 [`pdfs/01-llm-reasoning/Lightning-OPD`](pdfs/01-llm-reasoning/Lightning-OPD_2604.13010.pdf) — 工业级最快配方

**多模态主线（重点）**：
1. [`pdfs/02-vlm-mllm/VOLD`](pdfs/02-vlm-mllm/VOLD_2510.23497.pdf) — LLM→VLM 推理迁移
2. [`pdfs/02-vlm-mllm/Uni-OPD`](pdfs/02-vlm-mllm/Uni-OPD_2605.03677.pdf) — LLM/MLLM 统一框架
3. [`pdfs/02-vlm-mllm/Video-OPD`](pdfs/02-vlm-mllm/Video-OPD_2602.02994.pdf) — TVG 应用
4. [`pdfs/02-vlm-mllm/RAL`](pdfs/02-vlm-mllm/RAL_2602.04884.pdf) — 跨模态 attention 蒸馏

**VLA / Robotics 主线**：
1. [`pdfs/03-vla-robotics/SDM-Policy`](pdfs/03-vla-robotics/SDM-Policy_2412.09265.pdf) — 视觉运动蒸馏起点
2. [`pdfs/03-vla-robotics/Refined-Policy-Distillation`](pdfs/03-vla-robotics/Refined-Policy-Distillation_2503.05833.pdf) — VLA generalist + RL expert
3. [`pdfs/03-vla-robotics/VLA-OPD`](pdfs/03-vla-robotics/VLA-OPD_2603.26666.pdf) — VLA + OPD 完整框架
4. [`pdfs/03-vla-robotics/One-Step-Flow-Policy`](pdfs/03-vla-robotics/One-Step-Flow-Policy_2603.12480.pdf) — 流策略自蒸馏

**非自回归生成主线（扩散 / 流匹配）**：
1. [`pdfs/04-diffusion-flow/Align-Your-Flow`](pdfs/04-diffusion-flow/Align-Your-Flow_2506.14603.pdf) — 连续时间 flow map
2. [`pdfs/04-diffusion-flow/D-OPSD`](pdfs/04-diffusion-flow/D-OPSD_2605.05204.pdf) — OPSD 首迁移到扩散
3. [`pdfs/04-diffusion-flow/Flow-OPD`](pdfs/04-diffusion-flow/Flow-OPD_2605.08063.pdf) — Flow Matching 完整 OPD 框架
4. [`pdfs/04-diffusion-flow/AnyFlow`](pdfs/04-diffusion-flow/AnyFlow_2605.13724.pdf) — 任意步视频扩散
5. [`pdfs/04-diffusion-flow/pi-Flow`](pdfs/04-diffusion-flow/pi-Flow_2510.14974.pdf) — 策略式少步图像生成
6. [`pdfs/04-diffusion-flow/DMDR`](pdfs/04-diffusion-flow/DMDR_2511.13649.pdf) — DMD + RL 联合优化
7. [`pdfs/05-discrete-diffusion/dUltra`](pdfs/05-discrete-diffusion/dUltra_2512.21446.pdf) — 离散扩散语言模型

**完整深度路径**：
- 上面所有 + 4 篇深度解读
  - [青稞AI: OPD / Self-Distillation 深度解读](references/01_qingkeai_on-policy-distillation-deep-dive.md)
  - [青稞AI: 七篇论文深度理解](references/06_qingkeai_seven-papers-opd-deep-dive.md)
  - [青稞AI: 三大流派 9 篇近期工作](references/07_qingkeai_three-schools-9-papers.md)
  - [青稞AI: Rethinking OPD](references/02_qingkeai_rethinking-opd.md)
  - [青稞AI: DeepSeek-V4 多专家 OPD](references/03_qingkeai_deepseek-v4-opd.md)
- 收尾读 Survey [`pdfs/99-survey/OPD-Survey`](pdfs/99-survey/OPD-Survey_2604.00626.pdf)

---

## 🗂 仓库结构

```
opd-papers/
├── README.md
├── pdfs/                                                # 17 篇原始论文 PDF
│   ├── 01-llm-reasoning/
│   │   └── Lightning-OPD_2604.13010.pdf
│   ├── 02-vlm-mllm/                                     # 视觉-语言 / 多模态
│   │   ├── VOLD_2510.23497.pdf
│   │   ├── Uni-OPD_2605.03677.pdf
│   │   ├── Video-OPD_2602.02994.pdf
│   │   └── RAL_2602.04884.pdf
│   ├── 03-vla-robotics/                                 # 视觉-语言-动作
│   │   ├── VLA-OPD_2603.26666.pdf
│   │   ├── Refined-Policy-Distillation_2503.05833.pdf
│   │   ├── One-Step-Flow-Policy_2603.12480.pdf
│   │   └── SDM-Policy_2412.09265.pdf
│   ├── 04-diffusion-flow/                               # 扩散 / 流匹配（非自回归生成）
│   │   ├── AnyFlow_2605.13724.pdf
│   │   ├── D-OPSD_2605.05204.pdf
│   │   ├── Flow-OPD_2605.08063.pdf
│   │   ├── pi-Flow_2510.14974.pdf
│   │   ├── Align-Your-Flow_2506.14603.pdf
│   │   └── DMDR_2511.13649.pdf
│   ├── 05-discrete-diffusion/                           # 离散扩散 / Diffusion LM
│   │   └── dUltra_2512.21446.pdf
│   └── 99-survey/
│       └── OPD-Survey_2604.00626.pdf
└── references/                                          # 7 篇高质量中英文解读
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
- **解读**：`references/` 下的 markdown 都标注了原始来源链接和作者；如需正式引用请回到原始链接。知乎需要登录无法直接抓取，README 中给出链接但未做本地存档。
- **更新时间**：2026-05-15（17 篇 PDF + 7 篇解读）。
