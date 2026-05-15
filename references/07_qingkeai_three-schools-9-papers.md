# On-Policy Distillation 三大流派：一个方法解决两道难题

**来源**: 青稞 AI (qingkeai.online)
**原文链接**: <https://qingkeai.online/archives/On-Policy%20Distillation>
**抓取时间**: 2026-05-15

> 涵盖近半年 **9 篇** OPD 论文，按"训练稳定 / 自蒸馏 / 跨模态"三大主线串联。是目前最系统的近期工作综述之一。

---

## 痛点与破局

RL 训练过的人都知道那种感觉——跑了三天 GRPO，token 消耗是 SFT 的十几倍，最后一看提升，几乎为零。问题出在哪？Reward 信号太稀疏，credit assignment 做不到 token 级，rollout 全错时梯度直接归零。

但 SFT 也有自己的软肋：推理时遇到没见过的分布，模型容易"自信爆棚"，错得离谱。

2025 年下旬，Thinking Machines Lab 给出了一个折中方案——**On-Policy Distillation**。学生在自己的轨迹上接受 teacher 的分布监督，既保留了 on-policy 的零 exposure bias，又多了 token 级的密集信号。

这个思路出来后，社区迅速跟进。近半年 9 篇工作值得关注，大致分成三个方向。

---

## 一：让训练更稳定

### 1.1 Veto — 用几何空间解决崩溃问题

**论文**: *Stable On-Policy Distillation through Adaptive Target Reformulation* — <https://arxiv.org/abs/2601.07155>

标准做法用 KL 散度，但两个极端都很麻烦：
- **Forward KL**: 梯度在"学生无知、老师自信"的位置爆炸，量级冲到 10⁶，训练秒崩
- **Reverse KL**: mode-seeking，只学老师最确定的那条路，多样性归零

Veto 在 **logit 空间里做插值**，用参数 α 在 teacher 和 student 之间做几何平均：

```
P_target = (1-α)·P_T + α·P_S
```

α 是个旋钮：在 forward KL 场景下压制梯度爆炸，在 reverse KL 场景下控制熵正则强度。

**结论**：训练不稳定的根因不在数据质量，而在散度目标本身的几何特性。

### 1.2 EOPD — 高熵位置额外加 Forward KL

**论文**: *Entropy-Aware On-Policy Distillation of Language Models* — <https://arxiv.org/pdf/2603.07079>

Qwen3-8B→1.7B 蒸馏实验：teacher 产生的 token 中 18.5% 处于高熵区，但学生训练完后只保留了 6.8%。高熵区恰是推理的关键分叉点，学生在这些位置坍缩直接导致输出多样性骤降。

按 token 级熵动态切换散度目标：
- 熵低 → reverse KL，快速稳定
- 熵高 → 叠加 forward KL，保护 teacher 的分布形态

| 模型 | Avg@8 提升 | Pass@8 提升 |
|------|-----------|-----------|
| Qwen3-0.6B | +1.16 | +1.37 |
| Qwen3-1.7B | +0.99 | +2.39 |
| Qwen3-4B | +1.80 | **+5.05** |

### 1.3 REOPOLD — 把 RL 的工程技巧搬进来

**论文**: *Scaling Reasoning Efficiently via Relaxed On-Policy Distillation* — <https://arxiv.org/pdf/2603.11137>

两个贡献：

1. **证明带 stop-gradient 的 on-policy distillation 数学上等价于 on-policy policy gradient**。teacher-student 的 log-likelihood 比值天然就是 token 级 reward。
2. **把 RL 里的成熟技巧搬过来**：
   - Reward clipping：直接裁 reward 值，防止极端负值主导梯度
   - 熵引导采样：只在熵排名 top-p% 的 token 上算梯度
   - 两阶段训练：前期模仿 SFT 鼓励探索，后期切到熵掩码聚焦高不确定性 token

AIME-25 Pass@1 约 32–34%，样本效率比 ProRL、Still-3-1.5B 等高 **6.7–12 倍**。

---

## 二：自己当自己的老师

### 2.1 OPSD — 同一模型，加个答案就变 teacher

**论文**: *Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models* — <https://arxiv.org/pdf/2601.18734>

当模型"看到"正确答案时，它能倒推出高质量的推理过程：
- **Teacher**: 同一模型，条件是"问题 + 正确答案"
- **Student**: 同一模型，条件只有问题

OPSD 把生成长度上限设为 1,024 tokens，GRPO 用 16,384。两者精度持平，但 OPSD token 消耗只有 GRPO 的 **1/8 到 1/12**。

| 属性 | SFT | GRPO | OPSD |
|------|-----|------|------|
| On-policy | ✗ | ✓ | ✓ |
| 密集 token 级监督 | ✓ | ✗ | ✓ |
| 低采样成本 | ✓ | ✗ | ✓ |
| 不依赖外部 teacher | ✓ | ✓ | ✓ |

### 2.2 SDFT — 自蒸馏做持续学习

**论文**: *Self-Distillation Enables Continual Learning* — <https://arxiv.org/abs/2601.19897>

把专家示例变成 in-context 特权信息：
- Teacher：见过示例后的条件分布
- Student：没见过示例的基础分布

数学上等价于**隐式 IRL**——reward 被编码在 teacher 的 log probability 里，不需要单独训练 reward 模型。

依次训三个任务：工具使用 → 科学问答 → 医疗问答。SFT 每次学新任务就忘旧的，SDFT 三个任务训完后性能基本不下降。

### 2.3 SDPO — 环境反馈本身就是信号

**论文**: *Reinforcement Learning via Self-Distillation* — <https://arxiv.org/abs/2601.20802>

GRPO 只用一个标量 reward (0/1)，代码报错的一大段文字信息全丢了。

SDPO 的做法：让模型看到错误反馈后，重新评估自己之前的输出：
- Student: 原始生成 $\pi_\theta(a|s)$
- Self-teacher: 看到反馈 f 后重新打分 $\pi_\phi(a|s,f)$

每个 token 的 advantage 就是这两个分布的 log prob 差值。额外开销只有 **+5.8%**。

效果：达到同等精度需要的生成次数减少 **4 倍**；Olmo3-7B 化学任务，30 分钟达到 GRPO 5 小时的水平；回答长度比 GRPO 短 3–7 倍，精度还更高。

### 2.4 OPSDC — 想得越多错得越多，压缩反而提升精度

**论文**: *On-Policy Self-Distillation for Reasoning Compression* — <https://arxiv.org/abs/2603.05433>

**反直觉一幕**：o1、DeepSeek-R1、Qwen3 在简单问题上也会生成几千 token 的内部独白，大量 token 不必要，更糟糕的是它们还是错误的来源。

论文证明：把推理链从 n 压缩到 m 个 token，准确率提升比例与 **(n/m)²** 成正比，**指数级改善**。

OPSDC 不需要正确答案也不需要 reward，只需要一个"请简洁"的指令：
- Teacher: 带指令的同一模型
- Student: 不带指令的版本
- 每隔 K 步同步一次权重

30K token 预算下，推理 token **减少 40–58%**，准确率反而 **高 10–16 个百分点**。

---

## 三：从文本延伸到更多场景

### 3.1 OPCD — 把 prompt 里的人类经验烧进模型参数

**论文**: *On-Policy Context Distillation for Language Models* — <https://arxiv.org/abs/2602.12275>

很多人花大量时间做 prompt engineering，或者积累了高质量历史解题记录。问题是这些信息只存在于 context 里，一旦清空就没了。

OPCD：student 生成轨迹，teacher 带着 context 在这条轨迹上评分，用 reverse KL 更新。

| 模型 | 基线 | In-Context | Off-Policy CD | **OPCD** |
|------|------|-----------|---------------|------|
| Llama-3.1-8B-Instruct | 68.4% | 72.2% | 75.2% | **76.7%** |
| Qwen2.5-7B-Instruct | 46.4% | 52.6% | 58.5% | **62.3%** |

某些情况下甚至超过 in-context 本身。

### 3.2 Video-OPD — 视频时序定位任务上的 OPD

**论文**: *Video-OPD: Efficient Post-Training of Multimodal Large Language Models for Temporal Video Grounding via On-Policy Distillation* — <https://arxiv.org/abs/2602.02994>

TVG 任务：给视频和自然语言问题，输出时间段 [t_start, t_end]。

Video-OPD 用 32B teacher 对 8B student 做 token 级评分。每个 token 的 reward 考虑正确性和时序邻近性。Teacher 只评分不生成，保证 on-policy 特性。

加上两个训练策略：
- **TRPV**: 用 ground-truth IoU 过滤 teacher 不可靠的预测
- **DBTP**: 优先训练 teacher-student 分歧最大的样本

| Benchmark | GRPO | Video-OPD |
|-----------|------|-----------|
| Charades-TimeLens | 27.6 | **32.4** |
| ActivityNet-TimeLens | 32.1 | **35.8** |
| QVHighlights-TimeLens | 41.5 | **50.4** |

平均提升 **17%**，超过 GPT-4o、GPT-5、Gemini-2.0-Flash，接近 Gemini-2.5-Flash。

---

## 写在最后

on-policy distillation 解决的是一道经典难题：**如何同时获得 RL 的零 exposure bias 和 SFT 的密集监督**。

落地速度比想象中快。小米的 MiMo 系列、Hugging Face 的 "Unlocking On-Policy Distillation for Any Model Family" 都已经在用了。

趋势很明显：**更强的 teacher、更低的算力门槛、自蒸馏不需要外部模型**。
