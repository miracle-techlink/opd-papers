# Rethinking On-Policy Distillation of Large Language Models: 现象、机制与配方

**来源**: 青稞 AI (qingkeai.online)
**原文链接**: <https://qingkeai.online/archives/Rethinking-OPD>
**对应论文**: <https://arxiv.org/abs/2604.13016> · 代码 <https://github.com/thunlp/OPD>
**抓取时间**: 2026-05-15

---

## 前言

从 Qwen3、Mimo-Flash 到 GLM-5 系列，业界已达成共识——让学生模型自己生成轨迹，然后用更强教师模型的 per-token log-prob 作为密集奖励来引导更新。

在直觉上，这种做法看似完美。它绕开了离线蒸馏中的暴露偏差问题，让模型在自己真正会访问的状态空间里获得纠偏信号。

10月底，Thinking Machines Lab 提出的 OPD (On-Policy Distillation) 同时结合了强化学习的在线特性与知识蒸馏的密集奖励特性。展示数据表明该方法的蒸馏效率远高于传统 SFT，且不会发生灾难性遗忘——看似是 SFT 的完美平替。

但现实中 OPD 的训练过程往往不尽如人意——在哪种条件下能跑通，在哪种条件下会失败，其实并没有系统性的答案。

本工作从三个层面递进推进：

**第一层是"现象学"** (Phenomenology)：通过对照实验，归纳出 OPD 成功的两个前提条件——老师和学生要有相容的思维模式，以及老师必须给学生带来真正新的知识。

**第二层是"机制"** (Mechanism)：深入到 token level，成功的 OPD 有清晰的动态特征：训练过程中，学生和老师在高概率 token 上的重合会持续扩大，这些重合 token 承载了 97% 到 99% 的概率质量，也集中了几乎全部有效的梯度信号。

**第三层是"配方"** (recipe)：提出两个实用的策略来改进 OPD 的效果。

---

## OPD Framework

原始的 OPD 使用 teacher 和 student 的 log-prob 差值来估计 Reverse-KL。一个自然的改进是在每个位置处使用 top-k token 之间的 KL 来做估计：

$$\widetilde{D}_{\mathrm{KL}}^{(S_t)}(p_t \| q_t) \triangleq \sum_{v \in S_t} p_t(v) \bigl(\log p_t(v) - \log q_t(v)\bigr)$$

其中 $S_t$ 代表 student 的 top-k 候选 token。使用 top-k 估计的方差会更小，也便于观测相关的训练动态。

### 打开黑盒，直击本质

设计了一组训练动态指标：

- **Overlap Ratio**: Student 和 teacher 的 top-k 高概率 token 集合的交集比例。衡量两者在同一状态下"想说的词"有多少重叠。
- **Overlap-Token Advantage**: 在交集 token 上，teacher 和 student 的概率分布对齐程度。接近零说明 student 已经在 teacher 认可的 token 上配置了合理的概率质量。
- **Entropy Gap**: Teacher 和 student 在 student 生成路径上的熵差的绝对值。差距缩小意味着两者的"确定性程度"趋于一致。

这样做的原因是：如果不看训练过程，只能知道"有没有涨分"，但不知道训练过程中 student 的分布到底是怎么被 teacher 推着走的。

---

## OPD Phenomenology

在深入探讨 OPD 的 token 级机制之前，首先需要厘清一个核心问题：**到底是什么决定了 OPD 的有效性？**

传统的知识蒸馏经验通常认为，"Teacher 的能力越强，Student 的蒸馏上限就越高"。但在实际的 OPD 实验中，观察到了一个与直觉相悖的现象：

有时，一个在 Benchmark 上得分更高的强模型，不但无法带来可观的蒸馏收益，甚至会导致学生模型能力的退化；而某些跑分相对较低的模型，却能作为极佳的 teacher 引导学生实现性能跃升。

### Thinking pattern consistency: teacher 和 student 得先"说得到一块去"

做了一组对比实验：固定 student 为 Qwen3-1.7B-Base，teacher 选择一个是不开思考模式 (Non-thinking) 的 Qwen3-4B，一个是对 Qwen3-4B-Base 直接做 Zero-RL (GRPO) 的 Qwen3-4B-Base-GRPO。

单看数学 Benchmark 的平均得分，GRPO 老师其实是不如常规 4B 老师的。但蒸馏结果却恰恰相反，跟随 GRPO 老师训练的学生模型最终获得了更显著的性能提升。

背后的原因在于：由于 student 本身是 base 模型，它的思维模式与同样源自 base 的 GRPO teacher 更为契合。在训练初期的 Overlap Ratio（即师生在候选 top-k token 上的重合度）要高得多。

这说明，在 OPD 框架下，如果师生在思维模式上存在严重不匹配，teacher 即便具备更强的解题能力，其提供的 token level 的指导信号也很难被 student 转化为有效的梯度更新，而且这种早期不匹配造成的优化损失在后续训练中是难以弥补的。

### 高分不等于新知识 (Higher scores ≠ new knowledge)

那么，如果思维模式一致了，是不是只要 teacher model 更大、跑分更高，就万事大吉了？

事实并非如此。在 DeepSeek 和 Qwen 两个 family 里做了测试，对比了"同源但更大尺寸的 teacher" (DS-7B 和 Qwen3-4B) 和"经过额外 RL 后训练注入了新能力的 teacher" (Skywork 7B 和 Qwen3-4B-Math-RL)。

结果非常明显，如果 teacher 仅仅是把参数量从 1.5B 放大到了 7B，用的还是同一套数据和训练配方，那哪怕它的跑分比学生高，蒸馏带来的提升也微乎其微。

这意味着，在同源放大 (Scaling) 的设定下，两者的局部概率分布已经非常相似，teacher 侧几乎没有增量信息。**OPD 要想发挥威力，teacher 必须向 student 传授其在此前训练阶段未曾见过的新知识。**

### 反向蒸馏实验 (Reverse Distillation)

为了严谨地验证这两点，设计了一组反向蒸馏实验。

把一个已经由 DeepSeek-R1-Distill-Qwen-1.5B 经过 RL 得到的、能力很强的 JustRL-Deepseek-1.5B 拿来当 student，然后选择两个同族的 teacher：一个是 RL 之前的 base model (DeepSeek-R1-Distill-Qwen-1.5B)，另一个是尺寸更大的 DeepSeek-R1-Distill-Qwen-7B。注意，7B teacher 的跑分是比当前这个还要略微高的。

实验结果令人深思：无论面对哪个 teacher，JustRL-1.5B 均发生了严重的性能退化，并且两条训练曲线最终几乎精准地倒退回了同一水平（即 RL 之前的性能）。

这说明了什么？首先，7B 和 1.5B 由于出自相同的训练管线，在 student 视角的局部状态下，这俩 teacher 诱导出的目标分布几乎是不可区分的；其次，OPD 训练动态完全可以与 teacher 的 benchmark 分数解耦。

即便 7B teacher 的分数更高，由于它缺乏 RL 阶段注入的新知识，其提供的监督信号本质上是在迫使学生"遗忘"好不容易习得的强化学习策略，退化回陈旧的生成模式。

综合这部分的观察，可以得出一个结论：**OPD 并不是在单纯地"学习高分"，而是在主动提取并复刻 teacher 的概率分布模式。思维模式的一致性保证了这种复刻的可行性，而 teacher 侧真实的增量知识则决定了复刻带来的性能上限。**

---

## OPD Mechanism

### 高概率 Token 的渐进对齐 (Progressive Alignment of High-Probability Tokens)

依然沿用前面的典型对照组：以 R1-Distill-1.5B 为学生，对比成功的 teacher (JustRL-1.5B) 和失败的 teacher (R1-Distill-7B)。这一次，在训练的每一步都监控了三个核心指标：Overlap Ratio、Overlap-Token Advantage 以及 Entropy Gap。

成功的蒸馏过程展现出了一种极具标志性的"渐进对齐"特征。在成功的 OPD 中，观察到师生之间的 Overlap Ratio 随着训练步数稳步攀升（从约 72% 一路涨到了 91% 以上），同时优势值向零收敛，熵差距不断缩小。

这意味着，学生在自己生成的 trace 上，不仅逐步定位到了老师也偏好的高概率 token，而且正在精确校准自身在这些 token 上的概率，使之与老师保持一致。反观 7B teacher 的失败对照组，这三个指标从一开始就陷入停滞，几乎没有变化。

更关键的是，统计后发现，虽然师生重合的 top-k token 在词表中只占极小的一部分，但它们实际上承载了双方 97% 到 99% 的概率质量。这说明，**高概率 token 的对齐并不是细枝末节的副产物，而是核心概率分布演化的主轴。**

### 优化共享 Token 就足够了 (Optimizing Shared Tokens Alone Suffices)

把常规的 top-k 监督信号强行拆解成了互斥的两部分：一个版本仅仅在师生重合的 token 上计算蒸馏损失 (Overlap top-k)，另一个版本只在双方不重合的差异 token 上计算损失 (Non-Overlap top-k)。

实验结果给出了非常明确的答案：**仅仅在 Overlap token 上进行优化，就足以几乎完美复现标准 top-k OPD 的全部性能增益！而仅仅依靠 Non-Overlap token 训练出来的模型，性能则大幅落后。**

到这里，就看清了 OPD 的一个"飞轮效应"（Self-reinforcing dynamic）：

当师生在初始阶段存在合理的思维模式重合时（满足前面提到的条件），老师会对这些共享的 token 给出明确的打分。Reverse-KL 散度的优化目标会促使学生将概率质量更集中地倾注到这些被老师认可的重合 token 上。

随着重合 token 的概率越来越高，那些原本不重合的边缘 token 就自然而然地被"挤出"了学生的 top-k 集合。结果就是，重合区因为优化而变得更大，优化信号又因为重合区的扩大而变得更加纯粹和稳定，形成了一个良性循环。

---

## Practical Recipe

### 离线冷启动 (Off-policy Cold Start)

让 Qwen3-4B (Non-thinking) 作为 teacher 先 offline 生成一批回答，简单 filter 之后，用这批数据对 base 版本的 student model (Qwen3-1.7B-Base) 做一次 SFT。相当于在实战前，先让 student 背一背 teacher 的"优秀范文"，强行把学生的生成偏好往 teacher 的思维模式上拽。在此基础上，再启动正式的 OPD 训练。

实验结果证明，这一招极为有效。相比于从 base 模型直接硬上 OPD，经过 SFT 冷启动的 student 在各项 benchmark 上的最终表现都有了实质性的飞跃，并且最终的上限也显著更高。

透过前面提到的微观指标来看，原因非常清晰：**SFT 直接拉高了训练初始阶段的 Overlap Ratio，并且大幅压低了师生之间的 Entropy 差距。**

### 利用 Teacher 后训练 Prompt (Leveraging Teacher Post-Training Prompts)

第二招是从数据端入手：利用训练 teacher 所使用的 prompt 来做 OPD (Teacher-aligned prompt selection)。

将这种对齐拆分为两个颗粒度：**模板对齐 (template)** 和 **内容对齐 (content)**。

**Template 对齐**：把 DAPO-Math-17K 的指令模板，从原本的标准风格，换成 "Please reason step by step, and put your final answer within \\boxed{}." 模板，这个模版是 JustRL 在 RL 时使用的。问题的内容一模一样，变的只是 template。结果：teacher-aligned template 不仅验证集更高，overlap ratio 也起得更高、涨得更快。

**内容对齐**：直接复用 teacher 在做后训练时见过的那些题目。

一个更鲁棒的实操策略是：**在保留部分 teacher-aligned prompts 以获取精准的高频对齐信号的同时，一定要混入一些 OOD 的 prompt，以此来维持学生策略的多样性，防止模型因为过度拟合 teacher 的特定偏好而发生熵坍塌。**

---

## Discussion: Dense reward 不是没有代价

它的代价恰恰来自"on-policy"这三个字。teacher 每一步都在 student 自己走出来的 prefix 上打分；prefix 越长、漂移越深，teacher 给的 reward 就越不可靠。

### 奖励质量随轨迹深度而下降 (Reward Quality Degrades with Trajectory Depth)

把最大 response length 扫到 0.5K、1K、3K、7K、10K、15K。结果不是越长越好，而是很明显存在一个 sweet spot：太短，监督不够；中等长度，3K 和 7K 最稳；再往上到 10K、15K，效果开始平台甚至回落。

续写实验：从 1K prefix 接着写，teacher 还能比 student 多带来大约 +0.37 的准确率提升；截到 16K，这个优势只剩下 +0.02。

对长时程 reasoning、agentic multi-turn 任务来说，这可能就是 OPD 很难线性外推的一道硬边界。

### 全局信息量丰富的奖励不保证局部可利用性

失败的 teacher 并不是"分不清好坏"，它的全局奖励信号依然是富有信息的（AUROC 0.73-0.75）。**问题在于"局部优化几何学"(Local optimization geometry)：尽管全局信号是正确的，但该 teacher 在各位置上产生的 token 级优势值 (Advantage) 可能是高度各向异性 (Anisotropic) 的。** 当这些在不同位置上指向冲突的局部信号被聚合成一次梯度更新时，方向上发生了严重的相互抵消。

### 采样 Token 奖励已经足够 (Sampled-Token Reward Is Already Sufficient)

对比了计算 top-64、top-16、top-4、top-1，以及 Thinking Machines Lab 提到的、最常用的 sampled token OPD。出乎意料的是，**计算量最小的 sampled-token，效果竟然和拉满 top-64 几乎一样好。** 而同样只用了一个 token 的 top-1 策略，却在训练后期全盘崩溃。

Sampled-token 本质上是对应 token 级 Reverse KL 的无偏蒙特卡洛估计。

---

## 写在最后

> **我们是在整个社区对于 OPD 的热情持续高涨的时候，试图给 OPD 泼冷水。虽然免费的 dense reward 值得让我们欢欣鼓舞，但是正如天下没有免费的午餐，token 级别的监督信号并非天然可靠。**
