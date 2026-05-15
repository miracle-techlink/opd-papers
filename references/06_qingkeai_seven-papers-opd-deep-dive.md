# 七篇论文！深度理解 On-Policy Distillation 与 Self-Distillation

**来源**: 青稞 AI (qingkeai.online)
**原文链接**: <https://qingkeai.online/archives/20260304>
**原始知乎**: <https://zhuanlan.zhihu.com/p/2009410859999437740> · 作者 Escapist
**抓取时间**: 2026-05-15

---

## Introduction

### 从 SFT、RLHF/RLVR 到 On-Policy Distillation

大语言模型的后训练有两种主要方式：

**第一种：监督微调（SFT）/ off-Policy Distillation**
- 以高密度 token 监督换取训练稳定性
- 训练分布由数据集或教师采样固定给定
- 是一种 off-policy 的训练

**第二种：强化学习（RLHF/RLVR）**
- 采用 on-policy 采样直接优化当前策略的行为
- 奖励通常是稀疏、延迟且高方差的

**On-policy distillation 的关键意义**在于组合两种方法的优点：采样来自当前学生策略，监督来自教师分布的密集 token 级信号。

七篇论文的主线：GKD 的 on-policy KD 统一化 → MiniLLM 的 sequence-level reverse KL → G-OPD 的奖励外推 → OPSD/SDFT/SDPO 的自蒸馏扩展 → 工程化经验总结（Thinking Machines Lab）。

---

## Preliminary

### Forward KL 与 MLE/SFT 的等价关系

$$D_{\mathrm{KL}}(P\|Q_\theta)=\mathbb{E}_{y\sim P}\big[\log P(y)-\log Q_\theta(y)\big]$$

由于 $\mathbb{E}_{P}[\log P(y)]$ 与 θ 无关：

$$\min_\theta D_{\mathrm{KL}}(P\|Q_\theta)\Longleftrightarrow \max_\theta \mathbb{E}_{y\sim P}\big[\log Q_\theta(y)\big]$$

这正是 MLE/SFT 的形式。标准 KD 是把 one-hot 换成教师 soft label，但仍保持前向 KL 结构。

**Forward KL 特性**：
- 期望在教师/数据分布上取
- 学生给出低概率会产生明显惩罚
- 对学生在低概率区域的质量分配容忍度高
- 表现出 **mode covering**

### Reverse KL、mode-seeking 与支持集约束

$$D_{\mathrm{KL}}(Q_\theta\|P)=\mathbb{E}_{y\sim Q_\theta}\big[\log Q_\theta(y)-\log P(y)\big]$$

**Reverse KL 特性**：
- 期望测度变成了 Q_θ，具备 on-policy 属性
- 若 P(y) 极低而 Q_θ(y) 给出质量，惩罚迅速增大
- 表现出 **mode-seeking** 倾向
- 在数学、代码等"正确解峰值窄"任务中更匹配
- 依赖支持集重叠（先做领域 mid-training 再做 OPD 的理论原因）

### JSD / generalized JSD

$$\mathrm{JSD}_\alpha(P,Q)=\alpha D_{\mathrm{KL}}(P\|M_\alpha)+(1-\alpha)D_{\mathrm{KL}}(Q\|M_\alpha),\quad M_\alpha=\alpha P+(1-\alpha)Q$$

在 mode-covering 与 mode-seeking 之间连续插值，为 GKD 一类方法提供任务可调的散度族。

### 梯度分解与 bias-variance

$$\nabla_\theta \mathcal{L}(\theta) = \mathbb{E}_{y\sim q_\theta}[\nabla_\theta f_\theta(y)] + \mathbb{E}_{y\sim q_\theta}[f_\theta(y)\,\nabla_\theta \log q_\theta(y)]$$

第一项：直接梯度项。第二项：score function 项（策略梯度项）。

GKD 与 MiniLLM 的分歧本质上就是对这两项是否同时优化的不同立场。

---

## 1. GKD：广义的蒸馏框架与散度选择

把对齐位置从"数据/教师分布上的前缀"移动到"学生当前分布上的前缀"。

$$D(p_T\|p_S^\theta)(y|x)=\frac1{L_y}\sum_{t=1}^{L_y} D\!\left(p_T(\cdot|x,y_{< t})\;\|\;p_S^\theta(\cdot|x,y_{< t})\right)$$

- 监督信号来自教师，采样分布来自学生
- 获得 on-policy 且 dense 的训练信号
- 不对采样的分布求导，因此**没有策略梯度项**

可自然与 RL 目标并联：$\mathcal{L}=\mathcal{L}_{\mathrm{RL}}+\lambda\mathcal{L}_{\mathrm{GKD}}$。

---

## 2. MiniLLM：Sequence-level Reverse KL 与无偏梯度路径

$$\mathcal{L}(\theta)=D_{\mathrm{KL}}(q_\theta(y|x)\|p_T(y|x))$$

与 GKD 的两点区别：
1. **会对采样分布求导**，因此会有策略梯度项
2. **使用 Sequence-Level Reverse KL**，通过 reward-to-go 显式建模 n 步对未来推理轨迹的影响

定义序列奖励 $R_{\mathrm{seq}}(y)=\log p_T(y)-\log q_\theta(y)$，进一步利用因果性可替换为 reward-to-go：

$$R_t=\sum_{t'=t}^{T}r_{t'},\quad r_t=\log p_T(y_t|y_{< t},x)-\log q_\theta(y_t|y_{< t},x)$$

**稳定化创新**：
- **Length Normalization**: $R^{\text{Norm}}_{t+1}$ 除以剩余长度
- **Token-level KL 同步**
- **混合采样**: $\tilde{p}=\alpha p+(1-\alpha)q_\theta$
- **重要性采样**: $w_t \approx q_\theta/p$

### GKD vs MiniLLM

| 维度 | MiniLLM | GKD |
|------|---------|-----|
| 梯度 | 完整 $\mathbb{E}[f\nabla\log q]+\mathbb{E}[\nabla f]$ | 仅 $\mathbb{E}[\nabla f]$（近似） |
| 偏差 | 无偏 | 有偏 |
| 方差 | 高 | 低 |
| 稳定性 | 需要更多稳定化工程 | 更稳定 |

二者是针对不同训练预算与任务结构的估计器选择。

---

## 3. Learning beyond Teacher（G-OPD / ExOPD）：奖励外推

标准 OPD 等价于：

$$\max_\theta\\ \mathbb{E}_{y\sim \pi_\theta}\!\left[\log\pi^\*(y|x)-\log\pi_\theta(y|x)\right]$$

引入参考策略 π_ref 后可重写为：

$$\max_\theta\\ \mathbb{E}_{y\sim \pi_\theta}\!\left[ \log\frac{\pi^\*(y|x)}{\pi_{\mathrm{ref}}(y|x)} -\log\frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)} \right]$$

→ "隐式奖励 + KL 约束"的 RL 目标。隐式奖励 $r(x,y)=\log\frac{\pi^\*(y|x)}{\pi_{\mathrm{ref}}(y|x)}$ 可解释为"相对于参考模型的能力增量"。

G-OPD 引入外推系数 λ：

$$\pi_\theta^{\star}(y|x)\propto \pi_{\mathrm{ref}}(y|x)\left(\frac{\pi^\*(y|x)}{\pi_{\mathrm{ref}}(y|x)}\right)^\lambda$$

- λ=1：退化为标准 OPD
- 0<λ<1：保守插值
- **λ>1：ExOPD**，沿教师增量方向进一步外推

外推条件：
1. 教师相对参考的优势方向在学生容量内可表达
2. 外推系数不过度放大噪声
3. 采样探索仍能覆盖关键区域

---

## 4. OPSD：特权信息条件下的单模型自蒸馏（推理）

学生策略：$p_S(y_t|x,\hat y_{< t})$；教师策略：$p_T(y_t|x,y^\*,\hat y_{< t})$，其中 $y^\*$ 是标准答案。

**教师之所以更强，不是因为参数更大，而是因为条件信息更充分**。这样做把蒸馏问题重写为"同一模型内部条件能力迁移"。

实证发现：在数学推理任务上，**全词表蒸馏（忽略策略梯度项）往往比 sampled token policy-gradient（保留策略项）更有效**。原因：推理任务的局部 token 决策容错率低，低方差监督能更快逼近稳定解。

---

## 5. SDFT-CL：持续学习中的自蒸馏与抗遗忘

教师是同一模型的"带演示版本" $\pi(\cdot|x,c)$，学生是"裸问题版本" $\pi_\theta(\cdot|x)$。c 是 demonstrations。

$$\mathcal{L}_{\mathrm{SDFT}}(\theta)= D_{\mathrm{KL}}\!\left(\pi_\theta(\cdot|x)\\ \|\\ \pi(\cdot|x,c)\right)$$

由于教师来自同一模型家族且受演示锚定，学生被拉向的目标分布与原模型不会相距过远。更像信赖域更新，能在获得新能力的同时抑制参数漂移。

效果：不是绝对不遗忘，而是显著降低遗忘速率并提升新旧任务的联合可行解概率。

---

## 6. SDPO：利用 Rich Feedback 的自蒸馏强化学习

构造反馈条件教师：$\pi_T(\cdot|x,y,f)$，其中 f 是编译错误、执行日志、评测反馈等。

流程：
1. 学生先生成 y
2. 环境返回 f
3. 教师在 (x,y,f) 条件下重评每一步 token 合理性
4. 将该分布蒸馏回学生

本质是 **hindsight re-evaluation**：利用"事后知道结果"的信息把全局失败定位到局部决策。即使原环境仅输出一个标量结果，蒸馏后也可得到每个时间步的密集学习信号。

$$\mathcal{L}_{\mathrm{SDPO}}= \mathbb{E}_{y\sim \pi_S(\cdot|x)} \left[ \sum_t D\!\left(\pi_T(\cdot|x,y,f,y_{< t})\\ \|\\ \pi_S(\cdot|x,y_{< t})\right) \right]$$

适用：代码（rich feedback 天然丰富）、数学（借成功轨迹对失败轨迹进行对照式蒸馏）。

---

## 7. TML-OPD：Thinking Machines 工程实践配方

| 方法 | On-Policy 采样 | Dense 监督 |
|-----|--------------|----------|
| SFT | ✗ | ✓ |
| RL | ✓ | ✗ |
| **OPD** | **✓** | **✓** |

**信息密度**：RL 每次 rollout 仅 O(1) scalar reward；蒸馏在整个序列上 O(N) 监督。

**工程配方**：
1. 先做领域的 mid-training 扩大学生支撑
2. 再使用"mid-training 前的基座模型"作为蒸馏参照
3. 通过 OPD 把通用对话行为重新拉回，同时尽量保留领域增益

**算法选择指南**：
1. 任务可获得可靠教师分布 → 优先 OPD
2. 只能获得结果而无分布 → SDPO 式反馈教师
3. 严格序列级最优 → MiniLLM 式完整梯度 + 更多稳定化工程

---

## Conclusion

七篇工作的共同主线：**把"学生真实会经历的状态分布"与"教师提供的密集结构化信号"对齐**。

- **GKD**：强在统一与稳健
- **MiniLLM**：强在目标忠实性
- **G-OPD**：强在可调超越机制
- **OPSD/SDFT/SDPO**：强在把外部教师需求转化为条件化 Self-Distillation
- **TML-OPD**：强在工程实践配方

未来方向：自适应散度调度 / 多教师组合 / 测试时自蒸馏与在线反思。
