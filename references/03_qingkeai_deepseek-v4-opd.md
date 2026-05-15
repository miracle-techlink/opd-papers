# 从 DeepSeek V4 的多专家 on-policy Distillation 反观人类学习模式

**来源**: 青稞 AI (qingkeai.online)
**原文链接**: <https://qingkeai.online/archives/DeepSeek-V4-OPD>
**抓取时间**: 2026-05-15

---

## 专家蒸馏的演进

DeepSeek V4 采用了多专家 on-policy distillation (OPD) 方法来整合不同领域的专家模型。与前代 V3.2 的离线蒸馏不同，V4 的关键创新在于：**让 student 模型先自己生成回答，再由多个领域专家在 student 生成的轨迹上逐 token 提供反馈**。

这种方法本身并非全新概念。Qwen3 等团队早已采用 OPD 进行强弱模型蒸馏，实现了"约 1/10 的 GPU 算力做到 RL 同等效果"。V4 的不同之处在于将其作为多专家融合的主线。

## 为什么选择 OPD

从 KL 散度角度看：
- **Forward KL** (SFT/传统蒸馏)：学习者匹配教师分布，易导致模式覆盖时丧失原有能力
- **Reverse KL** (OPD/RL)：学习者采样轨迹，教师仅在该轨迹上反馈，避免灾难性遗忘

OPD 特别适合多专家整合，因为各领域专家可以独立训练至极致，而 on-policy 过程最小化知识冲突与分布稀释。

## 对人类学习的启示

作者用学霸 vs 普通学生的对比说明了三种学习模式：

- **纯 RL 模式**：直接硬刚难题，需要大量尝试但上限高
- **纯 Forward KL**：被动听讲和抄答案，易陷入"见过就会，稍改就懵"
- **最优模式（学霸）**：Pre-train + SFT 打基础，后续通过"先自己尝试再找专家反馈"的 on-policy 方式深化学习

关键区别在于，学霸在请教前已经充分尝试，获得的是针对自身思路的反馈，而非从头抄答案。

## OPD 的实践前提

OPD 并非绝对最优。其成功依赖于：

1. **高分布重叠**：Student 和 Teacher 的 top-k token 分布需要足够相近
2. **Off-policy 冷启动**：先用教师数据做 SFT 预热，提高初始重叠度
3. **适当的提示词**：混合 in-distribution 和 OOD 提示，保持探索能力

## 对当代学习方法的反思

文章指出，LLM 时代"递归式学习法"等方法不是方法本身优越，而是因人而异。正常人无法每天提出100个高质量问题。

更现实的做法是：
- 通过读书、技术报告积累 **Pre-train 阶段**的基础知识
- 建立与前沿领域的信息联系，才能向 LLM 这个"领域专家"提出有质量的问题
- 采用 **On-policy 蒸馏模式**：先思考，再向 LLM 确认反馈，而非直接获取答案

## 核心结论

LLM 没有改变最优学习的本质——Pre-train 奠基础，Post-train 通过高质量反馈深化。LLM 的价值在于将稀缺的"专家反馈"变为随处可得，但前提是学习者已通过基础学习能理解这些反馈。

## 参考文献

- DeepSeek V4: <https://arxiv.org/pdf/2505.09388>
- Rethinking OPD: <https://arxiv.org/html/2604.13016v1>
- Thinking Machines Lab — On-Policy Distillation: <https://thinkingmachines.ai/blog/on-policy-distillation/>
- Survey of OPD: <https://arxiv.org/pdf/2604.00626>
