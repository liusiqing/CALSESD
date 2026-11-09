# Exp2 v2：先训练共享模型，再比较 Critic / rollout 评分

只新增 exp2 目录，不修改核心代码。目录放到 CALSESD/exp2。
本实现不是 rollout 搜索参与 PPO 训练，而是对同一训练 checkpoint 比较评分后端。

从 CALSESD 的父目录执行：

```bash
python CALSESD/exp2/train_exp2.py --device cuda --updates 300 --seed 42
python CALSESD/exp2/eval_exp2.py --checkpoint CALSESD/checkpoints/RUN/checkpoint_latest.pt --device cuda
```

原训练入口相对输出路径基于 CALSESD 根目录，不基于 shell 工作目录。
评估默认使用 checkpoint 的 validation_seed 和 validation_seeds，并直接调用 validation.py。
初始 z 固定为 common，与 validation 的 common 协议一致；若 checkpoint 使用 provided，初始 z 协议不相同。
--validation-seeds 1 可用于烟雾测试（只取原验证种子的前缀），--chunk 2 降低显存。

主对比：固定共享种群、交叉、变异参数；关闭梯度精修、自适应控制、LCB、相对分歧阈值、确认与时间预算。
使用绝对 reopt_min_gain，保留相同 candidates/generations；不声称计算量或耗时公平。
rollout 使用固定 z 贪心延续到有限任务终止，不递归执行未来在线搜索；这与 Critic 的在线控制器价值语义不同，必须在论文中说明。

所有候选/原解共享场景；需求流绑定原始 batch 行号，交通流绑定调用/重复号。
重复数不确定性不作为 Critic ensemble 分歧；不使用实盘 RNG 推演真实未来。
无自动胜出确认，避免额外接受规则混入主对比。
评估输出 events 中标明 rollout 后端，记录真实前滚计数；成本配对标准误按 seed 聚合。

验证状态：仅语法和随机流派生测试，未在 PyTorch/PyG 上运行。请先冒烟测试。
