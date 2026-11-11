# Exp2：训练期间 rollout 参与进化

整个 exp2 目录放到 CALSESD 仓库根目录。只新增文件；不修改 models/env/training。
这是新的训练入口，不是原训练脚本的转发器。

## 训练
从 CALSESD 的父目录执行：

```bash
python CALSESD/exp2/train_exp2.py --exp2-backend rollout --device cuda --updates 300
```

每次真实决策：原 trigger 决定 due 行 -> 当前状态复制为独立影子环境 -> rollout 回报作为种群 fitness ->
精英选择/交叉/固定变异/移民 -> 按行接受 -> 真实 Actor 根据新 z 采样 -> 真实轨迹进入 PPO。
模拟轨迹不进入 PPO。Critic 仍训练真实轨迹价值，用于 GAE，而不参与 rollout 组的候选排序。
父类仍会编码 Critic 状态特征，rollout 评分不使用它；这是复用接口的少量冗余，不是 Critic 打分。

默认训练 warmup 仍遵循 search_warmup_updates；warmup 期间无在线搜索。
主实验采用纯后端对比：关闭梯度精修、自适应信任/变异、LCB/分歧阈值、确认和时间预算。
保留 cfg.reopt_min_gain 的绝对阈值；所有有效配置写入 config.json/protocol.json。
命令行 ablation_family/variant 被规范化为 none/default，仅用于原配置构造；实际运行标签和事件为 exp2/backend。
请勿用 exp1/exp3 名称解释本入口。

## 训练期后端对照

```bash
python CALSESD/exp2/train_exp2.py --exp2-backend critic --device cuda --updates 300 --seed 42
python CALSESD/exp2/train_exp2.py --exp2-backend rollout --device cuda --updates 300 --seed 42
```

必须分别训练两个 checkpoint。相同 seed 不代表耗时相同；共同随机算子也不能保证分化后种群相同。

## 小配置烟雾测试

```bash
python CALSESD/exp2/train_exp2.py --exp2-backend rollout --device cpu --updates 2 \
  --batch-size 2 --grid-size 4 --num-customers 3 --max-episode-steps 15 \
  --hidden-dim 16 --dynamic-dim 8 --num-heads 2 --latent-dim 4 \
  --static-gat-layers 1 --static-transformer-layers 1 \
  --reopt-candidates 4 --evolution-generations 1 --search-warmup-updates 0 \
  --online-search-interval 2 --online-cooldown-steps 1 \
  --online-min-remaining-customers 1 --validation-interval 1 --validation-seeds 1
python -m unittest CALSESD.exp2.test_exp2 -v
```

## 验证与独立评估
训练期间周期 system 验证也使用对应训练后端；policy 配对不搜索。
直接复用 validation.py 的实例种子序列、z 派生种子和搜索派生种子，强制 common 初始 z。
checkpoint 原配置若是 provided，本入口使用 common，不能声称 provided 协议相同。

```bash
python CALSESD/exp2/eval_exp2.py --checkpoint CALSESD/checkpoints/exp2_rollout/RUN/checkpoint_latest.pt --device cuda
```

默认相对路径基于仓库根目录。checkpoint 位于 checkpoints/exp2_BACKEND/时间戳。
resume 的路径按 shell 工作目录解释。续训必须保持配置，只允许更新总 updates 等输出参数：

```bash
python CALSESD/exp2/train_exp2.py [与原运行相同参数] --updates 600 --resume CHECKPOINT
```

init-checkpoint 只迁移严格匹配的网络权重，不恢复训练状态；不能将其称为从零训练。
checkpoint 使用专用 scheme，保存 backend/chunk/repeats、优化器、搜索调用号、触发器、persistent z 和 RNG。

## 模拟假设及限制
- 仅适配当前 Grid BatchCVRPEnv，未宣称支持任意环境。
- 隐藏需求重新采样；已揭示需求保留；独立交通 RNG，不读取实盘未来。
- 需求场景键包含原始 batch 行号，交通为共享场景；场景与候选分块无关。
- rollout 固定 z、Actor 贪心延续到完整有限回合；不递归触发搜索，也无终止 bootstrap。
- 真实训练动作是采样，真实后续决策还会搜索；模拟是候选评价代理，不是对实际在线控制器价值的无偏估计。
- PPO 是固定已采样 z/context 的条件 Actor 更新，不对进化选择或 Actor 在模拟中的影响求全梯度。
- 搜索计算量可能非常大。先降低 candidates/generations/repeats/customer 数验证，不要直接大规模运行。
- 验证适配器使用可恢复的模块依赖替换，只支持串行验证，不支持同进程并发验证。

## 验证状态
本次交付仅完成 AST/编译检查和纯 Python seed 测试；当前制作环境没有 torch/PyG，未运行单元测试或训练。
提供的 test_exp2 会测试影子隔离、隐藏需求无依赖、分块和行子集公共随机数一致性。
