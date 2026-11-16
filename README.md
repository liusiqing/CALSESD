# CALSESD

**C**ritic-**A**ware **L**atent-**S**pace **E**volutionary **S**earch for **D**ynamic CVRP

面向动态 CVRP（动态交通 + 需求逐步揭示）的强化学习框架：用 PPO 训练一个受隐变量 z 条件化的指针网络策略，并在每个 episode 内按触发器周期性地对 z 做 Critic 引导的进化搜索。

## 方法概览
- 环境：车辆在网格（或欧氏坐标）上服务客户。距离矩阵随时间按 AR(1) 过程变化，客户真实需求首次到访才揭示，容量不足时无法服务。
- 策略（Actor）：静态几何编码（GATv2 + Transformer）与动态特征编码融合，指针网络输出 logits，z 参与查询向量构造。
- 价值（Critic）：多头集成 V(s, z)，输入含图池化特征、物理上下文和在线控制器上下文，可选随机先验网络。
- 隐空间搜索：触发器选出到期的行，HybridLatentSearch 以保守得分（均值减 λ 倍集成标准差）为适应度做精英保留、交叉、自适应变异、移民和一步梯度精修；保守增益超过相对不确定性阈值才切换 z。
- 训练：Actor 使用 PPO；GAE 在 z 切换步用 V(s', z_old) 做 bootstrap。
- 含 mermaid 流程图：reset → 触发器 → （到期行）z 进化搜索 → 刷新状态 → Actor 采样 → env.step → 回到触发器。

## 仓库结构
env/、models/、training/ 各文件的一行说明（cvrp_env、cvrp_euclidean_env、batch_env、online_state、traffic；policy、value、hybrid_latent_search；train_ppo），以及 pyproject.toml。

## 环境细节
实例规模（12×12 网格、100 客户、容量 15）、需求模型、交通模型（relative/additive）、观测维度（6 + 3 + 2，Critic 另读 6 维在线上下文）、奖励（−cost/norm_dist，失败惩罚，reward_scale）、并行方式。

## 快速开始
依赖：PyTorch、PyTorch Geometric、NumPy。克隆目录须为 CALSESD。命令：--help-all、CPU 冒烟测试（--mode keep，2 个 update）、完整训练（--device cuda --updates 2000）、--resume 严格恢复、--init-checkpoint 热启动。

## 常用参数
表格：updates、batch-size、grid-size/num-customers/vehicle-capacity、max-episode-steps、mode、search-warmup-updates、online-search-interval、online-max-attempts-per-episode、reopt-candidates/evolution-generations、z-resample-prob、lcb-lambda/reopt-min-gain-ratio、failure-penalty/reward-scale、validation-interval/validation-seeds（均为当前默认值）。说明搜索预算默认为 0，结果与机器速度无关。

## 消融实验
none/default；exp1（none、value_gradient、random_matched）；exp3（fixed、trust_only、mutation_only、both）；基线用 --mode keep；同 seed 下训练实例序列相同，可配对比较。

## 验证与输出
贪心验证、训练/验证种子分离、system 协议加配对的 policy 协议（paired_validation 事件的 search_gain）、最优检查点选择规则；outputs/ 与 checkpoints/ 下各文件说明表。

## 开发
ruff check 与 black 命令（配置在 pyproject.toml）。
