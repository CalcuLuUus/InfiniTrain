# InfiniTrain 路线 A（两周）学习率调度器实现计划（plan.md）

> 目标：用 **2 周**实现一个工程化、可扩展、可恢复（state 可保存/加载）的 **LRScheduler** 模块，并按 `hwdoc/【训练方向 2025 冬季训练营】学习率调度器实现.pdf` 的验收方式完成对齐验证与材料准备。  
> 关键前提：InfiniTrain 当前 Optimizer 的 `learning_rate_` 是 `const float` 且无 `SetLearningRate()` 通路，**必须先打通 lr 可更新链路**，否则 scheduler 无法接入（注意：只做“最小接口改动”，不改变 Optimizer 更新算法/分布式执行逻辑）。

---

## 总交付标准（第 14 天，对齐 PDF 验收）
1. 基础策略（必做）：**ConstantLR / StepLR / Linear Warmup**（建议实现为对齐 PyTorch 的 `LinearLR`，Warmup 是其特例）  
2. 组合策略（必做）：**SequentialLR / ChainedScheduler / LambdaLR**  
3. `state_dict()/load_state_dict()`（或 `State()/LoadState()`）：支持保存/恢复 step（或 epoch）与必要超参；组合 scheduler 也可恢复  
4. 接入（必做）：在 `example/gpt2/main.cc` 与 `example/llama3/main.cc` 通过 gflags 配置启用，并在 log 中打印每步 lr（打印 **scheduler/optimizer 的当前 lr**，不再只打印 `FLAGS_learning_rate`）  
5. 对齐验证与提交材料（必做）：按 PDF 要求完成报告与测例：GPT-2 与 LLaMA-3 在 **8 卡 DDP（DP=8）** 下分别跑 **StepLR** 与 **ChainedScheduler(StepLR, LinearLR)**，`num_iteration >= 10`；提供两套框架（InfiniTrain & PyTorch）的运行命令与 log 截图（共 8 张）；PR **不包含** PyTorch 对齐代码，PyTorch 代码另行提交/发送

---

## 必做最小接口改动：Optimizer lr 可更新通路（第 2 天前完成）
### 改动点（明确到文件）
- `infini_train/include/optimizer.h`
  - 在 `class Optimizer` 增加虚函数：
    - `virtual float GetLearningRate() const = 0;`
    - `virtual void SetLearningRate(float lr) = 0;`
- `infini_train/src/optimizer.cc`
  - `SGD` / `Adam` 的 `learning_rate_` 从 `const float` 改为 `float`
  - 实现 `GetLearningRate/SetLearningRate`
- `infini_train/include/nn/parallel/ddp/distributed_optimizer.h`
  - `DistributedOptimizer` override Get/Set，并将 Set 透传给 `base_optimizer_`
- `infini_train/src/nn/parallel/ddp/distributed_optimizer.cc`
  - 实现 override（forward/透传即可）

### 验收（必须）
- 在训练 loop 的第 N 步将 lr 改为 1/10：
  - log 中 lr 发生变化
  - 训练能继续正常跑（不 crash）

---

# Week 1：链路打通 + 3 个基础策略 + e2e 接入

## Day 1：定位训练 loop 与 lr 注入点（必须产出）
**目标产出**
- 找到 gpt2（或 llama3）训练循环中 `optimizer->Step()` 的位置
- 在 log 中打印：`step` 与当前 lr（先打印 `FLAGS_learning_rate` 也可，但最终必须打印 scheduler/optimizer 的 lr）
- 明确 scheduler.step 调用时机：**按作业 PDF，推荐顺序为 `scheduler.Step(); optimizer.Step();`**。在 InfiniTrain 里建议把 `scheduler->Step()` 放在每次参数更新前（例如每个 iteration 开头，或紧挨 `optimizer->Step()` 之前），并写清楚 step 语义与 off-by-one 约定（尤其是 warmup/step 衰减点）

**建议搜索关键字**
- `optimizer->Step`
- `ZeroGrad`
- `Backward`
- `learning_rate` / `lr` / `SetLearningRate`

---

## Day 2：实现 Optimizer 的 lr 更新接口（必须产出）
**目标产出**
- 完成 `GetLearningRate/SetLearningRate` 的全链路（含 DistributedOptimizer 透传）
- 在训练中动态修改 lr 生效（最小回归）

---

## Day 3：Scheduler 骨架 + ConstantLR（闭环第一步）
**建议新增文件**
- `infini_train/include/lr_scheduler.h`
- `infini_train/src/lr_scheduler.cc`

**接口建议（够用就行）**
- 基类持有：`std::shared_ptr<Optimizer> optimizer_`、`float base_lr_`、`int64_t last_step_`
- `void Step()`：`++last_step_`；`lr = ComputeLR(last_step_)`；`optimizer_->SetLearningRate(lr)`
- `float ComputeLR(int64_t step)`：子类实现
- `state_dict/load_state_dict`：先保存 `last_step_` 与必要超参（可用 json 或内部 map）

**验收**
- 加 `--lr_scheduler=constant`
- 跑 50 step，lr 恒定且可见

---

## Day 4：实现 Linear Warmup / LinearLR（钉死 step 语义）
**实现要点**
- 参数：`warmup_steps`（或对齐 PyTorch 的 `start_factor/end_factor/total_iters`）
- 线性从 0（或 start_factor）到 `base_lr`（或线性到 end_factor × base_lr）

**验收**
- 打印 `0..warmup_steps+10` 的 lr 序列
- 明确并写进 README：第 0 步的 lr 定义、warmup 结束点定义（避免 off-by-one）

---

## Day 5：实现 StepLR（暴露边界条件）
**实现要点**
- 参数：`step_size`、`gamma`
- 每 `step_size` 步衰减一次

**验收**
- 打印衰减点前后 lr：`step_size-1 / step_size / step_size+1`

---

## Day 6：接入 gflags 配置 + 稳定日志输出
**新增 flags（建议）**
- `--lr_scheduler=constant|step|linear|sequential|chained|lambda`
- `--warmup_steps`（或 `--linear_start_factor/--linear_end_factor/--linear_total_iters`）
- `--lr_step_size`、`--lr_gamma`
- `--lr_milestones`（SequentialLR）
- `--lr_chain=...`（ChainedScheduler：用逗号指定子策略，例如 `step,linear`）
- `--lr_lambda=...`（LambdaLR：至少支持 1–2 种可配置的 lambda 形式，确保可对齐 PyTorch 测例）
- （可选）`--print_lr_every_n=1`

**验收**
- 不改代码，仅改参数即可切换策略
- 每 step 打印 `step, lr`

---

## Day 7：补齐 state 接口 + 做一次“伪 resume 一致性”回归（为 Week2 铺路）
**方法**
- 输出 200 step 的 lr 序列到文件 A
- 从 step=80 的 state 继续输出到 200 的序列 B
- 检查 A[81..200] 与 B[81..200] 完全一致

---

# Week 2：组合策略 + state_dict + 最终验收材料

## Day 8：实现 SequentialLR（必做）
**实现要点**
- 输入：子 scheduler 列表 + `milestones`
- 根据 step 选择子 scheduler 的 ComputeLR
- 子 scheduler 的 step 语义要与父 scheduler 一致（建议统一以 global step 驱动）

**验收**
- 组合：`Warmup(100) + StepLR(step_size=50,gamma=0.1)`
- 输出前 300 step lr 序列并保存

---

## Day 9：实现 ChainedScheduler（必做，且覆盖报告测例）
**实现要点**
- 输入：子 scheduler 列表
- 每一步按顺序执行所有子 scheduler 的 step/lr 变换（对齐 PyTorch `ChainedScheduler` 行为）

**验收（对齐报告测例）**
- 组合：`ChainedScheduler(StepLR, LinearLR)`（参数选择与 PyTorch 脚本一致）
- 输出前 300 step lr 序列并保存

---

## Day 10：完成 state_dict/load_state_dict + LambdaLR（必做）
**必须包含**
- `last_step_`
- SequentialLR：当前子 scheduler 索引 + 每个子 scheduler 的 state

**强回归**
- 连续跑 300 step：序列 A
- 跑到 step=137 保存 state，恢复后跑到 300：序列 B
- 检查 A 与 B 完全一致（diff=0）

**LambdaLR 说明**
- LambdaLR 本身的 “lambda 函数” 不适合直接序列化；对齐 PyTorch 的做法：state 只保存 step 等状态，恢复时由外部用相同 lambda 重新构造

---

## Day 11：把 scheduler 接入 GPT-2 & LLaMA-3，并跑通 DP=8（必须）
**验收**
- gpt2 / llama3 均至少跑 `num_iteration >= 10`，并打印 lr
- 用 `--nthread_per_process=8` 跑 DDP（DP=8），确保不会因为多线程/分布式优化器导致 lr 不一致或崩溃

---

## Day 12：按 PDF 跑齐 4 个对齐测例并收集截图（必须）
**4 个测例（报告硬性要求）**
- GPT-2（DP=8）：StepLR
- GPT-2（DP=8）：ChainedScheduler(StepLR, LinearLR)
- LLaMA-3（DP=8）：StepLR
- LLaMA-3（DP=8）：ChainedScheduler(StepLR, LinearLR)

**产出**
- 每个测例分别用 InfiniTrain 与 PyTorch 跑，保存运行命令与 log 输出截图（共 8 张）
- 额外保存：对应的 lr 序列（便于定位 loss 不一致是否来自 scheduler）

---

## Day 13：报告 + 代码质量回收
- 参数合法性检查（warmup_steps>0、milestones 单调递增）
- 错误提示清晰
- 改动面控制：只动 optimizer + 新增 scheduler 文件 + example 接入
- 报告必含：设计思路/架构、使用说明（flags）、4 个测例对齐结果（命令 + 截图）

---

## Day 14：最终交付包（按 PDF）
- 代码 PR：命名 `【训练营】学习率调度器实现`（PR 不包含 PyTorch 对齐代码）
- 项目报告：设计 + 结果展示 + 使用说明（flags）
- PyTorch 对齐代码：与报告一起按要求单独提交/发送
- 一页总结：实现范围、step 语义、已知限制（例如 LambdaLR 的可配置范围与序列化策略）

---

## 提高规划精度所需的最小资料（可选，但强烈建议提供）
1. `example/gpt2/main.cc` 中训练 loop 片段（包含 `ZeroGrad/Backward/optimizer->Step` 的 80–120 行）  
2. 你能跑起来 gpt2 的最小命令（CPU/CUDA、单卡/多卡）

提供后可把 `scheduler.Step(); optimizer.Step();` 的具体插入位置与 “warmup/step 衰减点定义”精确到补丁级指引。
