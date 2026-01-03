# Simulation Module

多日实验模拟模块 - 用于估算样本量和生成每日工作计划

## 概述

这个模块用于模拟30天的field实验，计算能收集多少样本。

**实验设计**:

**Day 1**:
- 6个interviewer (A-F)
- 每人5个DH bundle
- 总共30个DH bundle
- 无D2DS任务

**Day 2-30** (每天):
- **DH**: 每人1个新bundle（共6个DH bundle/天）
- **D2DS**: 6个bundle
  - 4个从已完成DH的bundle中抽取（复用）
  - 2个从剩余未用bundle中抽取（新bundle）

**抽样方式**: 无放回抽样

## 使用方法

### 快速运行

```bash
python tests/quick_simulate_30days.py
```

### Python代码

```python
from sd311_fieldprep.simulation.multiday import simulate_multiday_experiment

plan, stats = simulate_multiday_experiment(
    n_days=30,
    n_interviewers=6,
    day1_bundles_per_interviewer=5,
    daily_dh_per_interviewer=1,
    daily_d2ds_from_completed=4,
    daily_d2ds_new=2,
    start_date="2025-01-06",
    seed=42
)
```

## 输出文件

1. **`outputs/simulation/plan_30days.csv`** - 30天的完整工作计划
   - 包含每天每个interviewer的bundle分配
   - 格式: date, interviewer, task, bundle_id, list_code

2. **`outputs/simulation/stats_30days.csv`** - 每日统计数据
   - 每日bundle数、地址数
   - 累计统计

3. **`outputs/simulation/overlap_30days.csv`** - DH与D2DS重叠分析
   - Bundles、segments、addresses的overlap统计
   - 显示有多少接受DH only、D2DS only、或both

## 样本量估算 (30天)

### Bundle使用

- **总独立bundle**: 262个
  - Day 1: 30个DH
  - Day 2-30: 174个新DH + 58个新D2DS
- **DH任务总数**: 204次
- **D2DS任务总数**: 174次（其中116次复用DH bundle）

### 地址数（分配前）

- **总DH地址**: 12,576
- **总D2DS地址**: 10,743
- **总计**: 23,319

### DH样本（考虑treatment分配）

DH地址会被分配到3组：
- **50% control**（不访问）: ~6,288 addresses
- **25% full**（100%访问）: ~3,144 addresses
- **25% partial**（50%访问）: ~3,144 addresses → 实际访问 ~1,572

**DH访问样本** ≈ 4,716 addresses (3,144 full + 1,572 partial)

**DH control样本** ≈ 6,288 addresses (未访问，用于对照)

**DH总实验样本** = 12,576 addresses (全部用于分析)

### D2DS样本

所有D2DS地址都会被访问。

**D2DS有效样本** = 10,743 addresses

### 总样本汇总

**实际访问地址**:
- DH访问: 4,716 (3,144 full + 1,572 partial)
- D2DS访问: 10,743
- **总访问**: 15,459 addresses

**实验总样本**（包括control）:
- DH总样本: 12,576 (6,288 control + 3,144 full + 3,144 partial)
- D2DS总样本: 10,743
- **实验总样本**: 23,319 addresses

### DH与D2DS重叠

**Bundles**:
- DH only: 124 bundles (47.3%)
- D2DS only: 58 bundles (22.1%)
- Both: 80 bundles (30.5%) - 复用的bundle

**Segments**:
- DH only: 729 segments (47.7%)
- D2DS only: 355 segments (23.2%)
- Both: 443 segments (29.0%)

**Addresses**:
- DH only: 7,633 addresses (47.3%)
- D2DS only: 3,572 addresses (22.1%)
- Both: 4,943 addresses (30.6%) - 接受两种treatment的地址

💡 **关键发现**: 30.6%的地址会接受DH和D2DS两种treatment，这是设计的一部分（D2DS复用已完成DH的bundle）

## 每日平均

- DH地址/天: ~419
- D2DS地址/天: ~358
- 总地址/天: ~777

## Bundle需求

30天实验需要262个独立bundle：
- 可用bundle: 2,221个
- 使用率: 11.8%
- 剩余bundle: 1,959个

## 参数说明

```python
simulate_multiday_experiment(
    n_days=30,                          # 总天数
    n_interviewers=6,                   # 访员数量
    day1_bundles_per_interviewer=5,     # Day 1每人bundle数
    daily_dh_per_interviewer=1,         # Day 2+每人DH bundle数
    daily_d2ds_from_completed=4,        # D2DS中来自已完成DH的数量
    daily_d2ds_new=2,                   # D2DS中新bundle的数量
    bundle_file="...",                  # Bundle文件路径
    addr_assignment_file="...",         # 地址分配文件路径
    output_dir=None,                    # 输出目录
    start_date="2025-01-06",            # 起始日期
    list_code=30,                       # List code
    seed=42                             # 随机种子
)
```

## 下一步

生成plan后，可以使用`emit.py`为每一天生成daily工作文件：

```python
from sd311_fieldprep.emit import run_emit

# 为Day 1生成工作文件
run_emit(
    date='2025-01-06',
    plan_csv='outputs/simulation/plan_30days.csv',
    addr_assignment_file='outputs/sweep/locked/segment_addresses_b40_m2.parquet'
)
```

## 注意事项

1. **无放回抽样**: 每个bundle在DH中只使用一次，但可以在D2DS中重复使用
2. **DH Treatment分配**: 在emit阶段，DH地址会被自动分配到control/full/partial组
3. **D2DS复用**: Day 2+的D2DS任务会从已完成DH的bundle中抽取4个，确保同一区域先DH后D2DS
4. **随机化**: 使用seed参数确保结果可复现
