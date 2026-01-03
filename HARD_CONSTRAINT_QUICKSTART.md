# 🔒 Multi-BFS + Regroup 快速开始指南

## 📦 已创建的文件

### 核心算法文件
- ✅ `src/sd311_fieldprep/bundle_hard_constraint.py` - Multi-BFS + Regroup 算法（9 步）

### 测试和对比工具
- ✅ `tests/quick_comparison_hard_constraint.py` - Greedy vs Multi-BFS 对比测试
- ✅ `tests/analyze_bundle_balance_hard_constraint.py` - 分析工具
- ✅ `tests/quick_analyze_hard_constraint.py` - 快速分析脚本
- ✅ `tests/quick_filter_bundles.py` - 过滤和分析脚本

### 文档
- ✅ `src/sd311_fieldprep/README_HARD_CONSTRAINT.md` - 完整文档
- ✅ `HARD_CONSTRAINT_QUICKSTART.md` - 本文件
- ✅ `CONSTRAINT_COMPARISON.md` - 约束模式对比

---

## 🚀 4 步快速开始

### 第 1 步：运行对比测试

在 VS Code 中打开并运行：
```
tests/quick_comparison_hard_constraint.py
```

或在命令行：
```bash
cd /Users/iris/Dropbox/sandiego code/code/fieldprep
python tests/quick_comparison_hard_constraint.py
```

**输出文件**:
- `outputs/bundles/DH/bundles_multibfs_regroup.parquet`
- `outputs/bundles/DH/bundles_multibfs_regroup_map.html`
- `outputs/bundles/DH/comparison_greedy_vs_multibfs_regroup.png`

### 第 1.5 步：（可选）过滤 bundles 到目标范围

如果你想让 DH 和 D2DS 共用相同的 bundle file（已过滤到 [48, 72] 范围）：

```bash
python tests/quick_filter_bundles.py
```

**输出文件**:
- `outputs/bundles/DH/bundles_multibfs_regroup_filtered.parquet`
- `outputs/bundles/DH/bundles_multibfs_regroup_filtered_map.html`
- `outputs/bundles/DH/bundles_multibfs_regroup_dropped_map.html`

然后在 plan 命令中使用 `--bundle-file` 参数：
```bash
python cli.py plan --date 2025-10-18 \
  --bundle-file outputs/bundles/DH/bundles_multibfs_regroup_filtered.parquet \
  --sfh_min 48 --sfh_max 72
```

### 第 2 步：分析结果

在 VS Code 中打开并运行：
```
tests/quick_analyze_hard_constraint.py
```

或在命令行：
```bash
python tests/quick_analyze_hard_constraint.py
```

**输出**:
- 控制台显示完整分析报告
- 生成 `outputs/bundles/DH/bundle_analysis_multibfs_regroup.png`

### 第 3 步：查看结果

1. **查看地图**：在浏览器中打开 `bundles_multibfs_regroup_map.html`
2. **查看图表**：打开 `bundle_analysis_multibfs_regroup.png`
3. **查看数据**：在 QGIS 或 Python 中打开 `bundles_multibfs_regroup.parquet`

---

## 📊 预期结果对比

### Greedy 版本 (5 步, 无约束)
```
Mean: 99.1
Median: 95.0
Max: 700+ addresses  ← 😱 极端值！
CV: 0.42
```

### Multi-BFS + Regroup 版本 (9 步, 推荐配置)
```
Target: 60 addresses
Hard max: 72 addresses (1.2x)
Range: [48, 72] addresses   ← ✅ 严格控制！
CV: ~0.39
有效 bundles: 最大化（经过重组）
```

**关键改进**：
- ✅ 最大值从 700+ 降到 72 (降低 90%)
- ✅ 没有极端 outliers
- ✅ 保持路线连续性（endpoint contiguity）
- ✅ 循环重组最大化数据利用率
- ✅ Split 智能避免产生碎片

---

## 🎛️ 调整参数

### 在 quick_comparison_hard_constraint.py 中修改：

```python
# 配置区域
SESSION = "DH"
TARGET_ADDRS = 60                  # ← 目标地址数
HARD_MAX_MULTIPLIER = 1.2          # ← 硬约束倍数（推荐值）
MIN_BUNDLE_SFH = 40                # ← 最小 bundle 阈值

# 约束说明：
# - Hard max (Step 6, 8): 72 addresses (1.2x)
# - Split threshold (Step 5): 60 addresses (1.0x)
# - Regroup range (Step 9): [48, 72] addresses ([0.8x, 1.2x])
```

**说明**：
- `TARGET_ADDRS = 60`：目标地址数
- `HARD_MAX_MULTIPLIER = 1.2`：Merge 阶段的硬约束（1.2x = 72）
- Split threshold (固定 1.0x)：超过 60 addresses 的 bundle 会被拆分
- Regroup range：[48, 72] addresses

### 在 quick_analyze_hard_constraint.py 中修改：

```python
# 配置区域
TARGET_ADDRS = 60          # ← 目标地址数
HARD_MAX_MULTIPLIER = 1.5   # ← 检查阈值（用于分析 split 是否生效）
```

---

## 🔧 高级使用

### 在自己的脚本中使用

```python
from sd311_fieldprep.bundle_hard_constraint import _build_connected_bundles

# Multi-BFS + Regroup 版本
bundled = _build_connected_bundles(
    segs_m,
    seg_id_col="segment_id",
    target_addrs=60,
    method="multi_bfs",
    hard_max_multiplier=1.2,  # ← 硬约束参数
    min_bundle_sfh=40
)
```

### 命令行分析

```bash
# 分析并生成图表
python tests/analyze_bundle_balance_hard_constraint.py \
    outputs/bundles/DH/bundles_multibfs_regroup.parquet \
    --plot \
    --target 60 \
    --multiplier 1.2
```

---

## 📈 解读分析报告

### 关键指标

| 指标 | 含义 | 目标值 |
|---|---|---|
| **Constraint Violations** | 超过硬约束的 bundle 数 | 0 (完美) |
| **Max Size** | 最大 bundle 大小 | ≤ hard_max (72) |
| **CV** | 变异系数（平衡性） | < 0.15 (优秀) |
| **MASD** | 平均偏离度 | < 0.15 |
| **Range** | 大小范围 | [48, 72] |

### 约束状态判断

```
✅ Constraint Violations: NONE
   → Multi-BFS + Regroup 工作完美！

✅ Range: [48, 72] addresses
   → 所有 bundles 都在合理范围内

⚠️  15 bundles < 48
   → 循环重组未能完全消除（可接受）
```

---

## 🐛 故障排除

### 问题 1: 找不到文件

```
❌ Error: File not found: bundles_multibfs_regroup.parquet
```

**解决**:
```bash
# 先运行对比测试生成文件
python tests/quick_comparison_hard_constraint.py
```

### 问题 2: 仍有超过限制的 bundle

```
⚠️  Constraint Violations: 25 bundles (1.2%)
```

**可能原因**:
1. `_sweep_attach_residuals` 使用软约束 (1.1x)
2. Regroup 迭代次数不够

**解决**:
```python
# 在 bundle_hard_constraint.py line 1120
max_iterations=5  # 增大迭代次数到 10
```

### 问题 3: 太多 tiny bundles

```
💡 150 tiny bundles (< 48) remain.
```

**原因**:
- 重组后仍无法组成符合范围的 bundle
- 总地址数不足

**解决**:
- 降低 min_size (48 → 40)
- 或接受少量 tiny bundles

---

## 📝 Multi-BFS + Regroup 算法执行流程

### 9 步执行顺序

```
Step 1: Build graph + components
  • 构建邻居图，识别连通分量
            ↓
Step 2: Grow bundles (multi-BFS balanced)
  • 多源 BFS 同时增长
  • 创建 ~1800 个接近 target 的 bundles
            ↓
Step 3: Merge tiny (connected, no constraint, prefer smallest)
  • 连通性合并，无约束
  • 优先合并到最小的 bundle
            ↓
Step 4: Sweep residuals (soft 1.1x)
  • 附加剩余 segments
  • 软约束 66 addresses，允许 fallback
            ↓
Step 5: Split oversized (> 60)
  • 拆分超过 60 addresses 的 bundles
  • ✅ 带剩余检查，避免产生 < 40 的碎片
            ↓
Step 6: Merge tiny (connected, hard 1.2x)
  • 连通性合并 + 硬约束 72 addresses
            ↓
Step 7: Enforce contiguity
  • 强制路线连续性
  • 拆分不连续的 bundles
            ↓
Step 8: Final cleanup (connected, hard 1.2x)
  • 最终清理，连通性合并 + 硬约束
            ↓
Step 9: Regroup invalid bundles ([48, 72])  ← 🆕 新增！
  • 循环重组不合格的 bundles
  • 最多迭代 5 次
  • 最大化可用 bundles 数量
```

**关键设计**：
- Step 3 使用连通性合并（保持路线连续性）
- Step 5 智能 Split（避免产生碎片）
- Step 9 循环重组（最大化数据利用率）

---

## 📂 测试工作流程

```
1. quick_comparison_hard_constraint.py
   ↓
   生成 bundles_multibfs_regroup.parquet
   生成 bundles_multibfs_regroup_map.html
   生成 comparison_greedy_vs_multibfs_regroup.png

2. quick_analyze_hard_constraint.py
   ↓
   分析 bundles_multibfs_regroup.parquet
   生成 bundle_analysis_multibfs_regroup.png
   显示分析报告

3. (可选) quick_filter_bundles.py
   ↓
   过滤到 [48, 72] 范围
   生成 filtered/dropped 地图

4. 查看结果
   ↓
   - 浏览器打开 map.html
   - 查看 .png 图表
   - 读取控制台报告

5. (可选) 调整参数并重新运行
```

---

## 💡 推荐设置

### 严格控制（推荐用于生产）
```python
TARGET_ADDRS = 60
HARD_MAX_MULTIPLIER = 1.2  # hard max: 72 addresses
MIN_BUNDLE_SFH = 40        # 最小阈值
# split threshold 固定 1.0x (60)
# regroup range: [48, 72]
```

**特点**：
- ✅ 严格限制在 [48, 72] 范围内
- ✅ Split 避免产生碎片
- ✅ 循环重组最大化数据利用
- ✅ 最终结果：max ≤ 72

### 约束层级说明

```
Hard max (Step 6, 8):  1.2x (72)  - Merge 硬约束
Soft max (Step 4):     1.1x (66)  - Sweep 软约束
Split threshold (Step 5): 1.0x (60)  - 拆分阈值
Regroup range (Step 9):  [0.8x, 1.2x] ([48, 72])
```

---

## 📞 获取帮助

查看完整文档：
```
src/sd311_fieldprep/README_HARD_CONSTRAINT.md
CONSTRAINT_COMPARISON.md
```

---

**创建日期**: 2025-12-18
**最后更新**: 2025-12-23
**版本**: 2.0 - Multi-BFS + Regroup (9 步)
**适用于**: San Diego 311 Field Prep Project
