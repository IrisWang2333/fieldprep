# 约束模式对比：无约束 vs 软约束 vs 硬约束 + 重组

## 📋 快速回答

**`quick_comparison_hard_constraint.py` 调用的是哪个版本？**

```python
# quick_comparison_hard_constraint.py
from sd311_fieldprep.bundle import _build_connected_bundles as greedy_bundle
from sd311_fieldprep.bundle_hard_constraint import _build_connected_bundles_multibfs
```

**对比两个完全独立的算法：**
1. **Greedy (5 步, 无约束)** - 标准 bundle.py
2. **Multi-BFS (10 步, 硬约束 + 重组 + Eulerian)** - bundle_hard_constraint.py

---

## 📞 调用链

### Greedy 版本（5 步）
```
quick_comparison_hard_constraint.py
  ↓ 导入
sd311_fieldprep/bundle.py (_build_connected_bundles)
  ↓ 使用
混合约束模式 (无约束 + 软约束)
  ↓ 结果
Max = 700+ addresses ❌
```

### Multi-BFS + Regroup + Eulerian 版本（10 步）
```
quick_comparison_hard_constraint.py
  ↓ 导入
sd311_fieldprep/bundle_hard_constraint.py (_build_connected_bundles_multibfs)
  ↓ 使用
硬约束 (1.1x) + 拆分 (1.0x) + 循环重组 + Eulerian 验证
  ↓ 结果
Max ≤ 66 addresses ✅
所有 bundles 在 [54, 66] 范围内 ✅
所有 bundles 可行走（Eulerian 路径）✅
```

---

## 📊 快速对比表

| 算法 | 步骤数 | Target | 约束模式 | 最大值控制 | 实际 Max |
|---|---|---|---|---|---|
| **Greedy** | 5 | 60 | ❌ 无/软 | 无 | ~700+ |
| **Multi-BFS + Regroup** | 9 | 60 | 🔒 硬约束 + 重组 | [48, 72] | ≤ 72 |

**关键区别**：
- Greedy = 标准版本 = 会产生极端值
- Multi-BFS = 硬约束 + 自动重组 = 严格控制 + 最大化利用数据

---

## 🔄 Multi-BFS + Regroup 完整流程（9 步）

```
Step 1: Build graph + components
   ↓
   构建邻居图，识别连通分量

Step 2: Grow bundles (multi-BFS balanced)
   ↓
   多源 BFS 平衡增长，创建 ~1800 bundles (target=60)

Step 3: Merge tiny (connected, ❌ no constraint, prefer smallest)
   ↓
   连通性合并，无约束，优先合并到最小的 bundle

Step 4: Sweep residuals (soft 1.1x)
   ↓
   附加剩余 segments，软约束 66 addresses (1.1x)

Step 5: Split oversized (> 1.0x = 60)
   ↓
   自动拆分 > 60 addresses 的 bundles
   ✅ 含剩余部分检查（避免产生 < 40 的碎片）

Step 6: Merge tiny (connected, 🔒 hard 1.2x)
   ↓
   连通性合并 + 硬约束 72 addresses (1.2x)

Step 7: Enforce contiguity
   ↓
   强制路线连续性，拆分不连续的 bundles

Step 8: Final cleanup (connected, 🔒 hard 1.2x)
   ↓
   最终清理，连通性合并 + 硬约束

Step 9: Regroup invalid bundles ([48, 72]) ← 🆕 新增！
   ↓
   循环重组不合格的 bundles
   - 找出 < 48 或 > 72 的 bundles
   - 打散并重新组合
   - 最多迭代 5 次
   - 最大化可用 bundles 数量

最终结果：
  ✅ Max ≤ 72 addresses
  ✅ 所有 bundles 在 [48, 72] 范围内
  ✅ 路线连续性保证
  ✅ 最大化数据利用率
```

---

## 🆚 三种约束模式对比

### 1️⃣ 无约束（Greedy `_merge_tiny_bundles`）

```python
# 完全不检查大小
nearest_bid = min(cand, key=lambda b: src_cent.distance(bundle_cent[b]))
g.loc[g["bundle_id"] == bid, "bundle_id"] = nearest_bid  # 直接合并
```

**特点**：
- ✅ 几乎所有 tiny bundles 都能合并
- ✅ 很少剩余 singleton bundles
- ❌ 可能产生极端大的 bundles
- ❌ 分布不可预测

**实际结果**：
```
Target: 60
Max: 700+ addresses (极端！)
Singleton bundles: 0
```

---

### 2️⃣ 软约束（`_sweep_attach_residuals`）

```python
# 允许超过限制，但有建议值
if (totals.get(bid, 0) + add) <= soft_max * 1.10:
    return True  # 可以合并
else:
    # 还是会选择最近的作为 fallback
    chosen = ranked[0]  # last resort
```

**特点**：
- ✅ 有建议的大小限制
- ⚠️ 允许 10% 溢出
- ⚠️ 如果都超限，仍然会选最近的（不拒绝）
- ✅ 平衡了灵活性和控制

**实际结果**：
```
Target: 60
Soft max: 66 (1.1x)
实际 max: 可能超过 66（fallback 导致）
```

---

### 3️⃣ 硬约束 + 重组（Multi-BFS）

```python
# Step 3/6/8: 硬约束检查
if hard_max is not None:
    valid_cand = []
    for b in cand:
        if bundle_sizes.get(b, 0) + tiny_size <= hard_max:
            valid_cand.append(b)

    if not valid_cand:
        rejected_count += 1
        continue  # 拒绝合并

# Step 5: Split 带剩余检查
if min_bundle_sfh is not None and chunk_size >= min_bundle_sfh:
    remainder_size = current_size - chunk_size
    if remainder_size >= min_bundle_sfh:
        if chunk_size >= target_addrs * 0.8:
            break  # 提前停止，避免产生碎片

# Step 9: 循环重组
while iteration < max_iterations:
    # 找出不合格 bundles (< 48 或 > 72)
    invalid_bundles = bundle_sizes[(bundle_sizes < min_size) | (bundle_sizes > max_size)]
    # 打散并重新组合
    # 停止条件：无改进或无法组成合格 bundle
```

**特点**：
- ✅ 严格控制最大值 (≤ 72)
- ✅ 循环重组确保最大化可用数据
- ✅ Split 时避免产生碎片
- ✅ 分布可预测
- ✅ 路线连续性保证

**实际结果**：
```
Target: 60
Hard max: 72 (1.2x)
实际 max: ≤ 72 (绝不超过！)
有效 bundles: 最大化（经过重组）
Range: [48, 72] addresses
```

---

## 📊 实际效果对比表

| 特性 | Greedy (无约束) | Multi-BFS + Regroup |
|---|---|---|
| **步骤数** | 5 | 9 |
| **最大值控制** | ❌ 无 | ✅ 严格 (≤ 72) |
| **最大 bundle** | 700+ | ≤ 72 |
| **数据利用率** | 高 | 最大化（重组） |
| **可预测性** | ❌ 低 | ✅ 高 |
| **路线连续性** | ✅ 有 | ✅ 保证 |
| **适用场景** | 追求灵活性 | 现场工作部署 ✅ |

---

## 🔧 核心改进点

### 1. Split 剩余部分检查（Step 5）

**问题**：之前的 split 可能产生太小的碎片
```
Bundle: 150 addresses
Split: [60, 60, 30]  ← 30 < 40 (min_bundle_sfh) ❌
```

**解决**：
```python
# Line 822-828
if min_bundle_sfh is not None and chunk_size >= min_bundle_sfh:
    remainder_size = current_size - chunk_size
    if remainder_size >= min_bundle_sfh:
        if chunk_size >= target_addrs * 0.8:
            break  # 提前停止
```

**效果**：
```
Bundle: 150 addresses
Split: [48, 102]  ← 都 >= 40 ✅
```

### 2. 循环重组（Step 9）

**功能**：自动重组不合格的 bundles
```python
# 循环最多 5 次
while iteration < 5:
    # 1. 找出 < 48 或 > 72 的 bundles
    # 2. 提取所有 segments
    # 3. 用贪婪 BFS 重新组合 (目标 60)
    # 4. 只保留符合 [48, 72] 的 bundles
    # 5. 检查是否有改进
```

**停止条件**：
- ✅ 所有 bundles 都在 [48, 72] 范围内
- ⚠️ 待重组的 segments 总地址数 < 48（无法组成合格 bundle）
- ⚠️ 本轮迭代没有改进
- ⚠️ 达到最大迭代次数（5 次）

**效果**：
```
迭代前: 2580 tiny bundles (< 48), 0 oversized
迭代 1: 145 valid bundles formed from 6800 segments
迭代 2: 5 valid bundles formed from 350 segments
✅ 所有 bundles 在 [48, 72] 范围内

最终: 最大化可用 bundles 数量
```

---

## 💡 如何选择？

### 选择 Greedy（5 步）如果：
- 你想要最少的 bundles 数量
- 你不在意最大值
- 你追求灵活性
- 用于研究/实验

### 选择 Multi-BFS + Regroup（9 步）如果：
- ✅ 你需要现场工作可预测性
- ✅ 你不能接受 700+ 的 bundles
- ✅ 你需要严格的大小控制 [48, 72]
- ✅ 你想最大化可用数据
- ✅ **推荐用于实际部署**

---

## 📁 输出文件

### quick_comparison_hard_constraint.py 生成：

```
outputs/bundles/DH/
├── bundles_multibfs_regroup.parquet           ← Multi-BFS 数据
├── bundles_multibfs_regroup_map.html          ← 地图
└── comparison_greedy_vs_multibfs_regroup.png  ← 对比图
```

### quick_filter_bundles.py 生成（可选）：

```
outputs/bundles/DH/
├── bundles_multibfs_regroup_filtered.parquet      ← 过滤后数据
├── bundles_multibfs_regroup_filtered_map.html     ← 保留的地图
└── bundles_multibfs_regroup_dropped_map.html      ← 筛掉的地图
```

---

## 🎯 总结

**Greedy (5 步, 无约束)**：
- 灵活但不可控，会产生极端值 (700+ addresses)

**Multi-BFS + Regroup (9 步, 硬约束 + 重组)**：
1. ✅ 严格控制：Max ≤ 72 addresses (target=60, 1.2x)
2. ✅ Split 智能：避免产生 < 40 的碎片
3. ✅ 循环重组：最大化可用 bundles ([48, 72])
4. ✅ 路线连续性：保证所有 bundle 都是连续的
5. ✅ 可预测性：适合现场工作部署

**关键改进**：
- Step 5: Split 带剩余检查（避免碎片）
- Step 9: 循环重组（最大化数据利用率）
- 范围控制：严格限制在 [48, 72] addresses

---

**创建日期**: 2025-12-18
**最后更新**: 2025-12-23
**当前版本**: Multi-BFS + Regroup (9 步)
**适用版本**: San Diego 311 Field Prep v1.0
