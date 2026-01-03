# `_merge_tiny_bundles` vs `_sweep_attach_residuals` 约束详解

## 📋 两个函数的角色对比

| 函数 | 作用时机 | 处理对象 | 目的 |
|---|---|---|---|
| **`_merge_tiny_bundles_connected`** | 调用 3 次（Step 3, 6, 8） | 已分配的 **tiny bundles** | 消除碎片化 |
| **`_sweep_attach_residuals`** | 调用 1 次（Step 4） | **未分配的** segments | 回收剩余路段 |

---

## 🔍 函数 1: `_merge_tiny_bundles_connected` - 连通性合并小束

### 📌 作用

合并那些太小的 bundles（单路段 或 < min_bundle_sfh）到附近的**连通的** bundle。

### 🎯 处理对象

```python
# 识别 tiny bundles
tiny_mask = (bundle_seg_count <= 1)  # 单路段
if min_bundle_sfh is not None:
    tiny_mask |= (bundle_addr_total < min_bundle_sfh)  # 或小于阈值

# 例如：
tiny_bundle_A = 8 addresses (太小！)
tiny_bundle_B = 1 segment (singleton)
```

### 🔄 Greedy 版本约束（无约束）

```python
# bundle.py line 560-564
# 找最近的候选 bundle (基于距离)
nearest_bid = min(cand, key=lambda b: src_cent.distance(bundle_cent[b]))

# 直接合并，完全不检查大小！
g.loc[g["bundle_id"] == bid, "bundle_id"] = nearest_bid
```

**实际效果**：
```
Iteration 1:
  tiny_bundle (40 addrs) + nearest_bundle (120 addrs) → 160 addrs ✓

Iteration 2:
  tiny_bundle (35 addrs) + same_bundle (160 addrs) → 195 addrs ✓

Iteration 3:
  tiny_bundle (50 addrs) + same_bundle (195 addrs) → 245 addrs ⚠️

...重复多次...

Final:
  bundle → 700+ addrs 💥 极端值！
```

---

### 🔒 Multi-BFS 版本约束

Multi-BFS 版本使用**连通性合并**，并根据步骤使用不同约束：

#### Step 3: 连通性合并（❌ 无约束，优先最小）

```python
# bundle_hard_constraint.py Step 3
# 只合并到有 endpoint 共享的 bundles
# 优先选择最小的候选
# ❌ 无大小约束

for tiny_bundle in tiny_bundles:
    # 找所有连通的邻居
    connected_neighbors = [b for b in neighbors if shares_endpoint(tiny_bundle, b)]

    # 优先选择最小的
    smallest_neighbor = min(connected_neighbors, key=lambda b: bundle_sizes[b])

    # 直接合并（不检查大小）
    merge(tiny_bundle, smallest_neighbor)
```

**特点**：
- ✅ 只合并到连通的邻居（保证路线连续性）
- ✅ 优先选择最小的候选（自然倾向于平衡）
- ❌ 无大小约束（允许灵活合并）
- 💡 后续 Step 5 会拆分超大的 bundles

#### Step 6, 8: 连通性合并 + 硬约束 (1.2x = 72 addresses)

```python
# bundle_hard_constraint.py Step 6, 8
hard_max = target_addrs * 1.2  # 例如 60 * 1.2 = 72

# 检查每个连通的候选 bundle
tiny_addrs = 40
valid_cand = []

for b in connected_neighbors:
    current_size = bundle_sizes[b]  # 例如 50
    if current_size + tiny_addrs <= hard_max:  # 50 + 40 = 90 > 72?
        valid_cand.append(b)  # NO! 拒绝

# 从合格的候选中选择最小的
if valid_cand:
    chosen = min(valid_cand, key=lambda b: bundle_sizes[b])
    merge(tiny_bundle, chosen)
else:
    rejected_count += 1
    # 保留这个 tiny bundle，不合并
```

**实际效果**：
```
Step 6:
  tiny_bundle (40 addrs) + smallest_connected (50 addrs)
  → 50 + 40 = 90 ≤ 72? NO ❌ → 拒绝！

  tiny_bundle (40 addrs) + next_smallest (30 addrs)
  → 30 + 40 = 70 ≤ 72? YES ✓ → 合并

Final:
  bundle → ≤ 72 ✅
  rejected tiny bundles → 进入 Step 9 重组
```

---

## 🔍 函数 2: `_sweep_attach_residuals` - 附加剩余路段

### 📌 作用

把那些还没分配到任何 bundle 的路段（residual segments）附加到附近的 bundle。

### 🎯 处理对象

```python
# 找未分配的路段
unassigned = g["bundle_id"].isna()  # NaN = 未分配

# 例如：
residual_segment_1 = 15 addresses (孤立路段)
residual_segment_2 = 8 addresses (连接失败的路段)
```

### ⚠️ 软约束（Greedy 和 Multi-BFS 版本都一样）

```python
# bundle.py line 116-119 (Multi-BFS 版本也是一样的)
def can_accept(bid, add):
    if soft_max_bundle_sfh is None:
        return True  # 如果没设限，任何都可以

    # 软约束：允许到 1.1x
    # 例如 target=60 → soft_max = 66
    return (totals.get(bid, 0) + add) <= soft_max_bundle_sfh * 1.10
```

**关键逻辑**：

```python
# line 151-156
chosen = None
for bID in ranked:  # ranked = 按距离排序的候选
    if can_accept(bID, sfh_i):  # 检查软约束
        chosen = bID
        break

if chosen is None:
    chosen = ranked[0]  # ⚠️ fallback: 即使超限，还是选最近的！
```

**实际效果**：
```
Scenario 1: 有符合软约束的候选 (target=60, soft_max=66)
  residual_segment (15 addrs)
  candidate_bundle_A (60 addrs): 60 + 15 = 75 ≤ 66? NO
  candidate_bundle_B (50 addrs): 50 + 15 = 65 ≤ 66? YES ✓
  → 选择 B

Scenario 2: 所有候选都超过软约束
  residual_segment (15 addrs)
  candidate_bundle_A (65 addrs): 65 + 15 = 80 ≤ 66? NO
  candidate_bundle_B (62 addrs): 62 + 15 = 77 ≤ 66? NO
  → 仍然选择最近的 A ⚠️ (fallback)
  → 结果：A = 80 (超过了软约束，但会被 Step 5 split)
```

---

## 📊 两个函数的约束对比

### `_merge_tiny_bundles` / `_merge_tiny_bundles_connected` - 合并小束

| 版本 | 约束类型 | 选择标准 | 拒绝行为 | 结果 |
|---|---|---|---|---|
| **Greedy** | ❌ 无约束 | 距离最近 | 从不拒绝 | 700+ 极端值 |
| **Multi-BFS Step 3** | ❌ 无约束 | 连通 + 最小 | 从不拒绝连通的 | 允许大 bundle，后续拆分 |
| **Multi-BFS Step 6, 8** | 🔒 硬约束 1.2x | 连通 + 最小 + 符合约束 | 超限就拒绝 | ≤ 72，保留 tiny |

### `_sweep_attach_residuals` - 附加剩余

| 版本 | 约束类型 | 检查逻辑 | 拒绝行为 | 结果 |
|---|---|---|---|---|
| **Greedy** | ⚠️ 软约束 1.1x | `size + seg ≤ soft_max` | 优先不超，但有 fallback | 可能略微超限 |
| **Multi-BFS** | ⚠️ 软约束 1.1x | **完全一样** | **完全一样** | **完全一样** |

**关键**：Multi-BFS 版本**没改** `_sweep_attach_residuals`！

---

## 🔄 完整执行流程对比

### Greedy 版本 (bundle.py, 5 步)

```
1. Multi-BFS 初始分配
   → 创建 ~1800 bundles (大部分 ~60 addrs)

2. _merge_tiny_bundles()  ← ❌ 无约束 + 距离
   → 合并 tiny bundles，不检查大小
   → 某些 bundles 变成 200-300 addrs

3. _sweep_attach_residuals()  ← ⚠️ 软约束 1.1x
   → 附加剩余 segments
   → 允许到 66，但有 fallback

4. _enforce_endpoint_contiguity()
   → 拆分不连续的 bundles
   → 可能产生新的 tiny bundles

5. _merge_tiny_bundles()  ← ❌ 无约束 + 距离
   → 再次合并，还是不检查大小
   → 某些 bundles 继续增大 → 700+ addrs 💥

Final: Max = 700+, Tiny = 0
```

---

### Multi-BFS + Regroup 版本 (bundle_hard_constraint.py, 9 步)

```
Step 1: Build graph + components
   → 构建邻接图，识别连通分量

Step 2: Grow bundles (multi-BFS balanced)
   → 创建 ~1800 bundles (大部分 ~60 addrs)

Step 3: Merge tiny (connected, ❌ no constraint, prefer smallest)
   → 连通性合并，无约束
   → 优先合并到最小的连通邻居
   → 保留一些 tiny bundles

Step 4: Sweep residuals (soft 1.1x)  ← ⚠️ 软约束 1.1x (同 Greedy)
   → 附加剩余 segments
   → 允许到 66 addresses (1.1x)，但有 fallback
   → 可能有少数 bundles 略微超过 66

Step 5: Split oversized (> 60, with remainder check)  ← 🔒 自动拆分
   → Split threshold: 60 addresses (1.0x target)
   → 拆分超过 60 addresses 的 bundles
   → ✅ 剩余检查：避免产生 < 40 的碎片
   → BFS 切割，保证连通性

Step 6: Merge tiny (connected, 🔒 hard 1.2x)
   → 连通性合并 + 硬约束 72 addresses (1.2x)
   → 只合并到有 endpoint 共享的 bundles
   → 优先选择最小的候选
   → 清理 split 产生的碎片

Step 7: Enforce contiguity
   → 强制路线连续性，拆分不连续的 bundles

Step 8: Final cleanup (connected, 🔒 hard 1.2x)
   → 最终清理，连通性合并 + 硬约束
   → Max ≤ 72 ✅

Step 9: Regroup invalid bundles ([48, 72])  ← 🆕 循环重组
   → 找出 < 48 或 > 72 的 bundles
   → 打散并重新组合（目标 60）
   → 只保留符合 [48, 72] 的 bundles
   → 最多迭代 5 次
   → 最大化可用 bundles 数量

Final: Max ≤ 72 ✅, 路线连续性 ✅, Range = [48, 72] ✅
```

**关键改进**：
- Step 3 使用连通性合并（无约束，优先最小）
- Step 5 新增自动拆分（1.0x = 60 addresses），带剩余检查
- Step 6, 8 使用连通性合并 + 硬约束（1.2x = 72 addresses）
- Step 9 新增循环重组（最大化可用数据）

---

## 🎯 为什么软约束的 `_sweep_attach_residuals` 不改？

### 原因 1: 影响范围小

```python
# residual segments 通常很少
unassigned.sum()  # 通常 < 100 segments

# 每个 segment 通常很小
residual_segment.sfh_addr_count  # 通常 5-20 addresses

# 即使 fallback，增加也有限
bundle (65) + residual (15) = 80  # 超过 66，但会被 Step 5 拆分
```

### 原因 2: 软约束已经足够

```python
# 大部分情况下，能找到符合 1.1x 的候选
if can_accept(bID, sfh_i):  # 50 + 15 ≤ 66? YES
    chosen = bID  # 优先选择不超限的

# 只有极少数情况才 fallback
# 而且 fallback 的增量很小
# Step 5 会自动拆分超过 60 的 bundles
```

### 原因 3: 避免过度碎片化

```python
# 如果用硬约束，可能导致很多 residual 无法分配
if all_candidates_exceed_hard_max:
    residual_segment.bundle_id = NaN  # 仍未分配
    # → 大量孤立路段，不利于现场工作
```

### 原因 4: Step 5 自动拆分机制

```python
# 即使 sweep 产生了一些超过 60 的 bundles
# Step 5 会自动拆分它们
if bundle_size > target_addrs:  # > 60
    split_bundle_with_remainder_check()
    # → 拆分成多个 ≤ 60 的 bundles
```

---

## 💡 关键洞察

### Multi-BFS 的三种约束策略

Multi-BFS 版本在不同阶段使用了**三种不同的约束策略**：

#### 1️⃣ 连通性合并 - 无约束（Step 3）
- **选择标准**：基于 endpoint 共享（只选连通的 bundles） + 优先最小
- **使用时机**：Step 3 (早期清理)
- **约束类型**：❌ 无约束
- **保证连通性**：✅ 是
- **优先策略**：选择最小的连通邻居
- **目的**：早期灵活合并，保持路线连续性

#### 2️⃣ 连通性合并 + 硬约束（Step 6, 8）
- **选择标准**：基于 endpoint 共享 + 硬约束 1.2x + 优先最小
- **使用时机**：Step 6, 8 (split 后清理 + 最终清理)
- **约束类型**：🔒 硬约束 1.2x (72 addresses)
- **保证连通性**：✅ 是
- **优先策略**：从合格候选中选择最小的
- **目的**：清理碎片 + 保证连通性 + 控制最大值

#### 3️⃣ Sweep 软约束（Step 4）
- **选择标准**：基于距离 + 容量 + 软约束 1.1x
- **使用时机**：Step 4 (中期附加残留)
- **约束类型**：⚠️ 软约束 1.1x (66 addresses)，有 fallback
- **保证连通性**：N/A
- **优先策略**：优先不超限，但允许 fallback
- **目的**：回收剩余路段，避免孤立

---

### 为什么需要三种策略？

**Step 3 - 无约束连通性合并**：
- 早期阶段，数据分布还不确定
- 允许灵活合并，优先最小候选自然倾向于平衡
- 保证连通性，为后续步骤打好基础

**Step 4 - Sweep 软约束**：
- 处理残留 segments（通常很小）
- 软约束 + fallback 平衡了数据利用率和大小控制
- 即使略微超限，Step 5 会自动拆分

**Step 5 - Split (1.0x threshold)**：
- 自动拆分超过 60 addresses 的 bundles
- 剩余检查避免产生 < 40 的碎片
- 重置过大的 bundles

**Step 6, 8 - 硬约束连通性合并**：
- 清理 split 产生的碎片
- 硬约束确保不会再产生超过 72 的 bundles
- 连通性保证路线连续性

**Step 9 - 循环重组**：
- 最大化符合 [48, 72] 范围的 bundles 数量
- 打散不合格 bundles 并重新组合
- 确保最终结果的可用性

---

## 📊 数值示例对比 (target=60)

### Scenario 1: Merge tiny bundle (40 addrs)

**Greedy - 无约束 + 距离**：
```
candidate_A (60 addrs, 距离 100m)
  60 + 40 = 100? ✓ 合并（不检查）

candidate_B (120 addrs, 距离 80m)  ← 最近的
  120 + 40 = 160? ✓ 合并（不检查）

candidate_C (500 addrs, 距离 90m)
  500 + 40 = 540? ✓ 合并（不检查）💥
```

**Multi-BFS Step 3 - 无约束 + 连通 + 最小**：
```
candidate_A (60 addrs, connected)
candidate_B (120 addrs, connected)
candidate_C (30 addrs, connected)  ← 最小的连通邻居

→ 选择 C: 30 + 40 = 70 ✓ 合并
   (虽然超过 60，但 Step 5 会拆分)
```

**Multi-BFS Step 6 - 硬约束 1.2x (72) + 连通 + 最小**：
```
candidate_A (50 addrs, connected)
  50 + 40 = 90 ≤ 72? ❌ 拒绝

candidate_B (60 addrs, connected)
  60 + 40 = 100 ≤ 72? ❌ 拒绝

candidate_C (30 addrs, connected)  ← 最小的连通邻居
  30 + 40 = 70 ≤ 72? ✓ 合并

→ 只能合并到 candidate_C
```

---

### Scenario 2: Attach residual segment (15 addrs)

**软约束 - 1.1x (66)** (两个版本都一样)：
```
candidate_A (60 addrs)
  60 + 15 = 75 ≤ 66? ❌ 不符合软约束

candidate_B (50 addrs)
  50 + 15 = 65 ≤ 66? ✓ 符合软约束

→ 优先选择 B

如果所有候选都超过 66:
→ fallback 到最近的
→ 结果可能 > 66，但 Step 5 会拆分
```

---

## 🎯 总结

### 函数对比表

| 特性 | Greedy `_merge_tiny_bundles` | Multi-BFS Step 3 | Multi-BFS Step 6, 8 | `_sweep_attach_residuals` |
|---|---|---|---|---|
| **处理对象** | Tiny bundles (30-50 addrs) | Tiny bundles | Split 碎片 bundles | Residual segments (5-20 addrs) |
| **选择标准** | 距离（centroid） | 连通性 + 最小 | 连通性 + 最小 + 硬约束 | 距离 + 容量 |
| **约束类型** | ❌ 无约束 | ❌ 无约束 | 🔒 硬约束 1.2x | ⚠️ 软约束 1.1x |
| **保证连通性** | ❌ 否 | ✅ 是 | ✅ 是 | N/A |
| **拒绝行为** | 从不拒绝 | 从不拒绝连通的 | 超限就拒绝 | 优先不超，有 fallback |
| **结果** | 700+ 极端值 | 允许大 bundle | ≤ 72 | 可能略超 66，但会被拆分 |

### 核心结论

**Greedy vs Multi-BFS**：

1. **Greedy 的问题**：
   - 基于距离，可能破坏连通性
   - 无约束，产生 700+ 极端值
   - 累积效应严重

2. **Multi-BFS 的改进**：
   - Step 3: 连通性 + 无约束 + 优先最小 → 灵活合并，保持连通性
   - Step 5: Split (1.0x) + 剩余检查 → 拆分超大 bundles，避免碎片
   - Step 6, 8: 连通性 + 硬约束 1.2x → 清理碎片，控制最大值
   - Step 9: 循环重组 → 最大化可用数据

3. **Sweep 保持不变的原因**：
   - 软约束已经足够（影响小，有 fallback）
   - Step 5 会自动拆分超限的 bundles
   - 避免过度碎片化

**执行顺序的重要性**：
```
Step 3: 连通性合并（无约束，优先最小）
  → 早期灵活清理，保持连通性
        ↓
Step 4: Sweep 软约束
  → 回收残留，允许 fallback
        ↓
Step 5: Split (1.0x) + 剩余检查
  → 拆分超大 bundles，避免碎片
        ↓
Step 6: 连通性合并 + 硬约束 1.2x
  → 清理碎片 + 控制最大值
        ↓
Step 7: Enforce contiguity
  → 最终验证连续性
        ↓
Step 8: 最终清理 + 硬约束
  → 最后机会合并
        ↓
Step 9: 循环重组 [48, 72]
  → 最大化可用 bundles 数量
```

---

**创建日期**: 2025-12-18
**最后更新**: 2025-12-23
**当前版本**: Multi-BFS + Regroup (9 步)
**适用版本**: San Diego 311 Field Prep v2.0
