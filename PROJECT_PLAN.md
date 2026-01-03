# Fieldprep - Bundle 算法改进项目计划

**项目名称**: Bundle 平衡性优化 - Multi-source BFS 实现
**项目负责人**: Claude Code + User
**开始日期**: 2025-12-18
**当前状态**: 🚧 进行中

---

## 目录

1. [项目概述](#1-项目概述)
2. [当前问题分析](#2-当前问题分析)
3. [解决方案](#3-解决方案)
4. [修改记录](#4-修改记录)
5. [实施计划](#5-实施计划)
6. [测试计划](#6-测试计划)
7. [未来规划](#7-未来规划)
8. [回滚方案](#8-回滚方案)

---

## 1. 项目概述

### 1.1 背景

Fieldprep 项目的 Bundle 模块用于将街道路段分组为"束"，以便分配给现场调查员。当前使用的**贪婪 BFS 算法**虽然快速，但存在束大小不均衡的问题，导致调查员工作量分配不公平。

### 1.2 目标

- ✅ **主要目标**: 实现 Multi-source BFS 算法，显著提升束大小的平衡性
- ✅ **次要目标**: 保持向后兼容，支持多种算法切换
- ✅ **性能目标**: 保持计算时间在可接受范围（< 30秒 for 15,000 路段）
- ✅ **质量目标**: 变异系数 (CV) 从 ~25% 降低到 ~12%

### 1.3 成功指标

| 指标 | 当前值 | 目标值 | 度量方法 |
|-----|--------|--------|---------|
| 变异系数 (CV) | ~25% | < 15% | std / mean |
| ±20% 范围内束的比例 | ~60% | > 80% | 统计分析 |
| 计算时间 (15k 路段) | ~5秒 | < 15秒 | 性能测试 |
| 地理连续性 | 100% | 100% | 端点连通性检查 |

---

## 2. 当前问题分析

### 2.1 问题描述

**问题**: Bundle 大小不均衡

**表现**:
```
示例输出（当前贪婪算法）:
  束 1: 45 个地址   (目标: 120, 偏差: -62%)
  束 2: 178 个地址  (目标: 120, 偏差: +48%)
  束 3: 95 个地址   (目标: 120, 偏差: -21%)
  束 4: 152 个地址  (目标: 120, 偏差: +27%)

  变异系数: 27.3%
```

### 2.2 根本原因

**贪婪算法的问题**:

1. **局部最优**: 从随机种子开始贪婪生长，达到目标后立即停止
2. **先到先得**: 早期束可能占据"富矿"（高密度区域）
3. **无全局视野**: 不考虑整体分布，只关注单个束的增长

**代码位置** (bundle.py:148-180):
```python
while q and total < target_addrs:  # ← 达到目标就停止
    u = q.popleft()
    cur.append(u)
    total += int(g.loc[u, "sfh_addr_count"])

    # 优先添加高地址邻居（可能导致过度生长）
    for v in sorted(nbrs[u] & remaining, key=..., reverse=True):
        remaining.remove(v)
        q.append(v)
```

### 2.3 影响

- **调查员工作量不公**: 有的调查员只需访问 45 个地址，有的需要 178 个
- **时间预算难以规划**: 无法准确预估完成时间
- **资源浪费**: 需要更多时间重新平衡任务

---

## 3. 解决方案

### 3.1 技术方案选择

**选定方案**: Multi-source BFS with Backtracking

**理由**:
- ✅ 平衡性显著提升（预期 CV < 15%）
- ✅ 无外部依赖（纯 Python + NumPy/GeoPandas）
- ✅ 中等计算成本（预期 < 15秒）
- ✅ 保持地理连续性
- ✅ 易于理解和维护

**备选方案**:
- METIS: 需要外部库，复杂度高
- Spectral Clustering: 对大图性能较差
- 改进贪婪: 治标不治本

### 3.2 算法设计

#### 阶段 1: 多源同步 BFS

```
1. 估计束数量 n = 总地址数 / target_addrs
2. 选择 n 个空间分散的种子（K-means++ 风格）
3. 从所有种子同时开始 BFS
4. 边界冲突解决：将边界路段分配给当前较小的束
```

#### 阶段 2: 回溯平衡

```
1. 识别过大和过小的束（±15% 阈值）
2. 从过大束释放边缘路段
3. 将释放的路段分配给邻近的小束
4. 迭代直到收敛或达到最大迭代次数
```

### 3.3 兼容性设计

**向后兼容策略**:

```python
# CLI 新增 --method 参数（默认保持原行为）
python cli.py bundle --method greedy      # 原贪婪算法（默认）
python cli.py bundle --method multi_bfs   # 新 Multi-source BFS
```

**配置文件支持** (可选):
```yaml
# config/params.yaml
bundle:
  default_method: "multi_bfs"  # 或 "greedy"
  balance_tolerance: 0.15      # ±15%
  max_rebalance_iterations: 10
```

---

## 4. 修改记录

### 4.1 代码修改

#### 修改 1: bundle.py - 添加新算法函数

**文件**: `fieldprep/src/sd311_fieldprep/bundle.py`
**日期**: 2025-12-18
**类型**: 功能添加
**影响范围**: 无（新增函数，不影响现有代码）

**新增函数**:
1. `_multi_source_balanced_bfs()` - 主算法
2. `_select_spatially_distributed_seeds()` - 种子选择
3. `_backtrack_rebalance()` - 回溯平衡

**代码位置**: 在 `_grow_bundles_in_component()` 函数之后插入

**变更详情**:
```python
# 新增约 200 行代码
def _multi_source_balanced_bfs(...):
    """Multi-source BFS with balanced growth"""
    # 实现详见下文

def _select_spatially_distributed_seeds(...):
    """K-means++ style seed selection"""
    # 实现详见下文

def _backtrack_rebalance(...):
    """Backtrack to rebalance oversized bundles"""
    # 实现详见下文
```

---

#### 修改 2: bundle.py - 修改主流程函数

**文件**: `fieldprep/src/sd311_fieldprep/bundle.py`
**函数**: `_build_connected_bundles()`
**日期**: 2025-12-18
**类型**: 功能增强
**影响范围**: 低（添加条件分支，保留原逻辑）

**修改前** (bundle.py:421-470):
```python
def _build_connected_bundles(segs_m, seg_id_col, target_addrs, join_tol_m, seed, min_bundle_sfh):
    # ... 步骤 1-2

    for comp in sorted(set(comp_of.values())):
        comp_indices = [i for i, c in comp_of.items() if c == comp]
        local_map, _ = _grow_bundles_in_component(...)  # ← 固定使用贪婪
        # ...
```

**修改后**:
```python
def _build_connected_bundles(segs_m, seg_id_col, target_addrs, join_tol_m, seed, min_bundle_sfh,
                             method="greedy"):  # ← 新增参数，默认保持原行为
    # ... 步骤 1-2

    for comp in sorted(set(comp_of.values())):
        comp_indices = [i for i, c in comp_of.items() if c == comp]

        # ← 新增条件分支
        if method == "multi_bfs":
            local_map = _multi_source_balanced_bfs(g, nbrs, comp_indices, target_addrs, seed)
        else:  # method == "greedy" (默认)
            local_map, _ = _grow_bundles_in_component(g, nbrs, comp_indices, target_addrs, rng)
        # ...
```

---

#### 修改 3: bundle.py - 修改 run_bundle() 函数

**文件**: `fieldprep/src/sd311_fieldprep/bundle.py`
**函数**: `run_bundle()`
**日期**: 2025-12-18
**类型**: 参数传递
**影响范围**: 低（新增可选参数）

**修改**:
```python
def run_bundle(session: str,
               target_addrs: int,
               join_tol_m: float = 15.0,
               seed: int = 42,
               tag: str | None = None,
               min_bundle_sfh: int | None = None,
               method: str = "greedy"):  # ← 新增参数
    # ...

    bundled = _build_connected_bundles(
        segs_m,
        seg_id_col=seg_id,
        target_addrs=target_addrs,
        join_tol_m=join_tol_m,
        seed=seed,
        min_bundle_sfh=min_bundle_sfh,
        method=method  # ← 传递新参数
    )
```

---

#### 修改 4: cli.py - 添加命令行参数

**文件**: `fieldprep/cli.py`
**日期**: 2025-12-18
**类型**: CLI 增强
**影响范围**: 低（可选参数，默认值保持原行为）

**修改位置**: bundle 子命令定义 (cli.py:46-71)

**新增参数**:
```python
p2.add_argument(
    "--method",
    choices=["greedy", "multi_bfs"],
    default="greedy",
    help="Bundling algorithm: 'greedy' (fast, default) or 'multi_bfs' (balanced)"
)
```

**调用传递** (cli.py:64-71):
```python
p2.set_defaults(func=lambda a: run_bundle(
    session=a.session,
    target_addrs=a.target_addrs,
    join_tol_m=a.join_tol_m,
    seed=a.seed,
    tag=a.tag,
    min_bundle_sfh=a.min_bundle_sfh,
    method=a.method  # ← 传递新参数
))
```

---

### 4.2 文档修改

#### 修改 5: SWEEP_BUNDLE_TECHNICAL_GUIDE.md - 添加算法对比章节

**文件**: `SWEEP_BUNDLE_TECHNICAL_GUIDE.md`
**日期**: 2025-12-18
**类型**: 文档更新

**新增章节**:
- 6.4 算法对比：贪婪 vs Multi-source BFS
- 8.6 使用新算法的示例

---

#### 修改 6: README.md - 更新使用说明

**文件**: `fieldprep/README.md`
**日期**: 2025-12-18
**类型**: 文档更新

**新增内容**:
```markdown
## Bundle 算法选择

Fieldprep 支持两种 bundle 算法：

1. **greedy** (默认): 快速贪婪算法，适合原型和快速迭代
2. **multi_bfs**: 多源平衡 BFS，显著提升束大小的均衡性

使用示例：
```bash
# 使用平衡算法
python cli.py bundle --session DH --target_addrs 120 --method multi_bfs
```
```

---

### 4.3 测试文件添加

#### 新增 1: 单元测试

**文件**: `fieldprep/tests/test_bundle_algorithms.py` (新建)
**日期**: 2025-12-18
**类型**: 测试

**测试用例**:
- `test_greedy_backward_compatibility()` - 确保原算法未受影响
- `test_multi_bfs_balance()` - 验证新算法平衡性
- `test_seed_selection()` - 测试种子选择算法
- `test_backtrack_rebalance()` - 测试回溯平衡

---

#### 新增 2: 性能对比脚本

**文件**: `fieldprep/scripts/compare_bundle_algorithms.py` (新建)
**日期**: 2025-12-18
**类型**: 分析工具

**功能**:
- 并排运行两种算法
- 生成对比报告（均值、标准差、CV、分布图）
- 保存结果到 CSV

---

### 4.4 配置文件修改（可选）

#### 可选 1: params.yaml - 添加算法配置

**文件**: `fieldprep/config/params.yaml`
**日期**: 待定
**类型**: 配置增强
**状态**: 🔜 计划中

**新增配置**:
```yaml
bundle:
  default_method: "multi_bfs"      # 或 "greedy"
  balance_tolerance: 0.15          # ±15% 容差
  max_rebalance_iterations: 10
  seed_selection_method: "spatial" # 或 "random"
```

---

## 5. 实施计划

### 5.1 时间线

| 阶段 | 任务 | 预计时间 | 状态 | 完成日期 |
|-----|------|---------|------|---------|
| **Phase 1** | 项目规划和文档 | 2小时 | 🚧 进行中 | 2025-12-18 |
| 1.1 | 创建 PROJECT_PLAN.md | 1小时 | 🚧 进行中 | - |
| 1.2 | 需求分析和方案设计 | 1小时 | ✅ 完成 | 2025-12-18 |
| **Phase 2** | 核心算法实现 | 4小时 | ⏳ 待开始 | - |
| 2.1 | 实现种子选择函数 | 1小时 | ⏳ | - |
| 2.2 | 实现多源 BFS 主算法 | 2小时 | ⏳ | - |
| 2.3 | 实现回溯平衡函数 | 1小时 | ⏳ | - |
| **Phase 3** | 集成和接口 | 2小时 | ⏳ 待开始 | - |
| 3.1 | 修改 _build_connected_bundles | 0.5小时 | ⏳ | - |
| 3.2 | 修改 run_bundle 和 CLI | 0.5小时 | ⏳ | - |
| 3.3 | 向后兼容性检查 | 1小时 | ⏳ | - |
| **Phase 4** | 测试和验证 | 3小时 | ⏳ 待开始 | - |
| 4.1 | 单元测试编写 | 1.5小时 | ⏳ | - |
| 4.2 | 端到端测试 | 1小时 | ⏳ | - |
| 4.3 | 性能基准测试 | 0.5小时 | ⏳ | - |
| **Phase 5** | 文档和发布 | 2小时 | ⏳ 待开始 | - |
| 5.1 | 更新技术文档 | 1小时 | ⏳ | - |
| 5.2 | 更新用户文档 | 0.5小时 | ⏳ | - |
| 5.3 | 创建发布说明 | 0.5小时 | ⏳ | - |
| **总计** | | **13小时** | | |

### 5.2 里程碑

- ✅ **M1**: 方案设计完成 (2025-12-18)
- ⏳ **M2**: 核心算法实现完成 (预计 2025-12-19)
- ⏳ **M3**: 集成测试通过 (预计 2025-12-19)
- ⏳ **M4**: 文档完成，准备发布 (预计 2025-12-20)

---

## 6. 测试计划

### 6.1 单元测试

#### 测试 1: 种子选择算法

**目标**: 验证种子空间分布均匀

```python
def test_seed_selection_spatial_distribution():
    # 创建测试数据（9x9 网格）
    indices = list(range(81))
    g = create_grid_gdf(9, 9)

    # 选择 9 个种子
    seeds = _select_spatially_distributed_seeds(g, indices, 9, rng)

    # 验证：
    # 1. 数量正确
    assert len(seeds) == 9

    # 2. 空间分散（最小距离 > 阈值）
    for i, s1 in enumerate(seeds):
        for s2 in seeds[i+1:]:
            dist = g.loc[s1].geometry.distance(g.loc[s2].geometry)
            assert dist > 100  # 至少 100 米
```

---

#### 测试 2: 多源 BFS 平衡性

**目标**: 验证束大小的变异系数 < 15%

```python
def test_multi_bfs_balance():
    # 加载真实数据或创建模拟数据
    g, nbrs, comp_indices = load_test_data()
    target = 120

    # 运行算法
    assignment = _multi_source_balanced_bfs(g, nbrs, comp_indices, target, seed=42)

    # 统计束大小
    sizes = compute_bundle_sizes(g, assignment)

    # 验证平衡性
    mean_size = np.mean(sizes)
    std_size = np.std(sizes)
    cv = std_size / mean_size

    assert cv < 0.15, f"CV {cv:.2%} exceeds threshold 15%"
    assert all(s >= target * 0.85 for s in sizes), "Some bundles too small"
    assert all(s <= target * 1.15 for s in sizes), "Some bundles too large"
```

---

#### 测试 3: 向后兼容性

**目标**: 确保贪婪算法行为未改变

```python
def test_greedy_backward_compatibility():
    # 加载相同的测试数据
    g, nbrs, comp_indices = load_test_data()
    target = 120
    seed = 42

    # 运行旧版本逻辑（保存的基准结果）
    expected_assignment = load_baseline_greedy_result()

    # 运行当前版本（method="greedy"）
    actual_assignment = run_with_method(g, nbrs, comp_indices, target, seed, "greedy")

    # 验证结果一致
    assert actual_assignment == expected_assignment
```

---

### 6.2 集成测试

#### 测试 4: 端到端测试

**测试场景**: 完整的 sweep → bundle → emit 流程

```bash
# 1. Sweep
python cli.py sweep --buffers 25 --mins 6 --tag test_run

# 2. Bundle (贪婪)
python cli.py bundle --session DH --target_addrs 120 --method greedy --tag test_run
mv outputs/bundles/DH/bundles.parquet outputs/bundles/DH/bundles_greedy.parquet

# 3. Bundle (多源BFS)
python cli.py bundle --session DH --target_addrs 120 --method multi_bfs --tag test_run
mv outputs/bundles/DH/bundles.parquet outputs/bundles/DH/bundles_multi_bfs.parquet

# 4. 对比分析
python scripts/compare_bundle_algorithms.py \
  --greedy outputs/bundles/DH/bundles_greedy.parquet \
  --multi_bfs outputs/bundles/DH/bundles_multi_bfs.parquet \
  --target 120
```

**预期结果**:
```
Algorithm Comparison Report
═══════════════════════════

Greedy Algorithm:
  Mean: 118.3 addresses
  Std:  29.7
  CV:   25.1%
  Within ±20%: 58.3%

Multi-source BFS:
  Mean: 119.8 addresses
  Std:  14.2
  CV:   11.9% ✓
  Within ±20%: 87.5% ✓

Improvement: 52.5% reduction in CV
```

---

### 6.3 性能测试

#### 测试 5: 计算时间基准

**测试数据规模**:
- 小型: 1,000 路段
- 中型: 5,000 路段
- 大型: 15,000 路段

**测试代码**:
```python
import time

def benchmark_algorithms():
    datasets = {
        "small": load_data(1000),
        "medium": load_data(5000),
        "large": load_data(15000)
    }

    results = []

    for size, (g, nbrs, comp_indices) in datasets.items():
        # Greedy
        start = time.time()
        run_with_method(g, nbrs, comp_indices, 120, 42, "greedy")
        greedy_time = time.time() - start

        # Multi-BFS
        start = time.time()
        run_with_method(g, nbrs, comp_indices, 120, 42, "multi_bfs")
        multi_bfs_time = time.time() - start

        results.append({
            "size": size,
            "greedy_time": greedy_time,
            "multi_bfs_time": multi_bfs_time,
            "overhead": (multi_bfs_time / greedy_time - 1) * 100
        })

    return pd.DataFrame(results)
```

**预期结果**:
| 规模 | 贪婪 | Multi-BFS | 开销 |
|-----|------|-----------|------|
| 1k | 0.5s | 1.2s | +140% |
| 5k | 2.1s | 5.3s | +152% |
| 15k | 5.8s | 14.7s | +153% |

**验收标准**: Multi-BFS 在 15k 路段下 < 20 秒

---

## 7. 未来规划

### 7.1 短期改进（1-2 周）

#### 改进 1: 自适应算法选择

**目标**: 根据数据规模自动选择算法

**实现**:
```python
def auto_select_method(n_segments, target_addrs):
    if n_segments < 1000:
        return "multi_bfs"  # 小规模数据，优先平衡性
    elif n_segments > 10000:
        return "greedy"     # 大规模数据，优先速度
    else:
        return "multi_bfs"  # 中等规模，平衡性重要
```

**CLI 集成**:
```bash
python cli.py bundle --method auto  # 自动选择
```

---

#### 改进 2: 可视化对比工具

**目标**: 直观展示两种算法的差异

**功能**:
- 并排地图显示
- 束大小分布直方图
- 统计摘要表格

**工具**: `scripts/visualize_bundle_comparison.py`

---

### 7.2 中期扩展（1-2 个月）

#### 扩展 1: METIS 集成

**条件**: 用户反馈需要更高平衡性

**实现步骤**:
1. 添加 METIS Python 绑定到 requirements.txt (可选依赖)
2. 实现 `_balanced_partition_metis()` 函数
3. 添加 `--method metis` 选项
4. 性能测试和文档更新

**预期效果**: CV < 8%

---

#### 扩展 2: 约束优化

**目标**: 支持额外约束

**约束类型**:
- 不跨越主要道路（高速公路、大道）
- 遵守社区边界（CPD boundary）
- 限制束的空间跨度（最大半径）

**配置**:
```yaml
bundle:
  constraints:
    respect_major_roads: true
    max_spatial_extent_m: 1000
    respect_cpd_boundaries: true
```

---

### 7.3 长期愿景（3-6 个月）

#### 愿景 1: 机器学习优化

**目标**: 基于历史调查数据优化束的分配

**方法**:
- 收集调查员完成时间数据
- 训练模型预测"实际工作量"（而非地址数）
- 使用预测工作量而非地址数作为平衡目标

**效果**: 更精准的工作量平衡

---

#### 愿景 2: 实时动态调整

**目标**: 支持调查过程中的动态重新分配

**场景**:
- 调查员 A 提前完成 → 系统推荐从调查员 B 接手部分束
- 某个束因天气/安全原因需要跳过 → 自动重新分配

**技术栈**: Web API + 移动端集成

---

#### 愿景 3: 多目标优化

**目标**: 同时优化多个指标

**指标**:
1. 束大小平衡性
2. 空间紧凑性（减少行走距离）
3. 社区公平性（每个社区都有覆盖）
4. 调查员偏好（避免危险区域等）

**方法**: 帕累托前沿分析 + 交互式权重调整

---

## 8. 回滚方案

### 8.1 回滚触发条件

**触发回滚的情况**:
1. ❌ Multi-BFS 导致严重性能问题（> 60秒 for 15k 路段）
2. ❌ 破坏向后兼容性（贪婪算法结果改变）
3. ❌ 发现严重 Bug（束不连通、地址统计错误）
4. ❌ 用户反馈负面（实际效果不如预期）

### 8.2 回滚步骤

#### Step 1: 恢复 bundle.py

```bash
# 1. 备份当前版本
cp fieldprep/src/sd311_fieldprep/bundle.py \
   fieldprep/src/sd311_fieldprep/bundle.py.new

# 2. 从 Git 恢复旧版本
git checkout HEAD~1 -- fieldprep/src/sd311_fieldprep/bundle.py

# 3. 验证
python cli.py bundle --session DH --target_addrs 120  # 应该使用旧逻辑
```

---

#### Step 2: 恢复 cli.py

```bash
git checkout HEAD~1 -- fieldprep/cli.py
```

---

#### Step 3: 通知用户

**消息模板**:
```
紧急通知：Bundle 算法更新回滚

由于 [具体原因]，我们暂时回滚了 Multi-source BFS 算法更新。

影响：
- 2025-12-18 之后使用 --method multi_bfs 的用户

建议：
- 重新运行 bundle 命令（不带 --method 参数）
- 如有疑问，请联系技术支持

我们正在修复问题，预计 [日期] 重新发布。
```

---

### 8.3 部分回滚（保留代码但禁用）

如果只是暂时禁用新算法：

```python
# cli.py 中
p2.add_argument(
    "--method",
    choices=["greedy"],  # ← 移除 "multi_bfs"
    default="greedy",
    help="Bundling algorithm (multi_bfs temporarily disabled)"
)
```

---

## 9. 风险管理

### 9.1 技术风险

| 风险 | 可能性 | 影响 | 缓解措施 |
|-----|-------|------|---------|
| 性能不达标 | 中 | 高 | 1. 提前性能测试<br>2. 添加进度条<br>3. 支持算法切换 |
| 种子选择不均匀 | 低 | 中 | 1. 单元测试覆盖<br>2. 可视化验证 |
| 边界冲突处理 Bug | 中 | 中 | 1. 边界情况测试<br>2. 详细日志 |
| 回溯平衡不收敛 | 低 | 低 | 1. 最大迭代限制<br>2. 记录收敛状态 |

### 9.2 用户体验风险

| 风险 | 可能性 | 影响 | 缓解措施 |
|-----|-------|------|---------|
| 计算时间过长导致用户不耐烦 | 中 | 中 | 1. 添加进度条和预估时间<br>2. 文档说明预期时间 |
| 参数选择困难 | 低 | 低 | 1. 提供默认值<br>2. 文档提供使用指南 |
| 结果不如预期 | 低 | 高 | 1. 对比工具<br>2. 详细文档说明 |

---

## 10. 成功案例和最佳实践

### 10.1 预期成功案例

**案例 1: City Heights 社区调查**

**背景**:
- 300 个符合条件的路段
- 目标：每个束 120 个地址
- 4 位调查员

**贪婪算法结果**:
```
调查员 A: 78 个地址  (1.5 小时)
调查员 B: 145 个地址 (3.2 小时)
调查员 C: 132 个地址 (2.8 小时)
调查员 D: 95 个地址  (2.0 小时)

问题：工作量不均，B 需要加班
```

**Multi-BFS 结果**:
```
调查员 A: 115 个地址 (2.4 小时)
调查员 B: 122 个地址 (2.5 小时)
调查员 C: 118 个地址 (2.5 小时)
调查员 D: 125 个地址 (2.6 小时)

效果：工作量均衡，可预测性强
```

---

### 10.2 使用最佳实践

#### 实践 1: 首次使用

```bash
# 1. 先用贪婪算法快速预览
python cli.py bundle --session DH --target_addrs 120 --method greedy

# 2. 检查结果
python scripts/analyze_bundle_balance.py outputs/bundles/DH/bundles.parquet 120

# 3. 如果 CV > 20%，使用 Multi-BFS
python cli.py bundle --session DH --target_addrs 120 --method multi_bfs

# 4. 再次检查
python scripts/analyze_bundle_balance.py outputs/bundles/DH/bundles.parquet 120
```

---

#### 实践 2: 参数调优

**调优目标**: 在平衡性和速度之间找到最佳点

**步骤**:
1. 从默认参数开始
2. 如果 CV > 15%，减少 tolerance（更严格平衡）
3. 如果计算时间 > 30秒，增加 tolerance 或改用贪婪

**配置调整**:
```yaml
bundle:
  balance_tolerance: 0.12  # 从默认 0.15 降到 0.12（更严格）
  max_rebalance_iterations: 15  # 从 10 增加到 15
```

---

## 11. 联系和支持

### 11.1 问题报告

**GitHub Issues**: [项目仓库 URL]/issues

**报告模板**:
```markdown
**问题类型**: [Bug / 功能请求 / 性能问题]

**算法**: [greedy / multi_bfs]

**环境**:
- OS: [macOS / Linux / Windows]
- Python: [版本]
- 路段数量: [数量]

**复现步骤**:
1. ...
2. ...

**预期行为**:
...

**实际行为**:
...

**日志 / 截图**:
...
```

---

### 11.2 贡献指南

欢迎贡献！请参考 `CONTRIBUTING.md`

**贡献类型**:
- 🐛 Bug 修复
- ✨ 新功能
- 📝 文档改进
- ⚡ 性能优化
- ✅ 测试增强

---

## 12. 变更日志

### Version 2.0.0 (计划中 - 2025-12-20)

**新增**:
- ✨ Multi-source BFS 平衡算法
- ✨ 空间分散种子选择
- ✨ 回溯平衡优化
- ✨ `--method` CLI 参数
- 📊 Bundle 对比分析脚本

**改进**:
- ⚡ 向后兼容性保证
- 📝 详细的技术文档
- ✅ 完整的单元测试

**修复**:
- 无（新功能发布）

---

### Version 1.0.0 (当前版本 - 2025-12-18)

**功能**:
- ✅ 贪婪 BFS Bundle 算法
- ✅ 连通分量识别
- ✅ 小束合并
- ✅ 端点连续性保证

---

## 附录

### A. 术语表

| 术语 | 定义 |
|-----|------|
| Bundle | 一组地理上连接的路段，分配给一个调查员 |
| CV (Coefficient of Variation) | 变异系数 = std / mean，衡量分散程度 |
| BFS | 广度优先搜索 |
| Multi-source | 多个起点同时开始搜索 |
| Backtracking | 回溯，撤销部分决策并重新分配 |
| Seed | 种子，Bundle 生长的起点 |

### B. 参考资源

- [NetworkX Documentation](https://networkx.org/)
- [Graph Partitioning Survey](https://arxiv.org/abs/1311.2337)
- [METIS Manual](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview)

---

**文档版本**: 1.0
**最后更新**: 2025-12-18
**维护者**: Claude Code + User

---

**📝 下次更新**:
- [ ] Phase 2 实施后更新实施计划
- [ ] 测试结果记录
- [ ] 性能基准数据
- [ ] 用户反馈整合
