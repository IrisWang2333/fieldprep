# Linda Vista Pilot Area Exclusion

## 更新日期
2025-12-25

## 更新原因
Linda Vista 是之前的试点区域（pilot area），必须从当前研究中排除，以避免：
- 历史效应（previous treatment exposure）
- 数据污染（contamination）
- 选择性偏差（selection bias）

## 配置更改

### 文件位置
`config/params.yaml`

### 具体更改
```yaml
# 之前
cpd:
  exclude_list: ["Military Facilities"]

# 现在
cpd:
  exclude_list: ["Military Facilities", "Linda Vista"]  # Linda Vista = pilot area, must exclude
```

## Linda Vista CPD 信息

- **CPD Code**: 12
- **面积**: 11.05 km² (2,732 acres)
- **覆盖街段**: 689 segments (占 City 总数的 2.1%)

## 影响分析

### 对已有数据的影响

如果之前已经运行过 sweep/bundle/plan/emit，需要**重新运行**整个流程：

```bash
# Step 1: 重新运行 sweep（排除 Linda Vista）
python cli.py sweep --buffers 40 --mins 2 --tag locked

# Step 2: 重新生成 bundles
python tests/quick_comparison_hard_constraint.py
python tests/quick_filter_bundles.py

# Step 3: 重新运行 plan（如果需要）
python cli.py plan --date 2025-12-24 \
  --interviewer A B C D \
  --task D2DS DH \
  --bundle-file outputs/bundles/DH/bundles_multibfs_regroup_filtered.parquet

# Step 4: 重新运行 emit（如果需要）
python cli.py emit --date 2025-12-24 --session DH
```

### 预期结果变化

| 指标 | 之前 (未排除) | 预期 (排除后) |
|-----|-------------|--------------|
| **Eligible Segments** | 20,874 | ~20,185 (-689) |
| **SFH Addresses** | 215,424 | 会减少 |
| **Bundles** | 2,928 | 会减少 |

**注意**：实际数字取决于 Linda Vista 内有多少 SFH-eligible segments。

## 验证方法

运行 sweep 后检查输出：

```bash
# 查看 sweep 输出报告
cat outputs/sweep/locked/summary.csv

# 检查地图中是否有 Linda Vista 区域的路段
# 打开: outputs/sweep/locked/eligible_b40_m2_map.html
# Linda Vista 应该显示为红色（excluded）
```

## CPD 过滤逻辑

在 `src/sd311_fieldprep/utils.py` 中的 `apply_spatial_filters()` 函数：

```python
# 排除指定的 CPD
exclude_names = filter_config.get("cpd", {}).get("exclude", [])
if exclude_names:
    # 找到地址所在的 CPD
    cpd_mask = addrs['cpd_name'].isin(exclude_names)
    # 排除这些地址
    addrs = addrs[~cpd_mask]
```

## 其他被排除的区域

目前 `exclude_list` 包含：
1. **Military Facilities** - 军事设施（无法访问）
2. **Linda Vista** - 试点区域（避免数据污染）

## 相关文档

- Sweep 配置: `src/sd311_fieldprep/sweep.py`
- 过滤逻辑: `src/sd311_fieldprep/utils.py` (apply_spatial_filters)
- 参数配置: `config/params.yaml`

## 检查清单

- [x] 更新 `config/params.yaml` 添加 "Linda Vista" 到 exclude_list
- [ ] 重新运行 sweep
- [ ] 验证 Linda Vista 区域被正确排除
- [ ] 重新生成 bundles（如果需要）
- [ ] 更新人口统计表（demographics_table.py）以反映新的 eligible segments

---

**重要提醒**：
在重新运行 sweep 之前，当前的所有下游分析（bundles, demographics table 等）仍然**包含** Linda Vista 数据。必须重新运行整个流程才能完全排除 Linda Vista。

