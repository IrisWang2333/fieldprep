# D2DS可行性分析 - 计算方法详解

## 数据结构说明

### 输入数据
1. **panel** (segment-week level, 2021-2025)
   - `segment_id`: 路段ID
   - `week_start`: 周开始日期（Saturday）
   - `Y_it`: 该segment-week是否有坑洞 (0/1)
   - `R_it`: 该segment-week的坑洞是否在周五前全部修复 (0/1/NaN)

2. **bundles** (segment level)
   - `bundle_id`: bundle ID
   - `segment_id`: 路段ID
   - DH bundles: `bundle_id >= 5000`

3. **potholes** (pothole level, 2021-2025)
   - `segment_id`: 路段ID
   - `date_requested`: 坑洞报告日期
   - `date_closed`: 坑洞修复日期
   - `week_start`: 该坑洞所在的周（Saturday）

---

## 分析1: Bundle Eligibility（前一周有坑洞的bundles数量）

### 问题
每周有多少bundles满足条件：该bundle的至少一个segment在**前一周**有坑洞？

### 计算步骤

```python
# Step 1: 为每个segment添加bundle_id
panel['bundle_id'] = panel['segment_id'].map(segment_to_bundle_dict)

# Step 2: 找出每周有坑洞的bundles
bundles_with_potholes_by_week = (
    panel[panel['Y_it'] == 1]  # 只看有坑洞的segment-weeks
    .groupby(['week_start', 'bundle_id'])
    .size()  # 计算每个bundle-week的坑洞数
    .reset_index(name='pothole_count')
)

# Step 3: 对于每周 t，找出在 t-1 周有坑洞的bundles
for i, week in enumerate(all_weeks):
    if i == 0:
        continue  # 第一周没有前一周

    preceding_week = all_weeks[i-1]

    # 在 t-1 周有坑洞的bundles = 在第 t 周符合条件的bundles
    eligible_bundles = set(
        bundles_with_potholes_by_week[
            bundles_with_potholes_by_week['week_start'] == preceding_week
        ]['bundle_id'].unique()
    )
```

### 示例
```
Week 1 (2021-01-02): Bundle A, B, C 有坑洞
Week 2 (2021-01-09): Bundle A, B, C 符合条件（因为它们在Week 1有坑洞）
                     实际在Week 2有坑洞的是 Bundle B, D
Week 3 (2021-01-16): Bundle B, D 符合条件（因为它们在Week 2有坑洞）
```

### 输出指标
- `total_eligible`: 每周符合条件的总bundles数
- `eligible_dh`: 其中DH bundles数（bundle_id >= 5000）
- `eligible_non_dh`: 其中非DH bundles数

---

## 分析2: Unfixed Segments Availability（未修复的segments数量）

### 问题
对于每个有坑洞的bundle-week，有多少segments的坑洞在周五前没有被修复？

### 计算步骤

```python
# Step 1: 只看有坑洞的segment-weeks
panel_with_potholes = panel[panel['Y_it'] == 1]

# Step 2: 对每个bundle-week，统计unfixed segments
for (bundle_id, week), group in panel_with_potholes.groupby(['bundle_id', 'week_start']):
    total_with_potholes = len(group)  # 该bundle-week有多少segments有坑洞
    unfixed = (group['R_it'] == 0).sum()  # 其中多少没修复

    unfixed_by_bundle_week.append({
        'bundle_id': bundle_id,
        'week_start': week,
        'total_potholes': total_with_potholes,
        'unfixed': unfixed,
        'unfixed_rate': unfixed / total_with_potholes
    })
```

### R_it的定义
- `R_it = 1`: 该segment在这周有坑洞，且**所有**坑洞都在周五前修复
- `R_it = 0`: 该segment在这周有坑洞，但**不是所有**坑洞都在周五前修复
- `R_it = NaN`: 该segment在这周没有坑洞（Y_it = 0）

### 示例
```
Bundle A, Week 1:
  - Segment 1: 有2个坑洞，都在周五前修复 → R_it = 1 (fixed)
  - Segment 2: 有1个坑洞，没修复 → R_it = 0 (unfixed)
  - Segment 3: 有1个坑洞，没修复 → R_it = 0 (unfixed)

  Result: total_potholes=3, unfixed=2, unfixed_rate=66.7%
```

### 输出指标
- `unfixed`: 每个bundle-week的unfixed segments数量
- `unfixed_rate`: unfixed比例
- 分布统计：mean, median, 0个unfixed的比例

---

## 分析3: Fixed Before Saturday（在Saturday前修复的坑洞比例）

### 问题
D2DS segments的坑洞中，有多少在**该周的Saturday（调查开始）前**就被修复了？

### 时间定义
```
Week = Saturday to Friday (6 days)
Survey week: Saturday (Day 0) to Friday (Day 6)

如果坑洞在Week 1被报告（比如Monday），然后在Week 1的Saturday前被修复，
则这个坑洞在调查开始前就已经不存在了（contamination）。
```

### 计算步骤

```python
# Step 1: 对每个坑洞，找到它所在周的Saturday
potholes['week_start'] = ...  # Saturday of that week
potholes['week_saturday'] = potholes['week_start']  # Same as week_start

# Step 2: 检查是否在Saturday前修复
potholes['fixed_before_saturday'] = (
    (potholes['date_closed'].notna()) &  # 有修复日期
    (potholes['date_closed'] < potholes['week_saturday'])  # 在Saturday前修复
).astype(int)
```

### 示例
```
Pothole 1:
  - date_requested: 2021-01-04 (Monday)
  - week_start: 2021-01-02 (Saturday)
  - date_closed: 2021-01-01 (Friday, before the week)
  - fixed_before_saturday: 1 ✓ (contamination!)

Pothole 2:
  - date_requested: 2021-01-04 (Monday)
  - week_start: 2021-01-02 (Saturday)
  - date_closed: 2021-01-05 (Tuesday, during the week)
  - fixed_before_saturday: 0 ✗ (no contamination)
```

### 关键发现
- 实际结果: 0.0% 在Saturday前修复
- 这意味着：所有坑洞的修复日期都 >= week_start
- 原因可能是：
  1. 数据记录方式（坑洞报告后才记录修复）
  2. 实际工作流程（坑洞报告后才开始修复）

### 输出指标
- `fixed_before_saturday`: 在Saturday前修复的比例
- `days_to_fix`: 平均修复时间（天）

---

## 分析4: Historical D2DS Segments Exposure（历史坑洞暴露）

### 问题
1. D2DS segments的坑洞发生率是多少？
2. 有坑洞的segments中，修复率是多少？
3. 如果segment在**前一周**有坑洞，这周的状态如何？

### 计算步骤

#### 4.1 基础统计
```python
# 只看D2DS segments的数据
d2ds_panel = panel[panel['segment_id'].isin(d2ds_segments)]

# 坑洞发生率
pothole_rate = d2ds_panel['Y_it'].mean()

# 修复率（在有坑洞的segment-weeks中）
fix_rate = d2ds_panel['R_it'].sum() / d2ds_panel['Y_it'].sum()
```

#### 4.2 Preceding Week Analysis
```python
# Step 1: 按segment和week排序
d2ds_panel_sorted = d2ds_panel.sort_values(['segment_id', 'week_start'])

# Step 2: 为每个segment-week添加前一周的Y_it和R_it
d2ds_panel_sorted['prev_week_pothole'] = (
    d2ds_panel_sorted.groupby('segment_id')['Y_it'].shift(1)
)
d2ds_panel_sorted['prev_week_fixed'] = (
    d2ds_panel_sorted.groupby('segment_id')['R_it'].shift(1)
)

# Step 3: 统计前一周状态
prev_pothole_cases = d2ds_with_prev[d2ds_with_prev['prev_week_pothole'] == 1]
fixed_in_prev_week = prev_pothole_cases['prev_week_fixed'].sum()
not_fixed_in_prev_week = (prev_pothole_cases['prev_week_fixed'] == 0).sum()
```

### shift(1)的工作原理
```
Segment A:
  Week 1: Y_it=1, R_it=0  →  prev_week_pothole=NaN, prev_week_fixed=NaN
  Week 2: Y_it=0, R_it=NaN →  prev_week_pothole=1,   prev_week_fixed=0
  Week 3: Y_it=1, R_it=1   →  prev_week_pothole=0,   prev_week_fixed=NaN
  Week 4: Y_it=0, R_it=NaN →  prev_week_pothole=1,   prev_week_fixed=1
```

### 三种状态
1. **No pothole in prev week**: prev_week_pothole = 0
   - 前一周没有坑洞

2. **Pothole & fixed**: prev_week_pothole = 1, prev_week_fixed = 1
   - 前一周有坑洞，且在周五前修复

3. **Pothole & NOT fixed**: prev_week_pothole = 1, prev_week_fixed = 0
   - 前一周有坑洞，但没在周五前修复
   - 这是**control group**的来源

### 输出指标
- 坑洞发生率：1.19%
- 修复率：34.2%（在有坑洞的segment-weeks中）
- 前一周有坑洞的比例：1.19%
- 前一周有坑洞且修复：34.27%
- 前一周有坑洞但未修复：65.73% ← **这是treatment variation的来源**

---

## 关键假设和限制

### 假设
1. **Week定义**: Saturday-Friday（6天周期）
2. **R_it定义**: 所有坑洞都在周五前修复才算R_it=1
3. **DH bundles**: bundle_id >= 5000
4. **历史数据**: 2021-2025年数据代表未来模式

### 限制
1. **Unfixed availability**:
   - 计算的是单个bundle-week的unfixed数
   - 未考虑跨bundle采样（实际可以从多个bundles中选择）

2. **Contamination**:
   - 只检查了Saturday前修复
   - 未考虑其他可能的contamination sources

3. **Preceding week条件**:
   - 严格要求前一周有坑洞
   - 可能过于保守（可以考虑放宽到"前两周"）

---

## 数据流程总结

```
原始数据 (potholes.csv)
    ↓
建立panel (segment-week level, 2021-2025)
    ↓
计算Y_it和R_it
    ↓
添加bundle_id
    ↓
分析1: 统计每周符合条件的bundles (based on t-1 week)
分析2: 统计unfixed segments (R_it = 0)
分析3: 检查fix timing (date_closed < week_start)
分析4: D2DS segments的历史暴露率 (shift(1) for prev week)
```

所有计算都基于历史真实数据（2021-2025），没有使用模拟数据。
