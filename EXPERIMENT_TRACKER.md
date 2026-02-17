# SD311 Field Experiment Tracker

## Quick Links

| 资源 | 链接 |
|------|------|
| **GitHub 仓库** | https://github.com/IrisWang2333/fieldprep |
| **GitHub Actions (workflow日志)** | https://github.com/IrisWang2333/fieldprep/actions |
| **Google Drive (每周field files)** | https://drive.google.com/drive/u/2/folders/17Eexa-x7fOIB0gOu63SWUkZlNSr5oyk8 |
| **DataSD pothole数据** | https://seshat.datasd.org/td_optimizations/notification_activities.csv |

---

## 实验时间线

| 阶段 | 日期范围 | 备注 |
|------|----------|------|
| Pilot | 2025-12-27 | Week 1：24 DH bundles，无D2DS |
| Official Week 1 | 2026-01-10 | 24 DH bundles，无D2DS |
| Official Week 2+ | 2026-01-17 起 | 6 DH + 6 D2DS per week |
| 实验结束 | 2026-08-15 | |
| 排除日期 | 2026-05-23, 2026-07-04 | 假期/休息 |

---

## 实验设计

### DH (Door-to-Door) Sampling

**每周6个DH bundles = 4 conditional + 2 random**

| 类型 | bundle_type值 | 选择逻辑 |
|------|--------------|---------|
| Conditional | `conditional` | 前一周（周六到周五）有pothole的bundles |
| Random | `random` | 从所有未用过的bundles中随机选 |

**Fallback（conditional不够4个时）**：
- 把deficit加到random pool，确保总数仍为6个

**Without-replacement**：严格执行，DH bundles不重复使用（跨周）

### D2DS (Door-to-Door Second Stage) Sampling

**每周6个D2DS bundles = 4 d2ds_conditional + 2 d2ds_random**

| 类型 | bundle_type值 | 选择逻辑 |
|------|--------------|---------|
| D2DS Conditional | `d2ds_conditional` | 来自上周的DH bundles（优先DH conditional，不够则用DH random补） |
| D2DS Random | `d2ds_random` | 优先选前一周有pothole的bundles，不够则从所有未用过的bundles补 |

**Fallback层级（d2ds_conditional不够4个时）**：
1. 先取上周所有DH conditional bundles
2. 不够 → 从上周ALL DH bundles（包括random）补充
3. 仍不够 → 把deficit加到random pool

**Fallback层级（d2ds_random不够2个时）**（commit `a03739f`修复）：
1. 优先从eligible bundles（有pothole的）中选
2. 不够 → 取所有eligible，从所有未用过的bundles中补充差额

**Without-replacement例外**：上周DH conditional bundles豁免，允许D2DS复用

### Segment-Level DH Arm Assignment

DH bundles内每个segment随机分配treatment arm：

| Arm | dh_arm值 | treated_share | 概率 |
|-----|---------|--------------|------|
| Full | `Full` | 1.0 | 25% |
| Partial | `Partial` | 0.5 | 25% |
| Control | `Control` | 0.0 | 50% |

---

## 关键文件

### GitHub上的文件（每周自动生成）

```
outputs/plans/
├── bundles_plan_YYYY-MM-DD.csv          # 每周bundle分配（主要计划文件）
├── bundle_metadata_YYYY-MM-DD.csv       # bundle类型和pothole详情
└── segment_assignments_YYYY-MM-DD.csv   # segment级别的DH arm分配

data/historical_notification_activities/
└── notification_activities_YYYY-MM-DD.csv  # 每周pothole数据快照
```

### bundles_plan 格式

| 列名 | 说明 |
|------|------|
| `date` | 调查日期 |
| `interviewer` | 访谈员代码 (A-F) |
| `task` | `DH` 或 `D2DS` |
| `bundle_id` | Bundle编号 |
| `bundle_type` | `conditional` / `random` / `d2ds_conditional` / `d2ds_random` |
| `list_code` | 列表代码（默认30） |
| `sfh_bundle_total` | Bundle内SFH地址总数 |

**设计验证**：每个interviewer应有1个DH + 1个D2DS（Week 2+）

### bundle_metadata 格式

| 列名 | 说明 |
|------|------|
| `date` | 调查日期 |
| `task` | `DH` 或 `D2DS` |
| `bundle_id` | Bundle编号 |
| `bundle_type` | 类型标记 |
| `num_segments` | Bundle内segment数量 |
| `num_potholes_in_preceding_week` | 前一周pothole数量（用notification_activities数据） |
| `segments_with_potholes` | 有pothole的segment ID（逗号分隔） |

**验证规则**：所有`conditional`类型的DH bundles必须`num_potholes_in_preceding_week > 0`

### segment_assignments 格式

| 列名 | 说明 |
|------|------|
| `date` | 调查日期 |
| `bundle_id` | 所属Bundle |
| `segment_id` | Segment ID（格式：SS-XXXXXX-PV1） |
| `dh_arm` | `Full` / `Partial` / `Control` |
| `treated_share` | `1.0` / `0.5` / `0.0` |

**注意**：只包含DH bundles的segments，不含D2DS

### 本地关键数据文件

| 文件 | 路径 | 说明 |
|------|------|------|
| Bundle定义 | `outputs/bundles/DH/bundles_multibfs_regroup_filtered_length_3.parquet` | 1626个bundles，长度≤3km |
| 地址-Segment映射 | `outputs/sweep/locked/segment_addresses_b40_m2.parquet` | 40m范围内，最少2个地址 |
| Pothole数据（SAP） | `data/notification_activities.csv` | 自动从DataSD下载 |
| Pothole数据（DataSD公开） | `/Users/iris/Downloads/get_it_done_pothole_requests_datasd.csv` | Get It Done系统 |

---

## Pothole数据来源对比

| 数据源 | 用途 | 有pothole定义 | Fixed定义 |
|--------|------|--------------|-----------|
| **notification_activities** | Bundle选择（GitHub自动运行用） | `NOTIFICATION_DATE`在窗口内 | `ACTIVITY_CODE_TEXT == 'FIXED'` |
| **DataSD Get It Done** | 分析/核查用 | `date_requested`在窗口内 | `status == 'Closed'` |

**重要**：两个数据源的覆盖范围不同，DataSD覆盖更广（公开投诉）

**已知日期计算注意事项**：DataSD的`date_requested`有时间戳，`date_closed`只有日期（截断至00:00:00），计算days_to_fix时需将`date_requested`也截断至日期（`.dt.normalize()`），否则会出现负值。

---

## 每周Workflow流程

**触发时间**：每周六早上6:00 UTC（即周五晚10pm PST）

**步骤**：
1. 下载最新pothole数据（notification_activities）
2. 检查历史plan文件（without-replacement tracking）
3. 生成bundle计划（`plan.py`）
4. 生成field files（`emit.py`）
5. 提交bundles_plan、bundle_metadata、segment_assignments、notification_activities到GitHub
6. 上传field files到Google Drive

**手动触发**：
```
GitHub Actions → Weekly Plan and Emit → Run workflow
参数：date (YYYY-MM-DD), is_week_1 (true/false)
```

---

## 核心代码文件

| 文件 | 功能 |
|------|------|
| `src/sd311_fieldprep/plan.py` | Bundle抽样主逻辑，eligibility判断 |
| `src/sd311_fieldprep/emit.py` | 生成field files（路线、地址列表等） |
| `utils/sampling.py` | `sample_dh_bundles()` 和 `select_d2ds_bundles()` |
| `utils/bundle_tracker.py` | Without-replacement tracking |
| `utils/data_fetcher.py` | 下载notification_activities |
| `.github/workflows/weekly-plan-emit.yml` | GitHub Actions workflow |

---

## 分析脚本（本地）

| 脚本 | 路径 | 功能 |
|------|------|------|
| `map_survey_to_bundles.py` | `fieldprep/scripts/` | 将survey地址映射到segment和bundle |
| `analyze_segment_potholes.py` | `fieldprep/scripts/` | 用notification_activities分析segment pothole |
| `analyze_segment_potholes_datasd.py` | `fieldprep/scripts/` | 用DataSD数据分析segment pothole |

**分析脚本输入**：
- Survey数据：`/Users/iris/Downloads/250823-3.xlsx`（264条记录，2026-01-17到2026-02-07）
- Survey地址映射：`/Users/iris/Downloads/survey_address_mapping.csv`
- Pothole分析输出：`/Users/iris/Downloads/segment_pothole_analysis_datasd.csv`

---

## 已知问题和修复历史

| 日期 | Commit | 问题 | 修复 |
|------|--------|------|------|
| 2026-02-13 | `a03739f` | D2DS random不够时不补充，导致D2DS只有5个 | `sampling.py`：eligible不够时从all bundles补充差额 |
| 2026-02-13 | `15ebb01` | Partial segment地址分配bug | `emit.py`：修复地址分配逻辑 |
| 2026-02-12 | 早期 | D2DS conditional reuse失败 | 加入exempt_bundles机制 |

---

## 数据检查清单（每周）

运行以下检查验证当周plan：

```python
import pandas as pd

date = "2026-02-14"  # 替换为当周日期

plan = pd.read_csv(f"outputs/plans/bundles_plan_{date}.csv")
metadata = pd.read_csv(f"outputs/plans/bundle_metadata_{date}.csv")
seg = pd.read_csv(f"outputs/plans/segment_assignments_{date}.csv")

# 1. Interviewer分配是否均衡
for ivw in sorted(plan['interviewer'].unique()):
    p = plan[plan['interviewer'] == ivw]
    assert len(p[p['task'] == 'DH']) == 1, f"{ivw} DH数量不对"
    assert len(p[p['task'] == 'D2DS']) == 1, f"{ivw} D2DS数量不对"

# 2. DH数量
dh = plan[plan['task'] == 'DH']
assert len(dh) == 6
assert len(dh[dh['bundle_type'] == 'conditional']) == 4

# 3. D2DS数量
d2ds = plan[plan['task'] == 'D2DS']
assert len(d2ds) == 6
assert len(d2ds[d2ds['bundle_type'] == 'd2ds_conditional']) == 4

# 4. DH conditional有potholes
dh_cond_meta = metadata[(metadata['task'] == 'DH') & (metadata['bundle_type'] == 'conditional')]
assert all(dh_cond_meta['num_potholes_in_preceding_week'] > 0), "有conditional bundle没有pothole！"

# 5. Segment arm分配
assert set(seg['dh_arm'].unique()) == {'Full', 'Partial', 'Control'}
assert set(seg['treated_share'].unique()) == {0.0, 0.5, 1.0}

print("✓ 所有检查通过")
```
