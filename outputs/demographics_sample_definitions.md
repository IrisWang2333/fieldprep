# 人口统计表 - 三个样本集定义

## 样本集层级关系

```
City of San Diego 所有街段 (32,888 segments)
├── 基础数据源: DataSD/sd_paving_segs_datasd.shp
├── PCI 数据: 已包含在 shapefile 中 (pci23 字段)
└── Census 数据: 通过空间连接获得
    ├── Block Groups (族裔/人口)
    └── Census Tracts (收入/教育)

     ↓ Sweep 筛选 (63.5%)
     
SFH-Eligible Segments (20,874 segments)
├── 数据来源: outputs/sweep/locked/eligible_b40_m2.parquet
├── 地图: outputs/sweep/locked/eligible_b40_m2_map.html
├── 筛选条件:
│   ├── 地址缓冲距离: 40m (addr_buffer_m)
│   ├── 最小SFH数量: ≥2 addresses (sfh_min)
│   ├── Zoning 过滤: 应用了 params.yaml 中的设置
│   └── CPD 过滤: 排除了指定的 CPD 区域
└── 统计:
    ├── SFH 地址总数: 215,424
    ├── 平均每段: 10.3 addresses
    └── 总长度: 约占全市街道的 63.5%

     ↓ Bundle 构建 (82.2% of SFH-eligible)
     
Bundle Segments (17,166 segments)
├── 数据来源: outputs/bundles/DH/bundles_multibfs_regroup_filtered.parquet
├── 地图: outputs/bundles/DH/bundles_multibfs_regroup_filtered_map.html
├── 构建条件:
│   ├── 连通性: 路段必须相邻形成连通网络
│   ├── Bundle 目标大小: 60 addresses
│   ├── Bundle 大小范围: [48, 72] addresses (0.8x - 1.2x)
│   ├── Join 容差: 15m (街段连接判定距离)
│   └── 算法: Multi-BFS with hard constraints
└── 统计:
    ├── Bundle 数量: 2,928 bundles
    ├── SFH 地址总数: 179,112
    ├── 平均每个 bundle: 61.2 addresses
    └── 平均每段: 10.4 addresses
```

## 详细筛选流程

### 1️⃣  City Overall → SFH-Eligible

**命令：**
```bash
python cli.py sweep \
  --buffers 40 \
  --mins 2 \
  --tag locked
```

**筛选逻辑：**

```python
# sweep.py 中的处理流程

# Step 1: 空间过滤 (zoning, CPD)
addrs_filtered = apply_spatial_filters(addrs, zoning, cpd, filter_config)

# Step 2: 可寻址过滤 (需要有门牌号和街道名)
addrs_addressable = addrs_filtered[addressable_mask(addrs_filtered)]

# Step 3: SFH 过滤 (排除公寓等多户住宅)
_compose_address(addrs_addressable)
addrs_sfh = addrs_addressable[addrs_addressable['__unit_blank__']]

# Step 4: 最近邻连接 (40m buffer)
joined = gpd.sjoin_nearest(addrs_sfh, streets, max_distance=40)

# Step 5: 统计并筛选
counts = joined.groupby(segment_id).size()
eligible = streets[counts >= 2]  # 至少 2 个 SFH 地址
```

**结果：**
- 从 32,888 → 20,874 segments (减少 36.5%)
- 排除的路段主要是：
  - 商业区/工业区街道
  - 公园/空地周边街道
  - 多户住宅密集区 (公寓楼)
  - SFH 地址过少的路段

### 2️⃣  SFH-Eligible → Bundle Segments

**命令：**
```bash
# Step 1: 生成 bundles
python tests/quick_comparison_hard_constraint.py

# Step 2: 过滤到范围 [48, 72]
python tests/quick_filter_bundles.py
```

**构建逻辑：**

```python
# bundle_hard_constraint.py 中的处理流程

# Step 1-2: 构建邻接图
adjacent = find_adjacent_segments(segs, join_tol_m=15)

# Step 3: 连通性合并 (优先合并到最小的 bundle)
bundles = merge_connected_components(adjacent)

# Step 4: 标记初始 bundles
assign_bundle_ids(bundles)

# Step 5: 自动拆分 >1.0x target (>60 addresses)
bundles = split_oversized_bundles(bundles, target=60)

# Step 6: 硬约束清理 ≤1.2x (≤72 addresses)
bundles = enforce_hard_max(bundles, hard_max=72)

# Step 7-9: 重组不合格的 bundles 到 [0.8x, 1.2x] = [48, 72]
bundles = regroup_bundles(bundles, min=48, max=72)
```

**过滤条件（quick_filter_bundles.py）：**

```python
# 过滤到目标范围
valid_bundles = bundled[
    (bundled['bundle_addr_total'] >= 48) &  # 0.8x
    (bundled['bundle_addr_total'] <= 72)    # 1.2x
]
```

**结果：**
- 从 20,874 → 17,166 segments (减少 17.8%)
- 排除的路段主要是：
  - 孤立的、无法形成连通 bundle 的路段
  - Bundle 太小 (<48 addresses) 的路段
  - Bundle 太大 (>72 addresses) 且无法拆分的路段

## 三个样本集的地图对应关系

| 样本集 | 地图文件 | 地图内容 |
|-------|---------|---------|
| **City Overall** | ❌ 无专门地图 | 可查看 DataSD 原始数据 |
| **SFH-Eligible** | `outputs/sweep/locked/`<br>`eligible_b40_m2_map.html` | 蓝色：合格路段<br>灰色：不合格路段<br>绿色阴影：包含的 zoning<br>红色阴影：排除的 zoning |
| **Bundle Segments** | `outputs/bundles/DH/`<br>`bundles_multibfs_regroup_filtered_map.html` | 不同颜色：不同的 bundle<br>Tooltip: bundle_id, segment数量, 地址总数 |

## Census 数据连接

所有三个样本集都连接到相同的 Census 数据源：

### Block Groups (族裔/人口)
- **Shapefile**: nhgis0002_shape/nhgis0002_shapefile_tl2023_060_blck_grp_2023.zip
- **CSV**: nhgis_blockgroup_census/nhgis0008_ds258_2020_blck_grp.csv
- **范围**: California → SD County (2,057) → City extent (1,614)
- **变量**:
  - U7P001: Total population
  - U7P002: Hispanic or Latino
  - U7P005: White (Non-Hispanic)
  - U7P006: Black (Non-Hispanic)
  - U7P008: Asian (Non-Hispanic)

### Census Tracts (收入/教育)
- **Shapefile**: nhgis0002_shape/nhgis0002_shapefile_tl2023_us_tract_2023.zip
- **CSV**: nhgis0001_csv_popuraceedu/nhgis0001_ds267_20235_tract.csv
- **范围**: US-wide → SD County (1,385) → City extent (586)
- **变量**:
  - ASRTE001: Per capita income
  - ASP3E001: Population 25+ (total)
  - ASP3E022-025: Bachelor's+ degrees

### 空间连接方法
```python
# 最大面积重叠法
for each street segment:
    find all overlapping census units
    assign to the unit with largest overlap area
```

## 统计对比

| 指标 | City Overall | SFH-Eligible | Bundle | 
|-----|-------------|-------------|--------|
| **街段数量** | 32,888 | 20,874 (63.5%) | 17,166 (52.2%) |
| **SFH 地址** | N/A | 215,424 | 179,112 |
| **每段平均** | N/A | 10.3 | 10.4 |
| **Bundles** | N/A | N/A | 2,928 |
| **每 Bundle** | N/A | N/A | 61.2 addresses |
| | | | |
| **Avg PCI** | 66.2 | 66.8 | 67.2 |
| **Per Capita Income** | $60,611 | $58,836 | $58,578 |
| **College+** | 52.1% | 50.6% | 50.3% |
| **Hispanic** | 26.5% | 26.3% | 26.4% |
| **White (NH)** | 46.9% | 46.2% | 46.0% |
| **Asian (NH)** | 15.3% | 16.3% | 16.3% |
| **Black (NH)** | 4.6% | 4.5% | 4.5% |

## 关键 Insight

1. **递进式筛选**: City → SFH → Bundle 是逐步筛选的过程
2. **人口统计稳定性**: 三个样本的族裔比例非常接近，说明筛选过程没有显著的地理偏向
3. **收入差异**: City Overall 收入更高，因为包含了更多商业区/高收入区的街道
4. **Bundle 代表性**: Bundle segments 很好地代表了 SFH-eligible segments 的特征

