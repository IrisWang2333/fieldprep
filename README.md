# SD311 Field Prep - Pothole Treatment Experiment

Automated system for generating weekly field plans for pothole treatment experiments in San Diego.

## Overview

This system implements a **single-layer randomization design** for door-to-door field experiments:

- **Week 1**: 30 DH bundles (4 conditional + 26 random), no D2DS
- **Week 2+**: 6 DH bundles (4 conditional + 2 random) + 6 D2DS bundles (4 from DH conditional + 2 random)

**Key Features:**
- ✅ Automatic data updates from City of San Diego notification activities
- ✅ Conditional sampling based on pothole reports
- ✅ Weekly automated plan generation (GitHub Actions)
- ✅ Automatic upload to Google Drive

## Project Structure

```
fieldprep/
├── .github/workflows/        # GitHub Actions automation
│   └── weekly-plan-emit.yml  # Weekly plan & emit workflow
├── src/sd311_fieldprep/      # Main source code
│   ├── plan.py              # Plan generation (bundle sampling)
│   ├── emit.py              # Field file generation
│   └── simulation/          # Simulation tools
│       └── multiday.py      # Multi-week simulation
├── utils/                    # Utility modules
│   ├── data_fetcher.py      # Automatic data loading
│   └── sampling.py          # Conditional sampling logic
├── scripts/                  # Automation scripts
│   └── weekly_plan_emit.py  # Weekly automation wrapper
├── tests/                    # Tests and model building
│   └── build_pothole_model.py
├── data/                     # Data files (gitignored)
├── outputs/                  # Generated outputs
│   ├── bundles/             # Bundle definitions
│   ├── simulation/          # Simulation results
│   └── incoming/daily/      # Daily field files
└── requirements.txt          # Python dependencies
```

## Setup

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/fieldprep.git
cd fieldprep
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure GitHub Secrets

For automated Google Drive uploads, configure these secrets in your GitHub repository:

- `GOOGLE_DRIVE_CREDENTIALS`: Google Cloud service account JSON credentials
- `GOOGLE_DRIVE_BUCKET`: Google Cloud Storage bucket name (e.g., `gs://your-bucket-name`)

#### Setting up Google Cloud Storage:

1. Create a Google Cloud project
2. Enable Cloud Storage API
3. Create a service account with Storage Admin role
4. Download service account JSON key
5. Create a Cloud Storage bucket
6. Add the JSON key as GitHub secret `GOOGLE_DRIVE_CREDENTIALS`
7. Add bucket name as GitHub secret `GOOGLE_DRIVE_BUCKET`

## Usage

### Automated Weekly Run (GitHub Actions)

The system automatically runs **every Friday at 10:00 PM Pacific Time**:

1. Downloads latest pothole data
2. Calculates next Saturday's date
3. Generates plan for next week
4. Creates field files (routes, addresses, etc.)
5. Uploads to Google Drive

**Manual trigger:**
- Go to GitHub Actions → "Weekly Plan and Emit" → "Run workflow"

### Manual Local Run

Generate plan for next Saturday:

```bash
python scripts/weekly_plan_emit.py
```

Generate plan for specific date:

```bash
python scripts/weekly_plan_emit.py --date 2025-01-11
```

Week 1 (special sampling):

```bash
python scripts/weekly_plan_emit.py --date 2025-01-04 --week-1
```

### Running Simulation

Test the design over 30 weeks:

```bash
cd /path/to/fieldprep
export PYTHONPATH="/path/to/fieldprep/src:$PYTHONPATH"
python src/sd311_fieldprep/simulation/multiday.py
```

## Experimental Design

### Data Source

- **URL**: `https://seshat.datasd.org/td_optimizations/notification_activities.csv`
- **Filter**: `ACTIVITY_CODE_GROUP_TEXT = "ASPHALT"` AND `ACTIVITY_CODE_TEXT = "POTHOLE PATCHED (EA)"`
- **Update**: Automatically downloaded daily

### Week Definition

- **Week structure**: Saturday (day 0) → Friday (day 6)
- **Example**: Week 1 = 2025-01-04 (Sat) to 2025-01-10 (Fri)

### Bundle Eligibility

A bundle is **eligible** if at least one segment in the bundle had a pothole reported in the **preceding week**.

**Example:**
- Today: 2025-01-11 (Saturday, Week 2)
- Preceding week: 2025-01-04 to 2025-01-10 (Week 1)
- Eligible bundles: Those with ≥1 pothole in Week 1

### Sampling Logic

#### Week 1 (First week):
- **30 DH bundles**:
  - 4 conditional (from eligible bundles)
  - 26 random (from all bundles)
- **0 D2DS bundles**

#### Week 2+ (Subsequent weeks):
- **6 DH bundles**:
  - 4 conditional (from eligible bundles)
  - 2 random (from all bundles)
- **6 D2DS bundles**:
  - 4 from DH conditional (reused)
  - 2 random (new bundles)

### Segment-Level DH Treatment Assignment

For each segment within DH bundles, the system assigns a treatment intensity arm:

- **Full Treatment** (25% probability): All addresses in segment receive door hangers (treated_share = 1.0)
- **Partial Treatment** (50% probability): Half of addresses in segment receive door hangers (treated_share = 0.5)
- **Control** (25% probability): No addresses in segment receive door hangers (treated_share = 0.0)

Assignment is randomized independently for each segment using the specified seed.

**Output**: Segment assignments are saved to `outputs/plans/segment_assignments_YYYY-MM-DD.csv` with columns:
- `date`: Field date
- `bundle_id`: Bundle identifier
- `segment_id`: Street segment identifier
- `dh_arm`: Treatment arm (Full, Partial, Control)
- `treated_share`: Share of addresses to treat (1.0, 0.5, 0.0)

### Without Replacement

Each bundle is used at most once throughout the entire experiment.

## Output Files

### Plan Files

Generated in `outputs/plans/`:

- `bundles_plan_YYYY-MM-DD.csv`: Bundle assignments to interviewers
- `segment_assignments_YYYY-MM-DD.csv`: Segment-level DH treatment assignments (Full/Partial/Control)
- `bundles_plan_YYYY-MM-DD.html`: Map visualization of selected bundles

### Field Files

Generated in `outputs/incoming/daily/YYYY-MM-DD/`:

- `routes.csv`: Ordered routes for each interviewer
- `segments.geojson`: Map of all segments
- `sfh_points.csv`: Addresses to visit
- `starts.csv`: Starting points for each interviewer-task

## Key Modules

### `utils/data_fetcher.py`

Handles automatic data loading:

```python
from utils.data_fetcher import fetch_latest_notification_activities

# Download latest data
activities = fetch_latest_notification_activities(
    use_local=False,
    download_if_missing=True
)
```

### `utils/sampling.py`

Conditional sampling logic:

```python
from utils.sampling import sample_dh_bundles, select_d2ds_bundles

# Sample DH bundles
dh_sample = sample_dh_bundles(
    current_date=date,
    eligible_bundles=eligible,
    all_bundles=all_bundles,
    is_day_1=False
)

# Sample D2DS bundles (Week 2+)
d2ds_selection = select_d2ds_bundles(
    conditional_bundles=dh_sample['conditional'],
    all_bundles=all_bundles,
    n_from_conditional=4,
    n_random=2
)
```

### `plan.py`

Main plan generation:

```python
from sd311_fieldprep.plan import run_plan

plan_csv = run_plan(
    date='2025-01-11',
    interviewers=('A', 'B', 'C', 'D', 'E', 'F'),
    tasks=('DH', 'D2DS'),
    is_week_1=False
)
```

### `emit.py`

Generate field files:

```python
from sd311_fieldprep.emit import run_emit

run_emit(
    date='2025-01-11',
    plan_csv='outputs/plans/plan_2025-01-11.csv',
    bundle_file='outputs/bundles/DH/bundles.parquet'
)
```

## Troubleshooting

### Workflow not running

- Check cron schedule (UTC vs Pacific Time)
- Verify GitHub Actions is enabled
- Check for workflow errors in Actions tab

### Google Drive upload failing

- Verify `GOOGLE_DRIVE_CREDENTIALS` secret is set correctly
- Check service account has Storage Admin role
- Verify bucket name in `GOOGLE_DRIVE_BUCKET` secret

### Not enough eligible bundles

- Check data source is accessible
- Verify date range (preceding week calculation)
- Run simulation to check eligibility distribution

## License

[Add your license here]

## Contact

[Add contact information]
