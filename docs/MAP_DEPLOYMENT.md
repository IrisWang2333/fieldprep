# Map Deployment to Cloudflare Pages

## Overview

Weekly assignment maps are automatically generated as HTML files and can be deployed to Cloudflare Pages at `https://maps-deployment.pages.dev/`.

## Setup (One-time)

### 1. Create Cloudflare Pages Project

1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com/)
2. Navigate to **Workers & Pages** → **Create application** → **Pages**
3. Create a new project named `maps-deployment`
4. You can skip the Git connection (we'll deploy via GitHub Actions)

### 2. Get Cloudflare Credentials

**API Token:**
1. Go to [Cloudflare API Tokens](https://dash.cloudflare.com/profile/api-tokens)
2. Click **Create Token** → **Edit Cloudflare Workers** (use template)
3. Or create custom token with:
   - **Account** - Cloudflare Pages:Edit
   - **Zone** - (not needed for Pages)
4. Copy the token

**Account ID:**
1. Go to Cloudflare Dashboard → **Workers & Pages**
2. Click on your Pages project
3. Copy the Account ID from the URL or project settings

### 3. Configure GitHub Secrets

Add these secrets to your GitHub repository (Settings → Secrets and variables → Actions):

- `CLOUDFLARE_API_TOKEN`: Your Cloudflare API token
- `CLOUDFLARE_ACCOUNT_ID`: Your Cloudflare account ID

## Usage

### Deploy a Map

1. Go to **GitHub Actions** → **Deploy Map to Cloudflare Pages**
2. Click **Run workflow**
3. Enter the date (e.g., `2026-01-17`)
4. Click **Run workflow**

The map will be deployed to:
- **Main URL**: `https://maps-deployment.pages.dev/`
- **Date-specific**: `https://maps-deployment.pages.dev/` (always shows latest)

### Generate Map Locally

Maps are automatically generated when you run:

```bash
python scripts/weekly_plan_emit.py --date 2026-01-17
```

The map will be saved to:
```
outputs/plans/bundles_plan_2026-01-17.html
```

## How It Works

1. **Map Generation**: `plan.py` creates interactive Folium/Leaflet maps showing:
   - Bundle assignments colored by interviewer
   - Starting points for each assignment
   - Geographic layout of the field work

2. **Deployment**: GitHub Actions workflow:
   - Checks if map file exists
   - Copies map to deployment directory as `index.html`
   - Deploys to Cloudflare Pages using official action
   - Map becomes live at `https://maps-deployment.pages.dev/`

## Troubleshooting

### Workflow fails with "Map file not found"

**Cause**: The map hasn't been generated yet for that date.

**Solution**: First run the plan generation:
```bash
gh workflow run weekly-plan-emit.yml -f date=2026-01-17
```

Then deploy the map after it's generated.

### Cloudflare deployment fails

**Cause**: Missing or incorrect secrets.

**Solution**:
1. Verify `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` are set
2. Check that the API token has correct permissions
3. Ensure the Pages project name is `maps-deployment`

### Map shows old data

**Cause**: Browser caching.

**Solution**: Hard refresh (Ctrl+Shift+R or Cmd+Shift+R)

## Future Enhancements

- [ ] Archive multiple dates on the same deployment
- [ ] Create an index page listing all available maps
- [ ] Automatic deployment after weekly plan generation
- [ ] Map versioning and rollback capability
