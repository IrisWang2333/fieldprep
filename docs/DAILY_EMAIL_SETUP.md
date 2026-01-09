# Daily Assignment Emails - Automated Setup

This document explains how to set up automated daily email sending via GitHub Actions.

## Overview

The system automatically sends assignment emails to interviewers **every Saturday at 6:00 AM Pacific Time** from **January 10, 2026** to **August 8, 2026**.

## How it Works

1. **Scheduled Trigger**: GitHub Actions runs every Saturday at 6:00 AM PT (14:00 UTC)
2. **Date Range**: Only sends emails between Jan 10, 2026 and Aug 8, 2026
3. **Data Source**: Reads from Google Sheet "Daily Assignments" tab for today's date
4. **Email Content**:
   - DH and D2DS starting addresses from `outputs/incoming/daily/{date}/starts.csv`
   - San Diego weather forecast
   - Map URLs for each interviewer (A-F)
5. **Recipients**: All 6 interviewers listed in the sheet
6. **CC**: Supervisors (Carlie, Yifei) are automatically CC'd

## Required GitHub Secrets

You need to add **2 secrets** to your GitHub repository:

### 1. GMAIL_OAUTH_CREDENTIALS

Go to: Settings → Secrets and variables → Actions → New repository secret

**Name**: `GMAIL_OAUTH_CREDENTIALS`

**Value**: (paste the exact content below)
```json
{"installed":{"client_id":"164711318731-hhb4k0hpvlp70slp6u72i83m4j0fve5m.apps.googleusercontent.com","project_id":"sandiego311","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs","client_secret":"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","redirect_uris":["http://localhost"]}}
```

### 2. GMAIL_OAUTH_TOKEN

Go to: Settings → Secrets and variables → Actions → New repository secret

**Name**: `GMAIL_OAUTH_TOKEN`

**Value**: (paste the exact content below)
```json
{"token": "ya29.a0AUMWg_Ln_8tiEmqJ0DVluQTvzpBZcaZsdygzY3eftZfm2VrLUkl9NkrQbXejjKy5qgZXW5atxw8svsD3QH2xncgnssPaIsxL9IcxgqYTn1mjvEUIBxZMwQmk0RdBlfyQnezxyIxLLAqKCjKykz-zcbJCcQplGvfg2vjtjE3yelQAPKU7q-KJzuXrWsIQQVMbbaiWQnEaCgYKAU8SAQ4SFQHGX2Micmtw6ByhfRyCOG6xIZanOw0206", "refresh_token": "1//01Ly9eFIDK2k3CgYIARAAGAESNwF-L9IrolV5e6mTTDdq4sPtrc8OWoYGqRW_Ry8hNTxNal3uqM4wIEE4rUhAJVUFN5aSnyeMFv0", "token_uri": "https://oauth2.googleapis.com/token", "client_id": "164711318731-hhb4k0hpvlp70slp6u72i83m4j0fve5m.apps.googleusercontent.com", "client_secret": "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", "scopes": ["https://www.googleapis.com/auth/spreadsheets.readonly", "https://www.googleapis.com/auth/gmail.send"], "universe_domain": "googleapis.com", "account": "", "expiry": "2026-01-09T09:38:58Z"}
```

## Manual Triggering (for Testing)

You can manually trigger the workflow from GitHub Actions:

1. Go to: Actions → Daily Assignment Emails → Run workflow
2. Options:
   - **Date**: Leave empty for today, or specify YYYY-MM-DD (e.g., 2026-01-10)
   - **Dry run**: Check to preview emails without sending

## Schedule Details

- **Cron**: `0 14 * * 6` (UTC time, 6 = Saturday)
- **Local Time**: 6:00 AM Pacific (during standard time)
- **Frequency**: Every Saturday
- **Date Range**: January 10, 2026 - August 8, 2026
- **Note**: After August 8, 2026, the workflow will continue to run but skip sending emails

## Google Sheet Structure

### Tab 1: "Daily Assignments"
Must have columns: `Date`, `A`, `B`, `C`, `D`, `E`, `F`

Example:
| Date       | A          | B        | C      | D        | E      | F          |
|------------|------------|----------|--------|----------|--------|------------|
| 2026-01-10 | Jessica D. | Veronica | David  | Jane D.  | Carlie | Vickie     |
| 2026-01-17 | David      | Carlie   | Damien | Veronica | Rene   | Jessica D. |

### Tab 2: "interviewers"
Must have columns: `Name`, `Email`

Example:
| Name       | Email                       |
|------------|-----------------------------|
| Jessica D. | 619loveyourlife@gmail.com   |
| Veronica   | vrnc.sanchez1@gmail.com     |
| David      | dvalentini65@gmail.com      |
| Damien     | dkcobian@gmail.com          |
| Carlie     | creed@luthresearch.com      |
| Rene       | renebenjamin2112@gmail.com  |
| yifei      | ywang809@usc.edu            |

## Troubleshooting

### Emails not sending
1. Check workflow run logs: Actions → Daily Assignment Emails → [latest run]
2. Verify secrets are set correctly
3. Check if date exists in "Daily Assignments" sheet
4. Verify `starts.csv` exists in `outputs/incoming/daily/{date}/`

### OAuth token expired
If emails fail with authentication errors:
1. Run `python send_assignments.py --dry-run` locally
2. Complete OAuth flow in browser
3. Update `GMAIL_OAUTH_TOKEN` secret with new `token.json` content

### Wrong time zone
The workflow uses UTC time. If emails send at wrong time:
- Standard time: 6 AM PT = 14:00 UTC ✓
- Daylight time: 6 AM PDT = 13:00 UTC
- Update cron to `0 13 * * *` during daylight saving time

## Sender Account

Emails are sent from: **dschonho@usc.edu**

To change sender:
1. Update `DEFAULT_SENDER` in `send_assignments.py`
2. Re-authorize OAuth with new account
3. Update GitHub secrets

## Security Notes

⚠️ **Important**:
- OAuth credentials contain sensitive access tokens
- Never commit `client_secret.json` or `token.json` to Git
- These files are in `.gitignore`
- Only store them in GitHub Secrets

## Monitoring

To monitor email sending:
1. Check GitHub Actions logs daily
2. Verify interviewers receive emails
3. Monitor for authentication failures
4. Review weather forecast accuracy
