# Deployment Guide — Railway

## Step 1: Push to GitHub

```bash
# In d:\magicpin — run these in order
git remote add origin https://github.com/YOUR_USERNAME/magicpin-vera-bot.git
git branch -M main
git push -u origin main
```

Create the repo at: https://github.com/new
- Name: `magicpin-vera-bot`
- Visibility: **Public**
- No README/gitignore/license (already have them)

## Step 2: Deploy on Railway

1. Go to https://railway.app → Sign in with GitHub
2. Click **New Project** → **Deploy from GitHub repo**
3. Select `magicpin-vera-bot`
4. Railway auto-detects the Procfile and deploys

## Step 3: Set Environment Variables

In Railway → your service → **Variables** tab, add:

| Variable | Value |
|---|---|
| `GEMINI_API_KEY` | `AQ.Ab8RN6I3r_CRIHUEFzLA_Cn5mR_h0wUrWaDRwBSrQezBhs8UYg` |
| `GEMINI_MODEL` | `gemini-2.5-flash` |
| `TEAM_NAME` | `Ansh Sharma` |
| `CONTACT_EMAIL` | `ansh.sharma@kalvium.community` |

## Step 4: Get Live URL

Railway assigns a URL like: `https://magicpin-vera-bot-production.up.railway.app`

## Step 5: Verify

```bash
curl https://YOUR_RAILWAY_URL/v1/healthz
curl https://YOUR_RAILWAY_URL/v1/metadata
```

## Step 6: Run Judge Simulator

```bash
# In d:\magicpin
python judge_simulator.py --bot-url https://YOUR_RAILWAY_URL
```
