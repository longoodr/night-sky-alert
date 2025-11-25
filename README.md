# Night Sky Alert

This project checks the weather forecast for your location daily and alerts you via Pushover if viewing conditions are good that night for solar system bodies (Moon, Mars, Jupiter, Saturn).

## Features

When viewing conditions are favorable, you'll receive a Pushover notification containing:

- 🌙 notification message with the optimal viewing time window
- 📊 Visual chart showing:
  - Hour-by-hour visibility of celestial bodies (Moon, Mars, Jupiter, Saturn)
  - Color-coded altitude information (higher = better visibility)
  - Weather conditions overlaid on the timeline
  - Moon phase indicator

This helps you plan your night sky observations effectively.

## Getting Started - Fork and Configure for Your Own Use

This repository is designed to run automatically via GitHub Actions, sending you personalized night sky alerts. Here's how to set it up for your location:

### Fork the Repository

1. Click the **Fork** button at the top right of this repository
2. This creates your own copy where you can configure your personal settings

### Get Your Pushover Credentials

1. Create a free account at [Pushover.net](https://pushover.net/)
2. Note your **User Key** (displayed on your dashboard)
3. Create a new application at [Pushover.net/apps](https://pushover.net/apps/build)
   - Name it something like "Night Sky Alert"
   - This generates an **API Token/Key** for your app

### Find Your Coordinates

1. Go to [Google Maps](https://www.google.com/maps) and right-click your location
2. Click the coordinates to copy them (e.g., `28.661111, -81.365619`)
3. The first number is your **latitude**, the second is your **longitude**

### Configure GitHub Secrets

In your forked repository:

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret** and add each of the following:

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `LATITUDE` | Your location latitude | `28.661111` |
| `LONGITUDE` | Your location longitude | `-81.365619` |
| `PUSHOVER_USER_KEY` | Your Pushover User Key | `u9od2nfn3n5m4zc...` |
| `PUSHOVER_API_TOKEN` | Your Pushover Application API Token | `awzwdb3wu4juxx7...` |

### (Optional) Configure Variables

Skip this step to use default settings.
Still in **Settings** → **Secrets and variables** → **Actions**, click the **Variables** tab:

1. Click **New repository variable** to customize these settings:

| Variable Name | Default | Description |
|---------------|---------|-------------|
| `CLOUD_COVER_LIMIT` | `15` | Maximum cloud cover percentage required for viewing (0-100) |
| `PRECIP_PROB_LIMIT` | `5` | Maximum precipitation probability required for viewing (0-100) |
| `MIN_VIEWING_HOURS` | `1.0` | Minimum continuous viewing hours required |
| `START_TIME` | (Sunset) | Custom start time in 24-hour format (e.g., `18:00`) |
| `END_TIME` | (Sunrise) | Custom end time in 24-hour format (e.g., `02:00`). Wraps to next morning if needed |
| `MIN_MOON_ILLUMINATION` | `0.0` | Minimum moon illumination required for viewing (0.0 = new, 1.0 = full) |
| `MAX_MOON_ILLUMINATION` | `1.0` | Maximum moon illumination required for viewing (0.0 = new, 1.0 = full) |
| `CHECK_INTERVAL_MINUTES` | `15` | How often to check conditions within the window |

### Enable GitHub Actions

1. Go to the **Actions** tab in your forked repository
2. Click **"I understand my workflows, go ahead and enable them"**
3. The workflow will now run automatically every day at 2:00 PM UTC (adjust the schedule in `.github/workflows/night_sky_alert.yml` if needed)

### Test Your Setup

1. Go to **Actions** tab → **Night Sky Alert** workflow
2. Click **Run workflow** → **Run workflow** (green button)
3. Wait about 30-60 seconds for it to complete
4. Check the logs to see if conditions are good
5. If conditions are favorable, you should receive a Pushover notification with a visibility chart!

### Adjusting the Schedule

The workflow runs daily at 2:00 PM UTC by default. To change this:

1. Edit `.github/workflows/night_sky_alert.yml`
2. Modify the cron schedule (line 5):
   ```yaml
   - cron: '0 14 * * *'  # Minute Hour * * *
   ```
   - For 8:00 AM UTC: `'0 8 * * *'`
   - For 6:00 PM UTC: `'0 18 * * *'`
   - For multiple times per day: Add additional cron lines

## Setup

### GitHub Actions Configuration Details

**Environment variables used by the script:**
- **Required** (from Secrets): `LATITUDE`, `LONGITUDE`, `PUSHOVER_USER_KEY`, `PUSHOVER_API_TOKEN`
- **Optional** (from Variables or defaults): `CLOUD_COVER_LIMIT`, `PRECIP_PROB_LIMIT`, `START_TIME`, `END_TIME`, `MIN_VIEWING_HOURS`, `MIN_MOON_ILLUMINATION`, `MAX_MOON_ILLUMINATION`, `CHECK_INTERVAL_MINUTES`

## Local Development

For testing the script locally on your machine:

1. Clone your forked repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/night-sky-alert-redux.git
   cd night-sky-alert-redux
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and configure with your settings:
   ```bash
   cp .env.example .env
   ```
   
   Then edit `.env` with your actual values.
   See `.env.example` for detailed documentation of all configurable settings.

4. Run the script:
   ```bash
   python src/main.py
   ```

## Testing

Run the test suite to see example notifications and generated charts:

```bash
pytest tests/ -v
```

To update snapshots after intentional changes:

```bash
pytest tests/ --snapshot-update
```

