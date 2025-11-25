# Night Sky Alert

This project checks the weather forecast for your location and alerts you via Pushover if viewing conditions are good for specific astronomical bodies (Moon, Mars, Jupiter, Saturn).

## Setup

### 1. GitHub Secrets
Go to your repository settings -> Secrets and Variables -> Actions, and add the following **Repository Secrets**:

| Secret Name | Description |
|-------------|-------------|
| `LATITUDE` | Your location latitude (e.g., `40.7128`) |
| `LONGITUDE` | Your location longitude (e.g., `-74.0060`) |
| `PUSHOVER_USER_KEY` | Your Pushover User Key |
| `PUSHOVER_API_TOKEN` | Your Pushover Application API Token |

### 2. Configuration Variables (Optional)
You can set these as **Repository Variables** or leave them to use defaults:

| Variable Name | Default | Description |
|---------------|---------|-------------|
| `CLOUD_COVER_LIMIT` | `15` | Max percentage of cloud cover acceptable. |
| `PRECIP_PROB_LIMIT` | `5` | Max probability of precipitation acceptable. |
| `START_TIME` | (Sunset) | Custom start time (HH:MM) in local time. |
| `END_TIME` | (Sunrise)| Custom end time (HH:MM) in local time. |

## Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set environment variables (or create a `.env` file if you modify the script to load it):
   ```powershell
   $env:LATITUDE="40.7128"
   $env:LONGITUDE="-74.0060"
   $env:PUSHOVER_USER_KEY="your_key"
   $env:PUSHOVER_API_TOKEN="your_token"
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
