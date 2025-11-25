import os
import sys
import datetime
import requests
from skyfield.api import load, Topos, wgs84
from skyfield import almanac
import pytz
from pydantic_settings import BaseSettings
from pydantic import Field, ValidationError, field_validator
from typing import Optional
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class Settings(BaseSettings):
    latitude: float = Field(..., description="Latitude of the location")
    longitude: float = Field(..., description="Longitude of the location")
    cloud_cover_limit: float = Field(15.0, description="Max cloud cover percentage")
    precip_prob_limit: float = Field(5.0, description="Max precipitation probability percentage")
    min_viewing_hours: float = Field(1.0, ge=0.5, description="Minimum continuous viewing hours required")
    pushover_user_key: Optional[str] = Field(None, description="Pushover User Key")
    pushover_api_token: Optional[str] = Field(None, description="Pushover API Token")
    start_time: Optional[str] = Field(None, pattern=r'^\d{2}:\d{2}$', description="Custom start time HH:MM")
    end_time: Optional[str] = Field(None, pattern=r'^\d{2}:\d{2}$', description="Custom end time HH:MM")

    @field_validator('start_time', 'end_time')
    @classmethod
    def validate_time_format(cls, v):
        if v is None or v == '':
            return None
        try:
            h, m = map(int, v.split(':'))
            if not (0 <= h <= 23 and 0 <= m <= 59):
                raise ValueError(f"Time {v} out of range. Hours must be 0-23, minutes 0-59.")
        except ValueError as e:
            raise ValueError(f"Invalid time format: {e}")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

try:
    settings = Settings()
except ValidationError as e:
    print("Configuration Error:")
    print(e)
    sys.exit(1)

def validate_time_window_against_sun():
    """Validates custom time windows are within reasonable bounds of sunset/sunrise"""
    if not settings.start_time and not settings.end_time:
        return  # No custom times, nothing to validate
    
    # Quick calculation to get approximate sunset/sunrise
    try:
        ts = load.timescale()
        eph = load('de421.bsp')
        observer = wgs84.latlon(settings.latitude, settings.longitude)
        
        # Get timezone via Open-Meteo
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": settings.latitude, "longitude": settings.longitude, "forecast_days": 1}
        response = requests.get(url, params=params, timeout=10)
        tz_str = response.json().get('timezone', 'UTC')
        local_tz = pytz.timezone(tz_str)
        
        now = datetime.datetime.now(local_tz)
        t0 = ts.from_datetime(now)
        t1 = ts.from_datetime(now + datetime.timedelta(days=2))
        
        f = almanac.find_discrete(t0, t1, almanac.sunrise_sunset(eph, observer))
        
        sunset_dt = None
        sunrise_dt = None
        
        for t, event in zip(f[0], f[1]):
            dt = t.astimezone(local_tz)
            if event == 1 and sunset_dt is None:
                sunset_dt = dt
            elif event == 0 and sunset_dt is not None and sunrise_dt is None:
                sunrise_dt = dt
        
        if not sunset_dt or not sunrise_dt:
            print("Warning: Could not calculate sunset/sunrise for validation. Skipping time window validation.")
            return
        
        # Parse custom times
        if settings.start_time:
            h, m = map(int, settings.start_time.split(':'))
            start_candidate = now.replace(hour=h, minute=m, second=0, microsecond=0)
            
            # Check if start time is during daylight (between sunrise and sunset)
            if sunrise_dt <= start_candidate <= sunset_dt:
                print(f"Warning: START_TIME {settings.start_time} falls during daylight hours.")
                print(f"  Sunset today: {sunset_dt.strftime('%I:%M %p')}, Sunrise tomorrow: {sunrise_dt.strftime('%I:%M %p')}")
                print("  Consider setting START_TIME after sunset for optimal viewing.")
        
        if settings.end_time:
            h, m = map(int, settings.end_time.split(':'))
            end_candidate_today = now.replace(hour=h, minute=m, second=0, microsecond=0)
            end_candidate_tomorrow = (now + datetime.timedelta(days=1)).replace(hour=h, minute=m, second=0, microsecond=0)
            
            # Determine which day the end time is on
            if h < 12:
                end_dt = end_candidate_tomorrow
            else:
                end_dt = end_candidate_today
            
            # Check if end time is during daylight
            if end_dt > sunrise_dt and h >= 6:  # After sunrise in the morning
                print(f"Warning: END_TIME {settings.end_time} may be after sunrise ({sunrise_dt.strftime('%I:%M %p')}).")
                print("  Consider setting END_TIME before sunrise for optimal viewing.")
                
    except Exception as e:
        print(f"Warning: Could not validate time window against sun position: {e}")

# Target Bodies
TARGETS = ['Moon', 'Mars', 'Jupiter', 'Saturn']

def get_weather_forecast(lat, lon):
    # Using Open-Meteo API (Free, no key required for basic usage)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "cloud_cover,precipitation_probability,visibility",
        "timezone": "auto",
        "forecast_days": 2 
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching weather: {e}")
        sys.exit(1)

def send_pushover(message, title="Night Sky Alert", image_data=None):
    if not settings.pushover_user_key or not settings.pushover_api_token:
        print("Pushover credentials not provided. Skipping notification.")
        print(f"Would have sent: {title} - {message}")
        return

    url = "https://api.pushover.net/1/messages.json"
    data = {
        "token": settings.pushover_api_token,
        "user": settings.pushover_user_key,
        "message": message,
        "title": title,
        "monospace": 1  # Use monospace font for better grid alignment
    }
    
    try:
        if image_data:
            # Send with Base64-encoded image attachment
            data["attachment_base64"] = image_data
            data["attachment_type"] = "image/png"
        
        requests.post(url, data=data)
        print("Notification sent successfully" + (" with image." if image_data else "."))
    except Exception as e:
        print(f"Failed to send notification: {e}")

def create_visibility_chart(sorted_times, visibility_grid, body_visibility, targets):
    """Create a visual chart of body visibility over time"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a matrix for the heatmap
    num_bodies = len(targets)
    num_times = len(sorted_times)
    
    # Matrix: rows = bodies, cols = times
    # Values: 0 = not visible, 1 = visible, 2 = high (>15°)
    matrix = [[0 for _ in range(num_times)] for _ in range(num_bodies)]
    
    for t_idx, dt in enumerate(sorted_times):
        for b_idx, body_name in enumerate(targets):
            if body_name in visibility_grid[dt]['bodies']:
                # Check if it's high
                is_high = any(v['time'] == dt and v['high'] for v in body_visibility[body_name])
                matrix[b_idx][t_idx] = 2 if is_high else 1
    
    # Create heatmap
    cmap = matplotlib.colors.ListedColormap(['#1a1a1a', '#4a4a8a', '#ffd700'])
    bounds = [0, 0.5, 1.5, 2.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')
    
    # Set ticks and labels
    ax.set_yticks(range(num_bodies))
    ax.set_yticklabels(targets, fontsize=12)
    
    # Time labels (show every hour with AM/PM)
    time_labels = [dt.strftime('%I:%M %p') for dt in sorted_times]
    ax.set_xticks(range(num_times))
    ax.set_xticklabels(time_labels, rotation=45, ha='right', fontsize=10)
    
    # Grid
    ax.set_xticks([x - 0.5 for x in range(1, num_times)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, num_bodies)], minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
    
    # Title and legend
    ax.set_title('Night Sky Visibility', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#1a1a1a', label='Not Visible'),
        mpatches.Patch(facecolor='#4a4a8a', label='Visible'),
        mpatches.Patch(facecolor='#ffd700', label='High (>15°)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    
    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close()
    
    # Encode to base64
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return image_base64

def main():
    print(f"Starting Night Sky Alert for {settings.latitude}, {settings.longitude}")
    
    # Validate custom time windows against sunset/sunrise
    validate_time_window_against_sun()
    
    # 1. Fetch Weather & Determine Timezone (Open-Meteo handles auto-timezone)
    weather_data = get_weather_forecast(settings.latitude, settings.longitude)
    
    timezone_str = weather_data.get('timezone', 'UTC')
    try:
        local_tz = pytz.timezone(timezone_str)
    except pytz.UnknownTimeZoneError:
        print(f"Unknown timezone {timezone_str}, defaulting to UTC")
        local_tz = pytz.utc
        
    print(f"Detected Timezone: {local_tz.zone}")

    # 2. Determine Time Window (Tonight)
    now = datetime.datetime.now(local_tz)
    
    # Load Skyfield data
    ts = load.timescale()
    eph = load('de421.bsp')
    observer = wgs84.latlon(settings.latitude, settings.longitude)

    # Calculate Sun events for dynamic window
    t0 = ts.from_datetime(now)
    t1 = ts.from_datetime(now + datetime.timedelta(days=1))
    
    # Find sunset today and sunrise tomorrow
    f = almanac.find_discrete(t0, t1, almanac.sunrise_sunset(eph, observer))
    
    sunset_dt = None
    sunrise_dt = None

    # Logic to find the *next* sunset and the *following* sunrise
    # This is a simplified approach; for a robust script we iterate the events
    for t, event in zip(f[0], f[1]):
        dt = t.astimezone(local_tz)
        if event == 1: # Sunset
            if sunset_dt is None: sunset_dt = dt
        elif event == 0: # Sunrise
            if sunrise_dt is None and sunset_dt is not None: 
                sunrise_dt = dt
    
    # Fallback if astronomical calculation fails or happens at odd times (e.g. polar regions)
    if not sunset_dt:
        sunset_dt = now.replace(hour=18, minute=0, second=0, microsecond=0)
    if not sunrise_dt:
        sunrise_dt = (now + datetime.timedelta(days=1)).replace(hour=6, minute=0, second=0, microsecond=0)

    # Override with custom times if provided
    # Smart date handling: if time is before sunrise, it's "tomorrow morning", otherwise "today"
    if settings.start_time:
        h, m = map(int, settings.start_time.split(':'))
        start_candidate = now.replace(hour=h, minute=m, second=0, microsecond=0)
        # If the start time is before sunrise and sunrise is today, the start time is likely tonight
        # This handles cases like START_TIME=23:00 (tonight)
        sunset_dt = start_candidate
    
    if settings.end_time:
        h, m = map(int, settings.end_time.split(':'))
        end_candidate_today = now.replace(hour=h, minute=m, second=0, microsecond=0)
        end_candidate_tomorrow = (now + datetime.timedelta(days=1)).replace(hour=h, minute=m, second=0, microsecond=0)
        
        # If sunrise hasn't happened yet and end time is before sunrise, use today
        # Otherwise, if end time is after midnight (small hour value), use tomorrow
        if sunrise_dt and end_candidate_today < sunrise_dt and now < sunrise_dt:
            # We're currently before sunrise, and end time is also before sunrise -> use today
            sunrise_dt = end_candidate_today
        elif h < 12:  # Early morning hours (00:00 - 11:59) likely mean next day
            sunrise_dt = end_candidate_tomorrow
        else:  # Afternoon/evening hours mean today
            sunrise_dt = end_candidate_today

    print(f"Checking window: {sunset_dt} to {sunrise_dt}")

    # 3. Process Weather Data
    hourly = weather_data.get('hourly', {})
    times = hourly.get('time', [])
    cloud_covers = hourly.get('cloud_cover', [])
    precip_probs = hourly.get('precipitation_probability', [])

    good_viewing_slots = []
    all_window_slots = []  # Track all slots in the window for continuous viewing check

    # 4. Correlate Weather with Time Window
    print("\n--- Weather Check ---")
    for i, time_str in enumerate(times):
        # Open-Meteo returns ISO8601 strings
        dt = datetime.datetime.fromisoformat(time_str)
        # Ensure dt is timezone aware (Open-Meteo returns local time if requested, but let's be safe)
        if dt.tzinfo is None:
            dt = local_tz.localize(dt)
        
        if sunset_dt <= dt <= sunrise_dt:
            cc = cloud_covers[i]
            pp = precip_probs[i]
            
            is_good = cc <= settings.cloud_cover_limit and pp <= settings.precip_prob_limit
            status = "PASS" if is_good else "FAIL"
            print(f"{dt.strftime('%Y-%m-%d %H:%M')}: Cloud {cc}%, Precip {pp}% -> {status}")

            all_window_slots.append({'time': dt, 'is_good': is_good})
            if is_good:
                good_viewing_slots.append(dt)

    # 5. Check for minimum continuous viewing time
    print(f"\n--- Continuous Viewing Check ---")
    print(f"Required: {settings.min_viewing_hours} hour(s) continuous")
    
    longest_continuous_block = []
    current_block = []
    
    for slot in all_window_slots:
        if slot['is_good']:
            current_block.append(slot['time'])
        else:
            if len(current_block) > len(longest_continuous_block):
                longest_continuous_block = current_block.copy()
            current_block = []
    
    # Check final block
    if len(current_block) > len(longest_continuous_block):
        longest_continuous_block = current_block.copy()
    
    continuous_hours = len(longest_continuous_block)
    print(f"Longest continuous clear period: {continuous_hours} hour(s)")
    
    if continuous_hours > 0:
        start = longest_continuous_block[0].strftime('%I:%M %p')
        end = longest_continuous_block[-1].strftime('%I:%M %p')
        print(f"  From {start} to {end}")
    
    if continuous_hours < settings.min_viewing_hours:
        print(f"\nInsufficient continuous viewing time. Need {settings.min_viewing_hours} hours, found {continuous_hours} hours.")
        print("No alert will be sent.")
        return

    print(f"\n✓ Sufficient continuous weather window available!")
    print(f"Found {len(good_viewing_slots)} total hours with good weather.")

    # 6. Check Astronomical Bodies for good weather slots
    report_lines = []
    
    # Cache body objects
    bodies = {}
    for name in TARGETS:
        bodies[name] = eph[name]
    earth = eph['earth']

    # We will group by body to make the report readable
    # Track which bodies are visible during good weather
    body_visibility = {name: [] for name in TARGETS}
    body_slots_with_weather = {name: [] for name in TARGETS}  # Tracks time+weather for each body

    print("\n--- Astronomical Check (Good Weather Slots) ---")
    for dt in good_viewing_slots:
        t = ts.from_datetime(dt)
        observer_loc = earth + observer
        
        print(f"Time: {dt.strftime('%I:%M %p')}")
        for name, body in bodies.items():
            astrometric = observer_loc.at(t).observe(body)
            alt, az, distance = astrometric.apparent().altaz()
            
            alt_degrees = alt.degrees
            print(f"  {name}: Alt {alt_degrees:.2f}°, Az {az.degrees:.2f}°")
            
            if alt_degrees > 0:
                is_high = alt_degrees >= 15
                body_visibility[name].append({
                    'time': dt,
                    'alt': alt_degrees,
                    'high': is_high
                })
                body_slots_with_weather[name].append(dt)

    # 7. Check continuous viewing time for each visible body
    print("\n--- Continuous Viewing Per Body ---")
    bodies_meeting_requirement = {}
    
    for name in TARGETS:
        if not body_slots_with_weather[name]:
            print(f"{name}: Not visible above horizon")
            continue
        
        # Find longest continuous block for this body
        sorted_times = sorted(body_slots_with_weather[name])
        longest_block = []
        current_block = [sorted_times[0]]
        
        for i in range(1, len(sorted_times)):
            # Check if this slot is consecutive (1 hour apart)
            time_diff = (sorted_times[i] - sorted_times[i-1]).total_seconds() / 3600
            if time_diff <= 1.5:  # Allow up to 1.5 hours gap (accounts for hourly data)
                current_block.append(sorted_times[i])
            else:
                if len(current_block) > len(longest_block):
                    longest_block = current_block.copy()
                current_block = [sorted_times[i]]
        
        # Check final block
        if len(current_block) > len(longest_block):
            longest_block = current_block.copy()
        
        continuous_hours_for_body = len(longest_block)
        
        if continuous_hours_for_body >= settings.min_viewing_hours:
            start = longest_block[0].strftime('%I:%M %p')
            end = longest_block[-1].strftime('%I:%M %p')
            print(f"{name}: ✓ {continuous_hours_for_body} hours ({start}-{end})")
            bodies_meeting_requirement[name] = longest_block
        else:
            print(f"{name}: ✗ Only {continuous_hours_for_body} hours (need {settings.min_viewing_hours})")
    
    if not bodies_meeting_requirement:
        print(f"\n✗ No astronomical bodies meet the {settings.min_viewing_hours} hour continuous viewing requirement.")
        print("No alert will be sent.")
        return

    # 8. Create visibility grid and find best viewing blocks
    print("\n--- Analyzing Best Viewing Blocks ---")
    
    # Build a time-indexed grid of which bodies are visible
    all_qualified_times = set()
    for times_list in bodies_meeting_requirement.values():
        all_qualified_times.update(times_list)
    
    sorted_all_times = sorted(all_qualified_times)
    
    # For each time, count how many bodies are visible
    visibility_grid = {}
    for dt in sorted_all_times:
        visible_bodies = [name for name, times in bodies_meeting_requirement.items() if dt in times]
        visibility_grid[dt] = {
            'bodies': visible_bodies,
            'count': len(visible_bodies)
        }
    
    # Find continuous blocks and score them by: (duration * avg_bodies_visible)
    viewing_blocks = []
    current_block = {
        'times': [sorted_all_times[0]],
        'bodies_per_time': [visibility_grid[sorted_all_times[0]]['count']]
    }
    
    for i in range(1, len(sorted_all_times)):
        time_diff = (sorted_all_times[i] - sorted_all_times[i-1]).total_seconds() / 3600
        if time_diff <= 1.5:  # Consecutive
            current_block['times'].append(sorted_all_times[i])
            current_block['bodies_per_time'].append(visibility_grid[sorted_all_times[i]]['count'])
        else:
            # Save current block
            viewing_blocks.append(current_block.copy())
            current_block = {
                'times': [sorted_all_times[i]],
                'bodies_per_time': [visibility_grid[sorted_all_times[i]]['count']]
            }
    
    # Don't forget the last block
    viewing_blocks.append(current_block.copy())
    
    # Score and sort blocks
    for block in viewing_blocks:
        duration = len(block['times'])
        avg_bodies = sum(block['bodies_per_time']) / len(block['bodies_per_time'])
        max_bodies = max(block['bodies_per_time'])
        block['duration'] = duration
        block['avg_bodies'] = avg_bodies
        block['max_bodies'] = max_bodies
        block['score'] = duration * avg_bodies  # Score: longer + more bodies = better
        block['start'] = block['times'][0]
        block['end'] = block['times'][-1]
    
    # Sort by score (best first)
    viewing_blocks.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"Found {len(viewing_blocks)} viewing block(s):")
    for i, block in enumerate(viewing_blocks[:3], 1):  # Show top 3
        start_str = block['start'].strftime('%I:%M %p')
        end_str = block['end'].strftime('%I:%M %p')
        print(f"  #{i}: {start_str}-{end_str} ({block['duration']}h) - avg {block['avg_bodies']:.1f} bodies, max {block['max_bodies']}")

    # 9. Build visibility chart (ASCII grid)
    print("\n--- Visibility Chart ---")
    chart_lines = ["Time      " + " ".join([f"{name:8}" for name in TARGETS])]
    chart_lines.append("-" * (10 + 9 * len(TARGETS)))
    
    for dt in sorted_all_times:
        time_str = dt.strftime('%I:%M %p')
        row = f"{time_str:9}"
        for name in TARGETS:
            if name in visibility_grid[dt]['bodies']:
                # Check if it's high (>15°)
                is_high = any(v['time'] == dt and v['high'] for v in body_visibility[name])
                symbol = "▓▓▓▓" if is_high else "░░░░"
            else:
                symbol = "    "
            row += f"{symbol:8} "
        chart_lines.append(row)
    
    chart = "\n".join(chart_lines)
    print(chart)

    # 10. Construct Notification
    best_block = viewing_blocks[0]
    best_start = best_block['start'].strftime('%I:%M %p')
    best_end = best_block['end'].strftime('%I:%M %p')
    
    message_parts = [
        f"🌙 Clear Skies Alert! 🌙",
        f"Best Block: {best_start}-{best_end} ({best_block['duration']}h, up to {best_block['max_bodies']} bodies)",
        f"Weather: Cloud<{settings.cloud_cover_limit}%, Precip<{settings.precip_prob_limit}%",
        ""
    ]
    
    # Add per-body details
    for name in TARGETS:
        if name not in bodies_meeting_requirement:
            continue
        
        # Use the longest continuous block for this body
        longest_block = bodies_meeting_requirement[name]
        start_time = longest_block[0].strftime('%I:%M %p')
        end_time = longest_block[-1].strftime('%I:%M %p')
        
        # Check for high visibility (>15 deg) within this block
        high_times = []
        for visibility_info in body_visibility[name]:
            if visibility_info['high'] and visibility_info['time'] in longest_block:
                high_times.append(visibility_info['time'])
        
        high_str = ""
        if high_times:
            h_start = min(high_times).strftime('%I:%M %p')
            h_end = max(high_times).strftime('%I:%M %p')
            high_str = f" | >15° {h_start}-{h_end}"
        
        message_parts.append(f"{name}: {start_time}-{end_time}{high_str}")
    
    # Generate visual chart image
    print("\n--- Generating Visibility Chart Image ---")
    image_data = create_visibility_chart(sorted_all_times, visibility_grid, body_visibility, TARGETS)
    print(f"Chart generated ({len(image_data)} bytes base64)")

    full_message = "\n".join(message_parts)
    print("\n--- Sending Notification ---")
    print(full_message)
    send_pushover(full_message, image_data=image_data)

if __name__ == "__main__":
    main()
