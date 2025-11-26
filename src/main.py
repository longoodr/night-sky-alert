import os
import sys
import re
import random
import datetime
import requests
import numpy as np
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
import matplotlib.patheffects as path_effects
from matplotlib import font_manager

# Configure matplotlib to use portable custom fonts
# User must provide 'arial.ttf' and 'emoji.ttf' in the assets folder
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
text_font_path = os.path.join(project_root, 'assets', 'font.ttf')
emoji_font_path = os.path.join(project_root, 'assets', 'emoji.ttf')

# Register custom fonts
font_manager.fontManager.addfont(text_font_path)
font_manager.fontManager.addfont(emoji_font_path)

# Get internal font names to configure the family list correctly
text_font_name = font_manager.FontProperties(fname=text_font_path).get_name()
emoji_font_name = font_manager.FontProperties(fname=emoji_font_path).get_name()

# Set font family priority: Text Font -> Emoji Font
plt.rcParams['font.family'] = [text_font_name, emoji_font_name]
print(f"Font configuration: {text_font_name} with fallback to {emoji_font_name}")

class ChartColors:
    BACKGROUND = "#080F1B"
    WARNING = '#e07060'  # Warm coral - fits twilight palette
    
    # Navy to yellow via muted twilight purple
    # Midnight blue → twilight purple → dawn ember → sunrise yellow
    ALT_NOT_VISIBLE = '#00000a'
    ALT_LOW = '#1a3366'
    ALT_MID = '#775588'
    ALT_HIGH = '#cc8866'
    ALT_ZENITH = '#eee8bb'
    TEXT = ALT_ZENITH
    DOT_FILL = ALT_ZENITH
    DOT_EDGE = '#000022'
    GRID = ALT_ZENITH



class Settings(BaseSettings):
    latitude: float = Field(..., description="Latitude of the location")
    longitude: float = Field(..., description="Longitude of the location")
    cloud_cover_limit: float = Field(15.0, description="Max cloud cover percentage")
    precip_prob_limit: float = Field(5.0, description="Max precipitation probability percentage")
    min_viewing_hours: float = Field(1.0, ge=0.5, description="Minimum continuous viewing hours required")
    check_interval_minutes: int = Field(15, ge=1, le=60, description="Minutes between astronomical checks (1-60, default 15)")
    min_moon_illumination: float = Field(0.0, ge=0.0, le=1.0, description="Minimum moon illumination (0.0-1.0, 0=new, 1=full)")
    max_moon_illumination: float = Field(1.0, ge=0.0, le=1.0, description="Maximum moon illumination (0.0-1.0, 0=new, 1=full)")
    pushover_user_key: Optional[str] = Field(None, description="Pushover User Key")
    pushover_api_token: Optional[str] = Field(None, description="Pushover API Token")
    start_time: Optional[str] = Field(None, description="Custom start time HH:MM")
    end_time: Optional[str] = Field(None, description="Custom end time HH:MM")

    @field_validator('cloud_cover_limit', mode='before')
    @classmethod
    def validate_cloud_cover(cls, v):
        """Convert empty strings to default for cloud_cover_limit"""
        if v == '' or v is None:
            return 15.0
        return v
    
    @field_validator('precip_prob_limit', mode='before')
    @classmethod
    def validate_precip_prob(cls, v):
        """Convert empty strings to default for precip_prob_limit"""
        if v == '' or v is None:
            return 5.0
        return v
    
    @field_validator('min_viewing_hours', mode='before')
    @classmethod
    def validate_min_viewing_hours(cls, v):
        """Convert empty strings to default for min_viewing_hours"""
        if v == '' or v is None:
            return 1.0
        return v
    
    @field_validator('check_interval_minutes', mode='before')
    @classmethod
    def validate_check_interval(cls, v):
        """Convert empty strings to default for check_interval_minutes"""
        if v == '' or v is None:
            return 15
        return v
    
    @field_validator('min_moon_illumination', mode='before')
    @classmethod
    def validate_min_moon(cls, v):
        """Convert empty strings to default for min_moon_illumination"""
        if v == '' or v is None:
            return 0.0
        return v
    
    @field_validator('max_moon_illumination', mode='before')
    @classmethod
    def validate_max_moon(cls, v):
        """Convert empty strings to default for max_moon_illumination"""
        if v == '' or v is None:
            return 1.0
        return v

    @field_validator('start_time', 'end_time', mode='before')
    @classmethod
    def validate_time_format(cls, v):
        if v is None or v == '':
            return None
        if not isinstance(v, str):
            return v
        # Validate HH:MM format
        if not re.match(r'^\d{2}:\d{2}$', v):
            raise ValueError(f"Time must be in HH:MM format, got: {v}")
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


def calculate_viewing_window(now, latitude, longitude, start_time_str=None, end_time_str=None):
    """
    Calculate the viewing window (start and end times) based on sun position and settings.
    
    Args:
        now: Current datetime (timezone-aware)
        latitude: Observer latitude
        longitude: Observer longitude  
        start_time_str: Optional custom start time in "HH:MM" format
        end_time_str: Optional custom end time in "HH:MM" format
        
    Returns:
        tuple: (start_dt, end_dt) - Both timezone-aware datetimes
        
    Logic:
        - If currently in darkness (after sunset, before sunrise): use current night
        - Otherwise: use upcoming night (next sunset to following midnight/sunrise)
        - Default end time is midnight unless custom end_time specified
        - Custom times override astronomical calculations
    """
    ts = load.timescale()
    eph = load('de421.bsp')
    observer = wgs84.latlon(latitude, longitude)

    # Look back 24 hours and forward 24 hours to find relevant sunset/sunrise
    t0 = ts.from_datetime(now - datetime.timedelta(days=1))
    t1 = ts.from_datetime(now + datetime.timedelta(days=1))
    
    # Find all sun events in the 48-hour window
    f = almanac.find_discrete(t0, t1, almanac.sunrise_sunset(eph, observer))
    
    local_tz = now.tzinfo
    
    # Collect all sunsets and sunrises
    # Note: sunrise_sunset returns True (1) for sunrise, False (0) for sunset
    sunsets = []
    sunrises = []
    for t, event in zip(f[0], f[1]):
        dt = t.astimezone(local_tz)
        if event == 1:  # Sunrise (sun crosses above horizon)
            sunrises.append(dt)
        elif event == 0:  # Sunset (sun crosses below horizon)
            sunsets.append(dt)
    
    # Determine which night we're in or approaching
    sunset_dt = None
    sunrise_dt = None
    
    # Find the most recent sunset before or at now
    past_sunsets = [s for s in sunsets if s <= now]
    future_sunsets = [s for s in sunsets if s > now]
    past_sunrises = [s for s in sunrises if s <= now]
    future_sunrises = [s for s in sunrises if s > now]
    
    # Determine if we're currently in darkness (nighttime)
    # We're in darkness if: last sunset > last sunrise (sunset happened more recently)
    in_darkness = False
    if past_sunsets and past_sunrises:
        most_recent_sunset = max(past_sunsets)
        most_recent_sunrise = max(past_sunrises)
        if most_recent_sunset > most_recent_sunrise:
            # Sunset was more recent than sunrise = it's nighttime
            in_darkness = True
            sunset_dt = most_recent_sunset
            if future_sunrises:
                sunrise_dt = min(future_sunrises)
    elif past_sunsets and not past_sunrises:
        # Only sunsets in the past, no sunrise yet = still night from yesterday
        in_darkness = True
        sunset_dt = max(past_sunsets)
        if future_sunrises:
            sunrise_dt = min(future_sunrises)
    
    # If we're not in darkness (daytime) or didn't find current night, use upcoming night
    if sunset_dt is None:
        if future_sunsets:
            sunset_dt = min(future_sunsets)
            # Find the sunrise after this sunset
            sunrises_after_sunset = [s for s in sunrises if s > sunset_dt]
            if sunrises_after_sunset:
                sunrise_dt = min(sunrises_after_sunset)
    
    # Fallback if astronomical calculation fails
    if not sunset_dt:
        sunset_dt = now.replace(hour=18, minute=0, second=0, microsecond=0)
    if not sunrise_dt:
        sunrise_dt = (sunset_dt + datetime.timedelta(days=1)).replace(hour=6, minute=0, second=0, microsecond=0)

    # Apply default end time of midnight if no custom end time specified
    if not end_time_str:
        midnight = (sunset_dt + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        if midnight <= sunset_dt:
            midnight += datetime.timedelta(days=1)
        sunrise_dt = midnight

    # Override with custom start time if provided
    if start_time_str:
        h, m = map(int, start_time_str.split(':'))
        start_candidate = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if h >= 12:  # Evening time
            # If it's already past this time today but we're in darkness, 
            # the sunset already happened
            if start_candidate <= now and sunset_dt and sunset_dt <= now:
                # Keep the astronomical sunset
                pass
            else:
                sunset_dt = start_candidate
        else:  # Morning time - unusual for start
            sunset_dt = start_candidate
    
    # Override with custom end time if provided
    if end_time_str:
        h, m = map(int, end_time_str.split(':'))
        end_candidate = now.replace(hour=h, minute=m, second=0, microsecond=0)
        
        # End time should be after start time
        if end_candidate <= sunset_dt:
            end_candidate += datetime.timedelta(days=1)
        sunrise_dt = end_candidate
    
    # Final validation: ensure end is after start
    if sunrise_dt <= sunset_dt:
        sunrise_dt += datetime.timedelta(days=1)
    
    return sunset_dt, sunrise_dt


# Target Bodies
TARGETS = ['Moon', 'Mars', 'Jupiter', 'Saturn']

def format_body_list(bodies_in_block: list[str], total_bodies_tracked: int) -> str:
    """
    Format a list of astronomical bodies for display in notification.
    
    Args:
        bodies_in_block: Sorted list of body names visible in the block
        total_bodies_tracked: Total number of bodies being tracked/meeting requirements
    
    Returns:
        Formatted string with body names, prefixed with "including" only if partial visibility
        
    Examples:
        - All visible (2/2): "Jupiter and Saturn"
        - Partial (2/3): "including Jupiter and Saturn"
        - All visible (1/1): "the Moon"
        - Partial (1/3): "including the Moon"
    """
    if not bodies_in_block:
        return ""
    
    # Check if all tracked bodies are visible in the block
    all_bodies_visible = len(bodies_in_block) == total_bodies_tracked
    
    if len(bodies_in_block) == 1:
        return bodies_in_block[0] if all_bodies_visible else f"including {bodies_in_block[0]}"
    elif len(bodies_in_block) == 2:
        if all_bodies_visible:
            return f"{bodies_in_block[0]} and {bodies_in_block[1]}"
        else:
            return f"including {bodies_in_block[0]} and {bodies_in_block[1]}"
    else:
        if all_bodies_visible:
            return f"{', '.join(bodies_in_block[:-1])}, and {bodies_in_block[-1]}"
        else:
            return f"including {', '.join(bodies_in_block[:-1])}, and {bodies_in_block[-1]}"

def get_location_name(lat, lon):
    """Get location name from coordinates using Nominatim (OpenStreetMap) reverse geocoding"""
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": lat,
            "lon": lon,
            "format": "json",
            "zoom": 10  # City/town level
        }
        headers = {
            "User-Agent": "NightSkyAlert/1.0"  # Required by Nominatim
        }
        response = requests.get(url, params=params, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # Try to get city, town, or village name
        address = data.get('address', {})
        location = (
            address.get('city') or 
            address.get('town') or 
            address.get('village') or 
            address.get('county') or
            address.get('state') or
            'Unknown Location'
        )
        return location
    except Exception as e:
        print(f"Warning: Could not fetch location name: {e}")
        return None

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
        "sound": "spacealarm",
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

def create_visibility_chart(sorted_times, visibility_grid, body_visibility, targets, continuous_blocks=None, weather_blocks=None, moon_phase_emoji=None, location_name=None, date_str=None):
    """Create a visual chart with continuous altitude gradients and weather overlays
    
    Args:
        sorted_times: List of datetime objects for each time point
        visibility_grid: Dict mapping times to visible bodies
        body_visibility: Dict mapping body names to visibility data
        targets: List of body names
        continuous_blocks: Dict mapping body names to lists of times in continuous viewing blocks
        weather_blocks: Dict mapping datetime (hourly) to weather conditions {'cloud': float, 'precip': float, 'good': bool}
        moon_phase_emoji: Optional emoji string representing moon phase
        location_name: Optional location name for chart title
        date_str: Optional date string for chart title
    """
    # Dynamic height based on number of bodies
    # We use a larger minimum height to prevent the chart from looking squished,
    # but we'll manually position the axes to keep the bars at their original size.
    BODY_HEIGHT_INCH = 0.8
    BASE_HEIGHT_INCH = 2.0
    MIN_FIG_HEIGHT = 4.0
    
    needed_body_height = len(targets) * BODY_HEIGHT_INCH
    chart_height = max(MIN_FIG_HEIGHT, BASE_HEIGHT_INCH + needed_body_height)
    
    fig = plt.figure(figsize=(14, chart_height), facecolor=ChartColors.BACKGROUND)
    
    # Calculate normalized coordinates for the plot axes
    # We want the plot to be exactly 'needed_body_height' inches tall
    plot_height_norm = needed_body_height / chart_height
    
    # Center the plot vertically
    # But shift slightly down to account for title being at the top
    # Available space is roughly centered, but let's just center it in the figure
    plot_bottom_norm = (1.0 - plot_height_norm) / 2
    
    # Define axes positions [left, bottom, width, height]
    # Main plot: 10% left margin, 75% width
    ax = fig.add_axes([0.1, plot_bottom_norm, 0.75, plot_height_norm])
    
    # Set axis colors for dark mode
    ax.tick_params(axis='x', colors=ChartColors.TEXT)
    ax.tick_params(axis='y', colors=ChartColors.TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(ChartColors.TEXT)
    
    # Colorbar: To the right, tall (elongated scale)
    # Let's make it 80% of the figure height, centered
    cbar_height_norm = 0.8
    cbar_bottom_norm = (1.0 - cbar_height_norm) / 2
    cax = fig.add_axes([0.87, cbar_bottom_norm, 0.02, cbar_height_norm])
    
    # Create a matrix for the heatmap
    num_bodies = len(targets)
    num_times = len(sorted_times)
    
    # Matrix: rows = bodies, cols = times
    # Values: continuous altitude (0-90 degrees) or -1 for not visible
    matrix = [[-1 for _ in range(num_times)] for _ in range(num_bodies)]
    
    for t_idx, dt in enumerate(sorted_times):
        for b_idx, body_name in enumerate(targets):
            # Find visibility data for this exact time
            vis_data = [v for v in body_visibility[body_name] if v['time'] == dt]
            if vis_data:
                matrix[b_idx][t_idx] = vis_data[0]['alt']
            else:
                matrix[b_idx][t_idx] = -1
    
    # Navy → neutral gray → warm yellow (avoids green by desaturating in middle)
    colors_list = [
        (0.0, ChartColors.ALT_NOT_VISIBLE),   # Deep night (near black)
        (0.25, ChartColors.ALT_LOW),          # Dark navy
        (0.5, ChartColors.ALT_MID),           # Neutral gray-blue (pivot point)
        (0.75, ChartColors.ALT_HIGH),         # Warm tan
        (1.0, ChartColors.ALT_ZENITH)         # Bright starlight yellow
    ]
    
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('altitude', colors_list, N=256)
    
    # Normalize: -1 to 90 degrees mapped to 0-1 colormap range
    # Values < 0 (not visible) → 0 (black)
    # Values 0-90 (altitude) → 0.01-1.0 (gradient)
    normalized_matrix = []
    for row in matrix:
        normalized_row = []
        for val in row:
            if val < 0:
                normalized_row.append(0.0)  # Not visible
            else:
                # Map 0-90 degrees to 0.01-1.0
                normalized_row.append(0.01 + (val / 90.0) * 0.99)
        normalized_matrix.append(normalized_row)
    
    # Resample to 1-minute intervals using linear interpolation
    # This provides a smooth gradient when rendered with bilinear interpolation
    target_interval = 1
    
    # Calculate current interval
    if len(sorted_times) > 1:
        current_interval = (sorted_times[1] - sorted_times[0]).total_seconds() / 60
    else:
        current_interval = 15
        
    upsample_factor = int(current_interval / target_interval)
    if upsample_factor < 1: upsample_factor = 1
    
    upsampled_matrix = []
    for row in normalized_matrix:
        new_row = []
        for i in range(len(row) - 1):
            start_val = row[i]
            end_val = row[i+1]
            # Linear interpolation
            for j in range(upsample_factor):
                val = start_val + (end_val - start_val) * (j / upsample_factor)
                new_row.append(val)
        new_row.append(row[-1]) # Append last value
        upsampled_matrix.append(new_row)

    # Time labels - show at 30-minute intervals aligned to the hour/half-hour
    # Calculate time interval (assuming constant)
    if len(sorted_times) > 1:
        interval_minutes = (sorted_times[1] - sorted_times[0]).total_seconds() / 60
    else:
        interval_minutes = 15 # Default fallback

    start_time = sorted_times[0]
    end_time = sorted_times[-1]
    
    # Calculate aligned start/end times for padding
    aligned_start = start_time.replace(second=0, microsecond=0)
    if aligned_start.minute >= 30:
        aligned_start = aligned_start.replace(minute=30)
    else:
        aligned_start = aligned_start.replace(minute=0)
        
    aligned_end = end_time.replace(second=0, microsecond=0)
    if aligned_end.minute > 30:
        aligned_end = aligned_end.replace(minute=0) + datetime.timedelta(hours=1)
    elif aligned_end.minute > 0:
        aligned_end = aligned_end.replace(minute=30)
    
    # Calculate padding pixels
    # 1 pixel = target_interval minutes (1 minute)
    pixels_per_minute = 1.0 / target_interval
    
    pad_left_minutes = (start_time - aligned_start).total_seconds() / 60
    pad_left_pixels = int(round(pad_left_minutes * pixels_per_minute))
    
    pad_right_minutes = (aligned_end - end_time).total_seconds() / 60
    pad_right_pixels = int(round(pad_right_minutes * pixels_per_minute))
    
    # Add extra visual padding so dots at the edges aren't cut off
    # 1 pixel = 1 minute, so 5 pixels padding is sufficient
    visual_padding_pixels = 5
    
    # Pad the matrix
    padded_matrix = []
    for row in upsampled_matrix:
        # Pad left with first value
        left_padding = [row[0]] * (pad_left_pixels + visual_padding_pixels)
        # Pad right with last value
        right_padding = [row[-1]] * (pad_right_pixels + visual_padding_pixels)
        padded_matrix.append(left_padding + row + right_padding)
    
    # Render each row separately to avoid vertical blending
    # This allows us to use bilinear interpolation for smooth horizontal gradients
    # while keeping sharp vertical boundaries between bodies.
    im = None
    for i, row in enumerate(padded_matrix):
        # Reshape row to 1xN matrix
        row_data = [row]
        # Extent: [left, right, bottom, top]
        # Note: y-axis is inverted by default for imshow, so top is i-0.5, bottom is i+0.5
        # But extent expects (left, right, bottom, top) in data coordinates.
        # If we don't invert y-axis manually, 0 is top.
        # Let's use standard extent and let imshow handle it.
        # We want row i to cover y range [i-0.5, i+0.5]
        # Since y increases downwards (default), bottom is i+0.5, top is i-0.5
        extent = [0, len(row), i + 0.5, i - 0.5]
        
        im = ax.imshow(row_data, cmap=cmap, aspect='auto', interpolation='bilinear', 
                      vmin=0, vmax=1, extent=extent)
    
    # Set limits explicitly since we're adding multiple images
    ax.set_xlim(0, len(padded_matrix[0]))
    ax.set_ylim(num_bodies - 0.5, -0.5) # Inverted y-axis: max at bottom, min at top
    
    # Add colorbar showing altitude scale
    # Use the dedicated colorbar axes
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Altitude', rotation=270, labelpad=20, fontsize=11, color=ChartColors.TEXT)
    cbar.set_ticks([0, 0.17, 0.34, 0.51, 0.68, 0.84, 1.0])
    cbar.set_ticklabels(['≤0°', '15°', '30°', '45°', '60°', '75°', '90°'])
    cbar.ax.yaxis.set_tick_params(color=ChartColors.TEXT, labelcolor=ChartColors.TEXT)
    cbar.outline.set_edgecolor(ChartColors.TEXT)
    
    # Add outline to colorbar labels - use not visible color for contrast
    cbar_outline = [path_effects.withStroke(linewidth=2, foreground=ChartColors.ALT_NOT_VISIBLE)]
    cbar.ax.yaxis.label.set_path_effects(cbar_outline)
    for label in cbar.ax.get_yticklabels():
        label.set_path_effects(cbar_outline)
    
    # Set ticks and labels - add moon emoji to Moon label
    ax.set_yticks(range(num_bodies))
    y_labels = []
    for target in targets:
        if target == 'Moon' and moon_phase_emoji:
            y_labels.append(f'Moon {moon_phase_emoji}')
        else:
            y_labels.append(target)
    ax.set_yticklabels(y_labels, fontsize=12, fontweight='bold', color=ChartColors.TEXT)
    
    # Left align y-axis labels using exact width calculation
    # We need to draw the canvas to get the text extents
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    
    max_width = 0
    for label in ax.get_yticklabels():
        # Get bounding box in display coordinates
        bbox = label.get_window_extent(renderer)
        # Transform to axes coordinates
        bbox_axes = bbox.transformed(ax.transAxes.inverted())
        max_width = max(max_width, bbox_axes.width)
        
    # Position labels so the longest one ends just left of the axis (with padding)
    padding = 0
    x_offset = -(max_width + padding)
    
    # Text outline effect for readability over starfield
    text_outline = [path_effects.withStroke(linewidth=2, foreground=ChartColors.ALT_NOT_VISIBLE)]
    
    for label in ax.get_yticklabels():
        label.set_horizontalalignment('left')
        label.set_x(x_offset)
        label.set_path_effects(text_outline)
    
    # Generate ticks based on aligned times
    current_tick = aligned_start
    tick_positions = []
    tick_labels = []
    tick_colors = []
    
    while current_tick <= aligned_end:
        # Calculate position relative to aligned start
        # Since we padded the matrix, index 0 corresponds to aligned_start MINUS visual_padding_pixels
        # So aligned_start is at index visual_padding_pixels
        pos = (current_tick - aligned_start).total_seconds() / 60 * pixels_per_minute + visual_padding_pixels
        
        tick_positions.append(pos)
        tick_labels.append(current_tick.strftime('%I:%M %p'))
        
        # Check weather for this tick
        color = ChartColors.TEXT
        if weather_blocks:
            nearest_hour = current_tick.replace(minute=0, second=0, microsecond=0)
            if nearest_hour in weather_blocks and not weather_blocks[nearest_hour]['good']:
                color = ChartColors.WARNING
        tick_colors.append(color)
        
        current_tick += datetime.timedelta(minutes=30)
        
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
    
    # Apply colors and outline effect
    for i, color in enumerate(tick_colors):
        ax.get_xticklabels()[i].set_color(color)
        ax.get_xticklabels()[i].set_path_effects(text_outline)
        # if color == '#ff4444':
        #     ax.get_xticklabels()[i].set_weight('bold')

    # Add altitude dots at tick marks
    # Since imshow origin is 'upper' (default), y increases downwards.
    # Row i is centered at y=i. Top is i-0.5, Bottom is i+0.5.
    # We want high altitude to be visually 'up' (lower y value).
    for x_pos in tick_positions:
        col_idx = int(round(x_pos))
        if 0 <= col_idx < len(padded_matrix[0]):
            for body_idx in range(num_bodies):
                val = padded_matrix[body_idx][col_idx]
                if val > 0: # Visible
                    # val is 0.01 + (alt/90)*0.99
                    norm_alt = (val - 0.01) / 0.99
                    
                    # Map to y-position within the body's row
                    # Controls the empty space at the top and bottom of the bar where dots won't be drawn
                    dot_vertical_padding = 0.02
                    
                    # Calculate drawing range within the row (centered at body_idx)
                    # Row extends from body_idx-0.5 to body_idx+0.5
                    bottom_limit = body_idx + 0.5 - dot_vertical_padding
                    draw_height = 1.0 - (2 * dot_vertical_padding)
                    
                    y_pos = bottom_limit - (norm_alt * draw_height)
                    
                    ax.plot(x_pos, y_pos, 'o', color=ChartColors.DOT_FILL, markeredgecolor=ChartColors.DOT_EDGE, markeredgewidth=0.5, markersize=4)
    
    # Grid - horizontal only to separate bodies, no vertical lines
    ax.set_yticks([y - 0.5 for y in range(1, num_bodies)], minor=True)
    ax.grid(which='minor', axis='y', color=ChartColors.GRID, linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Title with location and date
    if location_name and date_str:
        title_text = f'Visibility Report · {location_name} · {date_str}'
    elif location_name:
        title_text = f'Visibility Report · {location_name}'
    elif date_str:
        title_text = f'Visibility Report · {date_str}'
    else:
        title_text = 'Visibility Report'
    
    # Text outline effect for readability over starfield
    text_outline = [path_effects.withStroke(linewidth=2, foreground=ChartColors.ALT_NOT_VISIBLE)]
    
    # Use suptitle to keep title at the top of the figure, independent of the plot axis
    # Align title with the plot center (0.475) to match the Time label
    title = fig.suptitle(title_text, fontsize=16, fontweight='bold', x=0.475, y=0.885, color=ChartColors.TEXT)
    title.set_path_effects(text_outline)
    
    # Calculate Y position for labels (Time and Warning)
    # Place them approx 0.95 inches below the plot area to clear rotated tick labels
    label_y_inch = 0.95
    label_y_pos = max(0.02, plot_bottom_norm - (label_y_inch / chart_height))
    
    # Add Time label manually to align with warning text
    # Plot is from 0.1 to 0.85 (width 0.75), so center is 0.475
    time_label = fig.text(0.475, label_y_pos, 'Time', ha='center', va='bottom', fontsize=12, fontweight='bold', color=ChartColors.TEXT)
    time_label.set_path_effects(text_outline)
    
    # Add note about red time labels (bottom right corner)
    if weather_blocks:
        has_bad_weather = any(not wb['good'] for wb in weather_blocks.values())
        if has_bad_weather:
            # Position in line with Time label, aligned with right edge of plot (0.85)
            warning_label = fig.text(0.85, label_y_pos, 'Red times indicate poor visibility conditions.',
                    ha='right', va='bottom', fontsize=9, style='italic', color=ChartColors.WARNING)
            warning_label.set_path_effects(text_outline)
    
    # plt.tight_layout() # Removed as it conflicts with add_axes
    
    # First render to determine actual output size
    buf_temp = io.BytesIO()
    plt.savefig(buf_temp, format='png', dpi=150, bbox_inches='tight', facecolor=ChartColors.BACKGROUND)
    buf_temp.seek(0)
    
    # Get dimensions from rendered image
    from PIL import Image, ImageDraw
    temp_img = Image.open(buf_temp)
    width, height = temp_img.size
    temp_img.close()
    
    # Generate starfield background at exact size
    def hex_to_rgb(hex_color):
        h = hex_color.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    
    bg_rgb = hex_to_rgb(ChartColors.BACKGROUND)
    starfield = Image.new('RGB', (width, height), bg_rgb)
    draw = ImageDraw.Draw(starfield)
    
    np.random.seed(42)
    n_stars = 10000
    for _ in range(n_stars):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        
        # Exponential size distribution (most stars tiny, exponential tail, truncate at 2)
        size = np.random.exponential(scale=0.4)
        size = min(size, 2.0)  # Truncate at 2
        size = max(size, 0.3)  # Minimum visible size
        
        # Stellar classification with realistic frequencies
        # Hotter stars are rarer but intrinsically brighter
        # Cooler stars are common but dimmer
        color_choice = np.random.choice([
            'O_blue',       # O stars - very rare, very hot, very bright
            'B_blue_white', # B stars - rare, hot, bright  
            'A_white',      # A stars - uncommon, bright
            'F_cream',      # F stars - moderate
            'G_yellow',     # G stars (Sun-like) - common
            'K_gold',       # K stars - very common
            'M_red',        # M stars - most common, dimmest
        ], p=[0.01, 0.04, 0.08, 0.15, 0.20, 0.25, 0.27])
        
        # Base colors and intrinsic brightness by spectral type
        # Hotter = bluer and intrinsically brighter, Cooler = redder and dimmer
        if color_choice == 'O_blue':
            base = (180, 200, 255)
            intrinsic_brightness = 0.9  # Very bright
        elif color_choice == 'B_blue_white':
            base = (200, 220, 255)
            intrinsic_brightness = 0.75
        elif color_choice == 'A_white':
            base = (240, 245, 255)
            intrinsic_brightness = 0.6
        elif color_choice == 'F_cream':
            base = (255, 250, 230)
            intrinsic_brightness = 0.45
        elif color_choice == 'G_yellow':
            base = (255, 245, 200)
            intrinsic_brightness = 0.35
        elif color_choice == 'K_gold':
            base = (255, 220, 160)
            intrinsic_brightness = 0.25
        else:  # M_red
            base = (255, 190, 140)
            intrinsic_brightness = 0.15
        
        # Brightness scales with size (bigger apparent size = brighter)
        # and intrinsic brightness (spectral type)
        size_factor = (size - 0.3) / 1.7  # Normalize size to 0-1 range
        brightness = intrinsic_brightness * (0.4 + 0.6 * size_factor)
        # Add slight random variation
        brightness *= np.random.uniform(0.8, 1.0)
        brightness = min(brightness, 0.85)
        
        # Blend star color with background based on brightness
        color = tuple(int(bg_rgb[i] + (base[i] - bg_rgb[i]) * brightness) for i in range(3))
        
        r = size
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
    
    # Convert starfield to format matplotlib can use as background
    starfield_array = np.array(starfield)
    
    # Now render again with starfield as background using figimage
    plt.close('all')
    fig2 = plt.figure(figsize=(14, chart_height), facecolor=ChartColors.BACKGROUND)
    
    # Place starfield image at pixel coordinates (0,0) - this goes behind everything
    fig2.figimage(starfield_array, xo=0, yo=0, zorder=-1)
    
    # Recreate axes on top of starfield - need to rebuild the chart
    # Actually, let's just composite with PIL instead
    plt.close('all')
    
    # Re-render the chart
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='none')  # transparent bg
    buf.seek(0)
    
    chart_img = Image.open(buf).convert('RGBA')
    starfield_rgba = starfield.convert('RGBA')
    
    # Composite chart on top of starfield
    result = Image.alpha_composite(starfield_rgba, chart_img)
    
    buf = io.BytesIO()
    result.save(buf, format='PNG')
    buf.seek(0)
    plt.close()
    
    # Encode to base64
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return image_base64

def main():
    print("=" * 60)
    print("NIGHT SKY ALERT SYSTEM")
    print("=" * 60)
    print(f"\n--- Configuration ---")
    print(f"Location: {settings.latitude}, {settings.longitude}")
    
    # Get location name from coordinates
    print(f"\n--- Reverse Geocoding ---")
    location_name = get_location_name(settings.latitude, settings.longitude)
    print(f"Location: {location_name}")
    
    print(f"Min viewing hours: {settings.min_viewing_hours}")
    print(f"Cloud cover limit: {settings.cloud_cover_limit}%")
    print(f"Precipitation limit: {settings.precip_prob_limit}%")
    print(f"Check interval: {settings.check_interval_minutes} minutes")
    print(f"Moon illumination range: {settings.min_moon_illumination*100:.0f}%-{settings.max_moon_illumination*100:.0f}%")
    
    # Validate custom time windows against sunset/sunrise
    validate_time_window_against_sun()
    
    # 1. Fetch Weather & Determine Timezone (Open-Meteo handles auto-timezone)
    print(f"\n--- Fetching Weather Data ---")
    weather_data = get_weather_forecast(settings.latitude, settings.longitude)
    
    if not weather_data:
        print("ERROR: Failed to fetch weather data")
        return
    
    timezone_str = weather_data.get('timezone', 'UTC')
    try:
        local_tz = pytz.timezone(timezone_str)
    except pytz.UnknownTimeZoneError:
        print(f"Unknown timezone {timezone_str}, defaulting to UTC")
        local_tz = pytz.utc
        
    print(f"Timezone: {local_tz.zone}")

    # 2. Determine Time Window (Tonight)
    now = datetime.datetime.now(local_tz)
    
    sunset_dt, sunrise_dt = calculate_viewing_window(
        now, 
        settings.latitude, 
        settings.longitude,
        settings.start_time,
        settings.end_time
    )

    print(f"\n--- Time Window ---")
    print(f"Start: {sunset_dt.strftime('%Y-%m-%d %I:%M %p')}")
    print(f"End:   {sunrise_dt.strftime('%Y-%m-%d %I:%M %p')}")
    window_hours = (sunrise_dt - sunset_dt).total_seconds() / 3600
    print(f"Duration: {window_hours:.1f} hours")
    
    # Load Skyfield data for body calculations
    ts = load.timescale()
    eph = load('de421.bsp')
    observer = wgs84.latlon(settings.latitude, settings.longitude)

    # 3. Process Weather Data
    hourly = weather_data.get('hourly', {})
    times = hourly.get('time', [])
    cloud_covers = hourly.get('cloud_cover', [])
    precip_probs = hourly.get('precipitation_probability', [])
    
    print(f"\n--- Weather Data ---")
    print(f"Received {len(times)} hourly weather records")

    good_viewing_slots = []
    all_window_slots = []  # Track all slots in the window for continuous viewing check
    weather_blocks = {}  # Track weather conditions for each hour for chart overlay

    # 4. Correlate Weather with Time Window
    print("\n--- Weather Analysis ---")
    print(f"Checking weather between {sunset_dt.strftime('%I:%M %p')} and {sunrise_dt.strftime('%I:%M %p')}")
    print(f"Limits: Cloud<={settings.cloud_cover_limit}%, Precip<={settings.precip_prob_limit}%")
    print("")
    
    good_hours = 0
    bad_hours = 0
    
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
            status = "[OK] PASS" if is_good else "[X] FAIL"
            print(f"  {dt.strftime('%I:%M %p')}: Cloud {cc:3}%, Precip {pp:3}% -> {status}")

            all_window_slots.append({'time': dt, 'is_good': is_good})
            weather_blocks[dt] = {'cloud': cc, 'precip': pp, 'good': is_good}
            if is_good:
                good_viewing_slots.append(dt)
                good_hours += 1
            else:
                bad_hours += 1
    
    print(f"\nWeather Summary: {good_hours} good hours, {bad_hours} bad hours ({good_hours}/{good_hours+bad_hours} total)")

    # 5. Check for minimum continuous viewing time
    print(f"\n--- Continuous Weather Check ---")
    print(f"Required: {settings.min_viewing_hours} continuous hour(s)")
    
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
        print(f"  Time: {start} to {end}")
    
    if continuous_hours < settings.min_viewing_hours:
        print(f"\n[X] INSUFFICIENT CONTINUOUS WEATHER")
        print(f"  Need: {settings.min_viewing_hours} hours")
        print(f"  Found: {continuous_hours} hours")
        print(f"  No alert will be sent.")
        return

    print(f"[OK] Sufficient continuous weather window!")
    print(f"  Total good hours: {len(good_viewing_slots)}")

    # 6. Generate fine-grained time slots for astronomical checks
    # We'll check astronomy every CHECK_INTERVAL_MINUTES for the ENTIRE night
    print(f"\n--- Generating {settings.check_interval_minutes}-minute interval slots ---")
    
    fine_grained_slots = []
    interval_delta = datetime.timedelta(minutes=settings.check_interval_minutes)
    
    # Generate slots for the entire night window (sunset to sunrise)
    current_time = sunset_dt
    while current_time <= sunrise_dt:
        fine_grained_slots.append(current_time)
        current_time += interval_delta
    
    print(f"Generated {len(fine_grained_slots)} time slots at {settings.check_interval_minutes}-minute intervals")
    print(f"Spanning {sunset_dt.strftime('%I:%M %p')} to {sunrise_dt.strftime('%I:%M %p')}")

    # 7. Check Astronomical Bodies for fine-grained slots
    report_lines = []
    
    # Cache body objects - map friendly names to ephemeris names
    BODY_MAPPINGS = {
        'Moon': 'moon',
        'Mars': 'mars barycenter',
        'Jupiter': 'jupiter barycenter',
        'Saturn': 'saturn barycenter'
    }
    
    bodies = {}
    for name in TARGETS:
        bodies[name] = eph[BODY_MAPPINGS[name]]
    earth = eph['earth']
    sun = eph['sun']  # Need sun for moon phase calculation

    # Initialize moon phase variables
    moon_phase_emoji = None
    moon_illumination = 0.5  # Default to half illuminated
    moon_meets_illumination = True  # Default to True if no moon check

    # Calculate moon phase (use middle of viewing window)
    if fine_grained_slots:
        mid_time = fine_grained_slots[len(fine_grained_slots) // 2]
        t_mid = ts.from_datetime(mid_time)
        
        # Calculate moon phase (fraction of lunar cycle)
        # This is the elongation between sun and moon
        moon = eph['moon']
        e = earth.at(t_mid)
        sun_pos = e.observe(sun).apparent()
        moon_pos = e.observe(moon).apparent()
        
        # Calculate phase angle (0 = new, 0.5 = full)
        # Using the difference in ecliptic longitude
        _, sun_lon, _ = sun_pos.ecliptic_latlon()
        _, moon_lon, _ = moon_pos.ecliptic_latlon()
        phase_angle = (moon_lon.degrees - sun_lon.degrees) % 360
        
        # Convert to moon phase (0-1 where 0=new, 0.5=full, 1=new)
        phase = phase_angle / 360.0
        
        # Select emoji based on phase
        if phase < 0.0625 or phase > 0.9375:
            moon_phase_emoji = '🌑'  # New moon
        elif phase < 0.1875:
            moon_phase_emoji = '🌒'  # Waxing crescent
        elif phase < 0.3125:
            moon_phase_emoji = '🌓'  # First quarter
        elif phase < 0.4375:
            moon_phase_emoji = '🌔'  # Waxing gibbous
        elif phase < 0.5625:
            moon_phase_emoji = '🌕'  # Full moon
        elif phase < 0.6875:
            moon_phase_emoji = '🌖'  # Waning gibbous
        elif phase < 0.8125:
            moon_phase_emoji = '🌗'  # Last quarter
        else:
            moon_phase_emoji = '🌘'  # Waning crescent
        
        # Calculate moon illumination (0 = new, 1 = full)
        # Illumination is approximately: (1 - cos(phase_angle_radians)) / 2
        import math
        phase_angle_radians = (phase_angle / 180.0) * math.pi
        moon_illumination = (1.0 - math.cos(phase_angle_radians)) / 2.0
        
        # Map emoji to text description
        emoji_descriptions = {
            '🌑': 'New Moon',
            '🌒': 'Waxing Crescent',
            '🌓': 'First Quarter',
            '🌔': 'Waxing Gibbous',
            '🌕': 'Full Moon',
            '🌖': 'Waning Gibbous',
            '🌗': 'Last Quarter',
            '🌘': 'Waning Crescent'
        }
        
        print(f"\nMoon Phase: {emoji_descriptions.get(moon_phase_emoji, 'Unknown')} ({phase*100:.1f}% of cycle)")
        print(f"Moon Illumination: {moon_illumination*100:.1f}% (Config allows {settings.min_moon_illumination*100:.0f}%-{settings.max_moon_illumination*100:.0f}%)")
        
        # Check if moon meets illumination requirements
        moon_meets_illumination = settings.min_moon_illumination <= moon_illumination <= settings.max_moon_illumination
        if not moon_meets_illumination:
            print(f"[!] Moon does not meet illumination requirements - will be excluded from visibility checks")

    # We will group by body to make the report readable
    # Track ALL visibility data (for charting), and separately track good weather times
    body_visibility = {name: [] for name in TARGETS}
    body_slots_all = {name: [] for name in TARGETS}  # ALL times body is visible
    body_slots_good_weather = {name: [] for name in TARGETS}  # Only times with good weather

    print("\n--- Astronomical Check (Fine-Grained Intervals) ---")
    for dt in fine_grained_slots:
        # Check if this time has good weather (find nearest hourly slot)
        nearest_hour = dt.replace(minute=0, second=0, microsecond=0)
        has_good_weather = nearest_hour in weather_blocks and weather_blocks[nearest_hour]['good']
        
        t = ts.from_datetime(dt)
        observer_loc = earth + observer
        
        # Only print every hour to avoid spam
        if dt.minute == 0 or dt == fine_grained_slots[0]:
            weather_status = "GOOD" if has_good_weather else "BAD"
            print(f"Time: {dt.strftime('%I:%M %p')} (Weather: {weather_status})")
        
        for name, body in bodies.items():
            # Skip Moon if it doesn't meet illumination requirements
            if name == 'Moon' and not moon_meets_illumination:
                continue
                
            astrometric = observer_loc.at(t).observe(body)
            alt, az, distance = astrometric.apparent().altaz()
            
            alt_degrees = alt.degrees
            
            if dt.minute == 0 or dt == fine_grained_slots[0]:
                print(f"  {name}: Alt {alt_degrees:.2f}°, Az {az.degrees:.2f}°")
            
            if alt_degrees > 0:
                is_high = alt_degrees >= 15
                body_visibility[name].append({
                    'time': dt,
                    'alt': alt_degrees,
                    'high': is_high
                })
                body_slots_all[name].append(dt)
                
                # Only add to good weather list if weather is acceptable
                if has_good_weather:
                    body_slots_good_weather[name].append(dt)

    # 8. Check continuous viewing time for each visible body (using GOOD WEATHER times only)
    print("\n--- Body Visibility Analysis (Good Weather Only) ---")
    bodies_meeting_requirement = {}
    
    for name in TARGETS:
        total_visible = len(body_slots_all[name])
        good_weather_visible = len(body_slots_good_weather[name])
        
        print(f"\n{name}:")
        print(f"  Total visible slots: {total_visible}")
        print(f"  Good weather slots: {good_weather_visible}")
        
        if not body_slots_good_weather[name]:
            print(f"  [X] Not visible above horizon during good weather")
            continue
        
        # Find longest continuous block for this body during good weather
        sorted_times = sorted(body_slots_good_weather[name])
        longest_block = []
        current_block = [sorted_times[0]]
        
        interval_hours = settings.check_interval_minutes / 60.0
        max_gap = interval_hours * 1.5  # Allow small gaps
        
        for i in range(1, len(sorted_times)):
            # Check if this slot is consecutive
            time_diff = (sorted_times[i] - sorted_times[i-1]).total_seconds() / 3600
            if time_diff <= max_gap:
                current_block.append(sorted_times[i])
            else:
                if len(current_block) > len(longest_block):
                    longest_block = current_block.copy()
                current_block = [sorted_times[i]]
        
        # Check final block
        if len(current_block) > len(longest_block):
            longest_block = current_block.copy()
        
        # Convert block length to hours
        continuous_hours_for_body = len(longest_block) * interval_hours
        
        if continuous_hours_for_body >= settings.min_viewing_hours:
            start = longest_block[0].strftime('%I:%M %p')
            end = longest_block[-1].strftime('%I:%M %p')
            print(f"  [OK] Meets requirement: {continuous_hours_for_body:.2f} hours ({start}-{end})")
            bodies_meeting_requirement[name] = longest_block
        else:
            print(f"  [X] Only {continuous_hours_for_body:.2f} hours continuous (need {settings.min_viewing_hours})")
    
    print(f"\n--- Bodies Meeting Requirements ---")
    print(f"Total: {len(bodies_meeting_requirement)} of {len(TARGETS)}")
    for name in bodies_meeting_requirement.keys():
        print(f"  [OK] {name}")
    
    if not bodies_meeting_requirement:
        print(f"\n[X] INSUFFICIENT ASTRONOMICAL VISIBILITY")
        print(f"  No bodies meet the {settings.min_viewing_hours} hour continuous viewing requirement")
        print(f"  No alert will be sent.")
        return

    # 8. Create visibility grid and find best viewing blocks
    print("\n--- Analyzing Best Viewing Blocks ---")
    
    # Build a time-indexed grid for ALL fine-grained time slots
    # This will be used for the chart (shows everything)
    visibility_grid = {}
    for dt in fine_grained_slots:
        visible_bodies = []
        for name in TARGETS:
            if dt in body_slots_all[name]:
                visible_bodies.append(name)
        visibility_grid[dt] = {
            'bodies': visible_bodies,
            'count': len(visible_bodies)
        }
    
    # Build a separate list of times that meet the requirements (for alerts)
    all_qualified_times = set()
    for times_list in bodies_meeting_requirement.values():
        all_qualified_times.update(times_list)
    
    sorted_all_times = sorted(all_qualified_times)
    
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


    # 10. Construct Notification
    best_block = viewing_blocks[0]
    best_start = best_block['start'].strftime('%I:%M %p')
    best_end = best_block['end'].strftime('%I:%M %p')
    
    # Simple message: just the header and time interval
    message_parts = [
        "Good night sky viewing! Go outside and look up! 🌌✨",
        f"{best_start} - {best_end}"
    ]
    
    # Generate visual chart image (use ALL fine-grained slots, not just qualified times)
    print("\n--- Generating Visibility Chart Image ---")
    
    # Filter targets to only include bodies visible across the ENTIRE best block
    # A body must be visible at EVERY time slot in the best block to be included in the chart
    filtered_targets = []
    for name in TARGETS:
        if name not in bodies_meeting_requirement:
            continue
        
        # Check if this body is visible at every time in the best block
        visible_at_all_times = all(
            name in visibility_grid[dt]['bodies']
            for dt in best_block['times']
        )
        
        if visible_at_all_times:
            filtered_targets.append(name)
            print(f"  Including {name} (visible entire block)")
        else:
            print(f"  Excluding {name} (not visible entire block)")
    
    # If no bodies are visible for the entire block, fall back to bodies meeting requirements
    if not filtered_targets:
        print("  Note: No bodies visible entire block, using all bodies meeting requirements")
        filtered_targets = list(bodies_meeting_requirement.keys())
    
    # Format date for chart title
    if sorted_all_times:
        chart_date = sorted_all_times[0].strftime('%B %d, %Y')  # e.g., "November 24, 2025"
    else:
        chart_date = None
    
    image_data = create_visibility_chart(
        fine_grained_slots,  # Use ALL time slots for full granularity
        visibility_grid, 
        body_visibility, 
        filtered_targets,  # Use filtered list instead of TARGETS
        continuous_blocks=bodies_meeting_requirement,
        weather_blocks=weather_blocks,
        moon_phase_emoji=moon_phase_emoji,
        location_name=location_name,
        date_str=chart_date
    )
    print(f"Chart generated ({len(image_data)} bytes base64)")

    full_message = "\n".join(message_parts)
    print("\n--- Sending Notification ---")
    # Print message without emojis for Windows console compatibility
    print(full_message.encode('ascii', errors='replace').decode('ascii'))
    send_pushover(full_message, image_data=image_data)

if __name__ == "__main__":
    main()

