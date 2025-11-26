import datetime
import pytest
import pytz
import sys
import os

from main import calculate_viewing_window

NYC_LAT = 40.7128
NYC_LON = -74.0060
TZ = pytz.timezone('America/New_York')


@pytest.mark.parametrize("hour,expected_start_date,expected_end_hour", [
    (1, 25, 0),
    (12, 26, 0),
    (15, 26, 0),
    (21, 26, 0),
])
def test_viewing_window_by_time_of_day(hour, expected_start_date, expected_end_hour):
    now = TZ.localize(datetime.datetime(2025, 11, 26, hour, 0, 0))
    
    search_start, search_end, astro_sunset, astro_sunrise = calculate_viewing_window(now, NYC_LAT, NYC_LON)
    
    assert search_start.day == expected_start_date
    assert 16 <= search_start.hour <= 17
    assert search_end.hour == expected_end_hour
    # Verify astronomical limits are returned
    assert astro_sunset is not None
    assert astro_sunrise is not None
    assert astro_sunrise > astro_sunset


@pytest.mark.parametrize("end_time_str,expected_end_hour,expected_end_day", [
    ("23:00", 23, 26),
    ("02:00", 2, 27),
    ("06:30", 6, 27),
])
def test_custom_end_time(end_time_str, expected_end_hour, expected_end_day):
    now = TZ.localize(datetime.datetime(2025, 11, 26, 15, 0, 0))
    
    search_start, search_end, astro_sunset, astro_sunrise = calculate_viewing_window(
        now, NYC_LAT, NYC_LON, end_time_str=end_time_str
    )
    
    assert search_end.hour == expected_end_hour
    assert search_end.day == expected_end_day


def test_custom_start_time():
    now = TZ.localize(datetime.datetime(2025, 11, 26, 15, 0, 0))
    
    search_start, search_end, astro_sunset, astro_sunrise = calculate_viewing_window(
        now, NYC_LAT, NYC_LON, start_time_str="20:00"
    )
    
    assert search_start.hour == 20
    assert search_start.minute == 0


def test_end_always_after_start():
    for hour in [0, 1, 6, 12, 15, 18, 21, 23]:
        now = TZ.localize(datetime.datetime(2025, 11, 26, hour, 0, 0))
        
        search_start, search_end, astro_sunset, astro_sunrise = calculate_viewing_window(now, NYC_LAT, NYC_LON)
        
        assert search_end > search_start


def test_default_end_is_midnight():
    now = TZ.localize(datetime.datetime(2025, 11, 26, 15, 0, 0))
    
    search_start, search_end, astro_sunset, astro_sunrise = calculate_viewing_window(now, NYC_LAT, NYC_LON)
    
    assert search_end.hour == 0
    assert search_end.minute == 0


def test_no_daytime_window():
    now = TZ.localize(datetime.datetime(2025, 11, 26, 1, 0, 0))
    
    search_start, search_end, astro_sunset, astro_sunrise = calculate_viewing_window(now, NYC_LAT, NYC_LON)
    
    assert not (6 <= search_start.hour <= 7)


def test_astronomical_limits_returned_with_custom_window():
    """Test that astronomical limits are always returned, even when custom times are used."""
    now = TZ.localize(datetime.datetime(2025, 11, 26, 15, 0, 0))
    
    # Set a custom narrow window (8PM to 11PM)
    search_start, search_end, astro_sunset, astro_sunrise = calculate_viewing_window(
        now, NYC_LAT, NYC_LON, start_time_str="20:00", end_time_str="23:00"
    )
    
    # Custom window should be respected
    assert search_start.hour == 20
    assert search_end.hour == 23
    
    # But astronomical limits should extend beyond
    assert astro_sunset.hour < 20  # Sunset is before 8PM in late November
    assert astro_sunrise.hour >= 6   # Sunrise is around 6-7AM


def test_search_window_clamped_to_astronomical_limits():
    """Test that search window cannot extend beyond astronomical limits."""
    now = TZ.localize(datetime.datetime(2025, 11, 26, 15, 0, 0))
    
    # Try to set window that goes past sunrise (sunrise is ~7AM in late Nov)
    search_start, search_end, astro_sunset, astro_sunrise = calculate_viewing_window(
        now, NYC_LAT, NYC_LON, end_time_str="08:00"  # 8AM is after sunrise
    )
    
    # Search end should be clamped to astronomical sunrise
    assert search_end <= astro_sunrise


def test_flood_fill_preserves_astronomical_bounds():
    """Test that flood-fill expansion respects hard astronomical limits."""
    now = TZ.localize(datetime.datetime(2025, 11, 26, 15, 0, 0))
    
    search_start, search_end, astro_sunset, astro_sunrise = calculate_viewing_window(
        now, NYC_LAT, NYC_LON
    )
    
    # Astronomical bounds should never be exceeded
    assert search_start >= astro_sunset
    assert search_end <= astro_sunrise
