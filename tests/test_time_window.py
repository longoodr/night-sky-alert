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
    
    start_dt, end_dt = calculate_viewing_window(now, NYC_LAT, NYC_LON)
    
    assert start_dt.day == expected_start_date
    assert 16 <= start_dt.hour <= 17
    assert end_dt.hour == expected_end_hour


@pytest.mark.parametrize("end_time_str,expected_end_hour,expected_end_day", [
    ("23:00", 23, 26),
    ("02:00", 2, 27),
    ("06:30", 6, 27),
])
def test_custom_end_time(end_time_str, expected_end_hour, expected_end_day):
    now = TZ.localize(datetime.datetime(2025, 11, 26, 15, 0, 0))
    
    start_dt, end_dt = calculate_viewing_window(
        now, NYC_LAT, NYC_LON, end_time_str=end_time_str
    )
    
    assert end_dt.hour == expected_end_hour
    assert end_dt.day == expected_end_day


def test_custom_start_time():
    now = TZ.localize(datetime.datetime(2025, 11, 26, 15, 0, 0))
    
    start_dt, end_dt = calculate_viewing_window(
        now, NYC_LAT, NYC_LON, start_time_str="20:00"
    )
    
    assert start_dt.hour == 20
    assert start_dt.minute == 0


def test_end_always_after_start():
    for hour in [0, 1, 6, 12, 15, 18, 21, 23]:
        now = TZ.localize(datetime.datetime(2025, 11, 26, hour, 0, 0))
        
        start_dt, end_dt = calculate_viewing_window(now, NYC_LAT, NYC_LON)
        
        assert end_dt > start_dt


def test_default_end_is_midnight():
    now = TZ.localize(datetime.datetime(2025, 11, 26, 15, 0, 0))
    
    start_dt, end_dt = calculate_viewing_window(now, NYC_LAT, NYC_LON)
    
    assert end_dt.hour == 0
    assert end_dt.minute == 0


def test_no_daytime_window():
    now = TZ.localize(datetime.datetime(2025, 11, 26, 1, 0, 0))
    
    start_dt, end_dt = calculate_viewing_window(now, NYC_LAT, NYC_LON)
    
    assert not (6 <= start_dt.hour <= 7)
