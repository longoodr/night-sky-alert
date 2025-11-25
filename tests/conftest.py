import pytest
import datetime
import pytz
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture
def mock_settings():
    """Mock settings for testing"""
    with patch('main.settings') as mock:
        mock.latitude = 28.661111
        mock.longitude = -81.365619
        mock.cloud_cover_limit = 15.0
        mock.precip_prob_limit = 5.0
        mock.min_viewing_hours = 1.0
        mock.check_interval_minutes = 1
        mock.pushover_user_key = "test_user_key"
        mock.pushover_api_token = "test_api_token"
        mock.start_time = None
        mock.end_time = None
        yield mock

@pytest.fixture
def mock_weather_data_good():
    """Mock weather data with good conditions"""
    return {
        'timezone': 'America/New_York',
        'hourly': {
            'time': [
                '2025-11-24T20:00:00',
                '2025-11-24T21:00:00',
                '2025-11-24T22:00:00',
                '2025-11-24T23:00:00',
                '2025-11-25T00:00:00',
                '2025-11-25T01:00:00',
                '2025-11-25T02:00:00',
                '2025-11-25T03:00:00',
            ],
            'cloud_cover': [10, 5, 8, 12, 10, 15, 20, 25],
            'precipitation_probability': [0, 0, 2, 0, 5, 3, 10, 15],
            'visibility': [10000] * 8
        }
    }

@pytest.fixture
def mock_weather_data_mixed():
    """Mock weather data with mixed conditions"""
    return {
        'timezone': 'America/New_York',
        'hourly': {
            'time': [
                '2025-11-24T20:00:00',
                '2025-11-24T21:00:00',
                '2025-11-24T22:00:00',
                '2025-11-24T23:00:00',
                '2025-11-25T00:00:00',
                '2025-11-25T01:00:00',
                '2025-11-25T02:00:00',
                '2025-11-25T03:00:00',
            ],
            'cloud_cover': [10, 25, 8, 30, 10, 5, 20, 10],
            'precipitation_probability': [0, 10, 2, 0, 5, 3, 15, 2],
            'visibility': [10000] * 8
        }
    }

@pytest.fixture
def mock_weather_data_poor():
    """Mock weather data with poor conditions"""
    return {
        'timezone': 'America/New_York',
        'hourly': {
            'time': [
                '2025-11-24T20:00:00',
                '2025-11-24T21:00:00',
                '2025-11-24T22:00:00',
                '2025-11-24T23:00:00',
            ],
            'cloud_cover': [80, 90, 75, 85],
            'precipitation_probability': [60, 70, 50, 65],
            'visibility': [2000] * 4
        }
    }

@pytest.fixture
def mock_skyfield_data():
    """Mock Skyfield ephemeris and astronomical calculations"""
    with patch('main.load') as mock_load:
        # Mock timescale
        mock_ts = MagicMock()
        mock_load.timescale.return_value = mock_ts
        
        # Mock ephemeris
        mock_eph = MagicMock()
        mock_load.return_value = mock_eph
        
        # Mock observer
        mock_observer = MagicMock()
        
        # Mock sun events (sunset/sunrise)
        eastern = pytz.timezone('America/New_York')
        sunset = eastern.localize(datetime.datetime(2025, 11, 24, 18, 30))
        sunrise = eastern.localize(datetime.datetime(2025, 11, 25, 6, 45))
        
        # Mock almanac find_discrete
        with patch('main.almanac.find_discrete') as mock_find:
            mock_time_sunset = MagicMock()
            mock_time_sunset.astimezone.return_value = sunset
            mock_time_sunrise = MagicMock()
            mock_time_sunrise.astimezone.return_value = sunrise
            
            mock_find.return_value = (
                [mock_time_sunset, mock_time_sunrise],
                [1, 0]  # 1=sunset, 0=sunrise
            )
            
            yield {
                'load': mock_load,
                'ts': mock_ts,
                'eph': mock_eph,
                'observer': mock_observer,
                'sunset': sunset,
                'sunrise': sunrise
            }

def create_mock_body_position(altitude, azimuth=180):
    """Helper to create mock body position"""
    mock_astrometric = MagicMock()
    mock_apparent = MagicMock()
    mock_altaz = MagicMock()
    
    mock_alt = MagicMock()
    mock_alt.degrees = altitude
    mock_az = MagicMock()
    mock_az.degrees = azimuth
    
    mock_altaz.return_value = (mock_alt, mock_az, MagicMock())
    mock_apparent.altaz.return_value = mock_altaz.return_value
    mock_astrometric.apparent.return_value = mock_apparent
    
    return mock_astrometric

@pytest.fixture
def mock_body_positions():
    """Mock positions for Moon, Mars, Jupiter, Saturn with sub-hour variation"""
    def get_position(body_name, time_hour, time_minute=0):
        """Return altitude based on body, hour, and minute (for smooth gradients)"""
        # Base positions per hour
        positions = {
            'Moon': {20: 25, 21: 30, 22: 35, 23: 40, 0: 38, 1: 35, 2: 30, 3: 25},
            'Mars': {20: 5, 21: 8, 22: 12, 23: 15, 0: 18, 1: 20, 2: 18, 3: 15},
            'Jupiter': {20: 40, 21: 42, 22: 45, 23: 47, 0: 45, 1: 42, 2: 38, 3: 35},
            'Saturn': {20: -5, 21: -3, 22: 2, 23: 5, 0: 8, 1: 10, 2: 8, 3: 5}
        }
        
        base_alt = positions.get(body_name, {}).get(time_hour, -10)
        
        # Add sub-hour variation based on minutes (smooth transition within hour)
        # Each 15 minutes represents ~1/4 progress to next hour
        next_hour = (time_hour + 1) % 24
        next_alt = positions.get(body_name, {}).get(next_hour, base_alt)
        
        # Linear interpolation based on minutes
        minute_fraction = time_minute / 60.0
        interpolated_alt = base_alt + (next_alt - base_alt) * minute_fraction
        
        return interpolated_alt
    
    return get_position
