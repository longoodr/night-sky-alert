"""Tests for Settings configuration, especially GitHub Actions empty string handling."""
import pytest
from unittest.mock import patch, MagicMock


class TestSettingsEmptyStringHandling:
    """Test that Settings correctly handles empty strings from GitHub Actions.
    
    GitHub Actions passes empty strings instead of null for unset environment
    variables. These tests verify that empty strings are properly normalized
    to use field defaults.
    """

    @pytest.fixture
    def mock_geocode(self):
        """Mock geocode_location to avoid network calls."""
        with patch('main.geocode_location') as mock:
            mock.return_value = (28.5383, -81.3792, "Orlando, Florida")
            yield mock

    def test_empty_strings_use_defaults_with_location(self, mock_geocode):
        """Empty strings for optional fields should use Field defaults when location is provided."""
        from main import Settings
        
        # Simulate GitHub Actions passing empty strings for all optional fields
        settings = Settings(
            location="Orlando, Florida",
            latitude="",
            longitude="",
            cloud_cover_limit="",
            precip_prob_limit="",
            min_viewing_hours="",
            check_interval_minutes="",
            min_moon_illumination="",
            max_moon_illumination="",
            pushover_user_key="",
            pushover_api_token="",
            start_time="",
            end_time="",
        )
        
        # Verify defaults are applied
        assert settings.cloud_cover_limit == 15.0
        assert settings.precip_prob_limit == 5.0
        assert settings.min_viewing_hours == 1.0
        assert settings.check_interval_minutes == 15
        assert settings.min_moon_illumination == 0.0
        assert settings.max_moon_illumination == 1.0
        
        # Verify optional fields are None
        assert settings.pushover_user_key is None
        assert settings.pushover_api_token is None
        assert settings.start_time is None
        assert settings.end_time is None

    def test_empty_strings_use_defaults_with_coordinates(self):
        """Empty strings for optional fields should use Field defaults when coordinates are provided."""
        from main import Settings
        
        settings = Settings(
            location="",
            latitude=28.5383,
            longitude=-81.3792,
            cloud_cover_limit="",
            precip_prob_limit="",
            min_viewing_hours="",
            check_interval_minutes="",
            min_moon_illumination="",
            max_moon_illumination="",
            pushover_user_key="",
            pushover_api_token="",
            start_time="",
            end_time="",
        )
        
        # Verify defaults are applied
        assert settings.cloud_cover_limit == 15.0
        assert settings.precip_prob_limit == 5.0
        assert settings.min_viewing_hours == 1.0
        assert settings.check_interval_minutes == 15
        assert settings.min_moon_illumination == 0.0
        assert settings.max_moon_illumination == 1.0
        
        # Verify coordinates are preserved
        assert settings.latitude == 28.5383
        assert settings.longitude == -81.3792

    def test_explicit_values_override_defaults(self, mock_geocode):
        """Explicitly provided values should override defaults."""
        from main import Settings
        
        settings = Settings(
            location="Orlando, Florida",
            cloud_cover_limit=25.0,
            precip_prob_limit=10.0,
            min_viewing_hours=2.0,
            check_interval_minutes=30,
            min_moon_illumination=0.2,
            max_moon_illumination=0.8,
            pushover_user_key="my_user_key",
            pushover_api_token="my_api_token",
            start_time="21:00",
            end_time="05:00",
        )
        
        assert settings.cloud_cover_limit == 25.0
        assert settings.precip_prob_limit == 10.0
        assert settings.min_viewing_hours == 2.0
        assert settings.check_interval_minutes == 30
        assert settings.min_moon_illumination == 0.2
        assert settings.max_moon_illumination == 0.8
        assert settings.pushover_user_key == "my_user_key"
        assert settings.pushover_api_token == "my_api_token"
        assert settings.start_time == "21:00"
        assert settings.end_time == "05:00"

    def test_mixed_empty_and_explicit_values(self, mock_geocode):
        """Mix of empty strings and explicit values should work correctly."""
        from main import Settings
        
        settings = Settings(
            location="Orlando, Florida",
            cloud_cover_limit=25.0,  # explicit
            precip_prob_limit="",     # empty -> default
            min_viewing_hours=2.0,    # explicit
            check_interval_minutes="", # empty -> default
            min_moon_illumination="",  # empty -> default
            max_moon_illumination=0.5, # explicit
            pushover_user_key="my_key", # explicit
            pushover_api_token="",      # empty -> None
            start_time="",              # empty -> None
            end_time="05:00",           # explicit
        )
        
        assert settings.cloud_cover_limit == 25.0
        assert settings.precip_prob_limit == 5.0  # default
        assert settings.min_viewing_hours == 2.0
        assert settings.check_interval_minutes == 15  # default
        assert settings.min_moon_illumination == 0.0  # default
        assert settings.max_moon_illumination == 0.5
        assert settings.pushover_user_key == "my_key"
        assert settings.pushover_api_token is None  # default
        assert settings.start_time is None  # default
        assert settings.end_time == "05:00"

    def test_no_location_or_coords_raises_error(self):
        """Should raise error when neither location nor coordinates provided."""
        from main import Settings
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                location="",
                latitude="",
                longitude="",
            )
        
        assert "Either 'location' OR both 'latitude' and 'longitude'" in str(exc_info.value)

    def test_partial_coordinates_raises_error(self):
        """Should raise error when only one coordinate is provided."""
        from main import Settings
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                location="",
                latitude=28.5383,
                longitude="",
            )
        
        assert "Either 'location' OR both 'latitude' and 'longitude'" in str(exc_info.value)


class TestNormalizeEmptyStringHelper:
    """Test the normalize_empty_string helper function."""

    def test_empty_string_returns_default(self):
        """Empty string should return the default value."""
        from main import normalize_empty_string
        
        assert normalize_empty_string("", default=15.0) == 15.0
        assert normalize_empty_string("", default=None) is None
        assert normalize_empty_string("", default="fallback") == "fallback"

    def test_none_returns_default(self):
        """None should return the default value."""
        from main import normalize_empty_string
        
        assert normalize_empty_string(None, default=15.0) == 15.0
        assert normalize_empty_string(None, default=None) is None

    def test_non_empty_value_preserved(self):
        """Non-empty values should be returned as-is."""
        from main import normalize_empty_string
        
        assert normalize_empty_string("hello", default=None) == "hello"
        assert normalize_empty_string(25.0, default=15.0) == 25.0
        assert normalize_empty_string(0, default=10) == 0
        assert normalize_empty_string(0.0, default=1.0) == 0.0
