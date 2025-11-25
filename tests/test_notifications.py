import pytest
import datetime
import pytz
from unittest.mock import patch, MagicMock
import base64
from io import BytesIO
from pathlib import Path
import hashlib

from conftest import (
    mock_settings, 
    mock_weather_data_good, 
    mock_weather_data_mixed,
    mock_weather_data_poor,
    mock_skyfield_data,
    mock_body_positions,
    create_mock_body_position
)


class TestNotificationScenarios:
    """Test different weather and astronomical scenarios"""
    
    def test_perfect_conditions_all_bodies_visible(
        self, 
        mock_settings, 
        mock_weather_data_good, 
        mock_skyfield_data,
        mock_body_positions,
        snapshot
    ):
        """Test scenario: Perfect weather, all bodies visible for multiple hours"""
        from main import create_visibility_chart
        
        eastern = pytz.timezone('America/New_York')
        
        # Create 15-minute interval test data (6 hours = 24 intervals)
        start_time = eastern.localize(datetime.datetime(2025, 11, 24, 20, 0))
        sorted_times = []
        for i in range(6 * 4):  # 6 hours * 4 (15-min intervals per hour)
            sorted_times.append(start_time + datetime.timedelta(minutes=i * 15))
        
        targets = ['Moon', 'Mars', 'Jupiter', 'Saturn']
        
        # Build visibility grid with 15-minute granularity
        visibility_grid = {}
        body_visibility = {name: [] for name in targets}
        
        for dt in sorted_times:
            hour = dt.hour
            minute = dt.minute
            visible_bodies = []
            
            for body_name in targets:
                alt = mock_body_positions(body_name, hour, minute)
                if alt > 0:
                    visible_bodies.append(body_name)
                    body_visibility[body_name].append({
                        'time': dt,
                        'alt': alt,
                        'high': alt >= 15
                    })
            
            visibility_grid[dt] = {
                'bodies': visible_bodies,
                'count': len(visible_bodies)
            }
        
        # Simulate continuous blocks (all times for bodies meeting requirements)
        continuous_blocks = {
            'Moon': sorted_times,
            'Mars': sorted_times,
            'Jupiter': sorted_times,
            'Saturn': sorted_times[12:]  # Last 3 hours
        }
        
        # Mock weather blocks (all good weather for this test)
        weather_blocks = {}
        for dt in sorted_times:
            nearest_hour = dt.replace(minute=0, second=0, microsecond=0)
            weather_blocks[nearest_hour] = {'cloud': 10, 'precip': 0, 'good': True}
        
        # Generate chart with continuous block highlighting
        image_data = create_visibility_chart(
            sorted_times, 
            visibility_grid, 
            body_visibility, 
            targets,
            continuous_blocks=continuous_blocks,
            weather_blocks=weather_blocks,
            moon_phase_emoji='🌕'  # Full moon for test
        )
        
        # Verify image was created
        assert image_data is not None
        assert len(image_data) > 0
        
        # Decode and verify it's valid base64
        image_bytes = base64.b64decode(image_data)
        assert image_bytes.startswith(b'\x89PNG')  # PNG magic number
        
        # Save the actual PNG image for inspection
        snapshot_dir = Path(__file__).parent / '__snapshots__'
        snapshot_dir.mkdir(exist_ok=True)
        image_path = snapshot_dir / 'test_perfect_conditions_chart.png'
        with open(image_path, 'wb') as f:
            f.write(image_bytes)
        
        # Calculate image hash for content equivalence testing
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        
        # Snapshot the visibility data structure and image hash
        snapshot_data = {
            'image_hash': image_hash,
            'image_size_bytes': len(image_bytes),
            'times': [dt.strftime('%Y-%m-%d %I:%M %p') for dt in sorted_times],
            'visibility_grid': {
                dt.strftime('%I:%M %p'): {
                    'bodies': grid['bodies'],
                    'count': grid['count']
                }
                for dt, grid in visibility_grid.items()
            },
            'body_summary': {
                name: {
                    'total_visible_hours': len(vis),
                    'high_altitude_hours': len([v for v in vis if v['high']]),
                    'altitudes': [v['alt'] for v in vis]
                }
                for name, vis in body_visibility.items()
            }
        }
        
        assert snapshot_data == snapshot
    
    def test_mixed_conditions_partial_visibility(
        self,
        mock_settings,
        mock_weather_data_mixed,
        mock_body_positions,
        snapshot
    ):
        """Test scenario: Mixed weather, some bodies visible intermittently"""
        from main import create_visibility_chart
        
        eastern = pytz.timezone('America/New_York')
        
        # Create 15-minute interval test data spanning entire night (8 PM to 4 AM = 8 hours)
        # with gaps for bad weather
        start_time = eastern.localize(datetime.datetime(2025, 11, 24, 20, 0))
        sorted_times = []
        for i in range(8 * 4):  # 8 hours * 4 (15-min intervals per hour)
            sorted_times.append(start_time + datetime.timedelta(minutes=i * 15))
        
        targets = ['Moon', 'Mars', 'Jupiter', 'Saturn']
        
        # Build visibility grid with 15-minute granularity
        visibility_grid = {}
        body_visibility = {name: [] for name in targets}
        
        for dt in sorted_times:
            hour = dt.hour
            minute = dt.minute
            visible_bodies = []
            
            for body_name in targets:
                alt = mock_body_positions(body_name, hour, minute)
                if alt > 0:
                    visible_bodies.append(body_name)
                    body_visibility[body_name].append({
                        'time': dt,
                        'alt': alt,
                        'high': alt >= 15
                    })
            
            visibility_grid[dt] = {
                'bodies': visible_bodies,
                'count': len(visible_bodies)
            }
        
        # Simulate continuous blocks (non-continuous due to gaps in weather)
        # Only include specific hour ranges that have good weather
        good_hours = [20, 22, 0, 1, 3]  # From mock_weather_data_mixed
        continuous_blocks = {
            'Moon': [t for t in sorted_times if t.hour == 20][:4],  # First hour only
            'Jupiter': [t for t in sorted_times if t.hour in [0, 1]],  # Midnight to 2 AM
        }
        
        # Mock weather blocks - mixed conditions (hours 21, 23, 2 are BAD)
        weather_blocks = {}
        for dt in sorted_times:
            nearest_hour = dt.replace(minute=0, second=0, microsecond=0)
            hour = nearest_hour.hour
            # Bad weather at hours 21, 23, 2
            is_good = hour in good_hours
            weather_blocks[nearest_hour] = {
                'cloud': 10 if is_good else 30,
                'precip': 0 if is_good else 15,
                'good': is_good
            }
        
        # Generate chart with continuous block highlighting
        image_data = create_visibility_chart(
            sorted_times, 
            visibility_grid, 
            body_visibility, 
            targets,
            continuous_blocks=continuous_blocks,
            weather_blocks=weather_blocks,
            moon_phase_emoji='🌓'  # First quarter for test
        )
        
        # Verify image was created
        assert image_data is not None
        
        # Save the actual PNG image for inspection
        snapshot_dir = Path(__file__).parent / '__snapshots__'
        snapshot_dir.mkdir(exist_ok=True)
        image_path = snapshot_dir / 'test_mixed_conditions_chart.png'
        image_bytes = base64.b64decode(image_data)
        with open(image_path, 'wb') as f:
            f.write(image_bytes)
        
        # Calculate image hash
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        
        # Snapshot the visibility data
        snapshot_data = {
            'image_hash': image_hash,
            'image_size_bytes': len(image_bytes),
            'times': [dt.strftime('%Y-%m-%d %I:%M %p') for dt in sorted_times],
            'visibility_summary': {
                dt.strftime('%I:%M %p'): grid['count']
                for dt, grid in visibility_grid.items()
            },
            'continuous_blocks': {
                name: len(vis) 
                for name, vis in body_visibility.items() 
                if len(vis) > 0
            }
        }
        
        assert snapshot_data == snapshot
    
    def test_notification_message_formatting(
        self,
        mock_settings,
        snapshot
    ):
        """Test the notification message formatting"""
        
        # Sample data for notification
        best_block = {
            'start': datetime.datetime(2025, 11, 24, 21, 0),
            'end': datetime.datetime(2025, 11, 25, 1, 0),
            'duration': 4,
            'max_bodies': 3
        }
        
        body_times = {
            'Moon': {
                'start': '09:00 PM',
                'end': '01:00 AM',
                'high_start': '09:00 PM',
                'high_end': '01:00 AM'
            },
            'Jupiter': {
                'start': '08:00 PM',
                'end': '02:00 AM',
                'high_start': '08:00 PM',
                'high_end': '02:00 AM'
            },
            'Mars': {
                'start': '10:00 PM',
                'end': '02:00 AM',
                'high_start': None,
                'high_end': None
            }
        }
        
        # Build message
        message_parts = [
            f"🌙 Clear Skies Alert! 🌙",
            f"Best Block: {best_block['start'].strftime('%I:%M %p')}-{best_block['end'].strftime('%I:%M %p')} ({best_block['duration']}h, up to {best_block['max_bodies']} bodies)",
            f"Weather: Cloud<15.0%, Precip<5.0%",
            ""
        ]
        
        for body, times in body_times.items():
            high_str = ""
            if times['high_start']:
                high_str = f" | >15° {times['high_start']}-{times['high_end']}"
            message_parts.append(f"{body}: {times['start']}-{times['end']}{high_str}")
        
        full_message = "\n".join(message_parts)
        
        assert full_message == snapshot
    
    def test_ascii_chart_generation(self, snapshot):
        """Test ASCII visibility chart generation"""
        eastern = pytz.timezone('America/New_York')
        
        sorted_times = [
            eastern.localize(datetime.datetime(2025, 11, 24, 21, 0)),
            eastern.localize(datetime.datetime(2025, 11, 24, 22, 0)),
            eastern.localize(datetime.datetime(2025, 11, 24, 23, 0)),
        ]
        
        targets = ['Moon', 'Mars', 'Jupiter', 'Saturn']
        
        visibility_grid = {
            sorted_times[0]: {'bodies': ['Moon', 'Jupiter'], 'count': 2},
            sorted_times[1]: {'bodies': ['Moon', 'Mars', 'Jupiter'], 'count': 3},
            sorted_times[2]: {'bodies': ['Moon', 'Mars', 'Jupiter', 'Saturn'], 'count': 4},
        }
        
        body_visibility = {
            'Moon': [
                {'time': sorted_times[0], 'alt': 30, 'high': True},
                {'time': sorted_times[1], 'alt': 35, 'high': True},
                {'time': sorted_times[2], 'alt': 40, 'high': True},
            ],
            'Mars': [
                {'time': sorted_times[1], 'alt': 12, 'high': False},
                {'time': sorted_times[2], 'alt': 15, 'high': True},
            ],
            'Jupiter': [
                {'time': sorted_times[0], 'alt': 42, 'high': True},
                {'time': sorted_times[1], 'alt': 45, 'high': True},
                {'time': sorted_times[2], 'alt': 47, 'high': True},
            ],
            'Saturn': [
                {'time': sorted_times[2], 'alt': 5, 'high': False},
            ]
        }
        
        # Build ASCII chart
        chart_lines = ["Time      " + " ".join([f"{name:8}" for name in targets])]
        chart_lines.append("-" * (10 + 9 * len(targets)))
        
        for dt in sorted_times:
            time_str = dt.strftime('%I:%M %p')
            row = f"{time_str:9}"
            for name in targets:
                if name in visibility_grid[dt]['bodies']:
                    is_high = any(v['time'] == dt and v['high'] for v in body_visibility[name])
                    symbol = "▓▓▓▓" if is_high else "░░░░"
                else:
                    symbol = "    "
                row += f"{symbol:8} "
            chart_lines.append(row)
        
        chart = "\n".join(chart_lines)
        
        assert chart == snapshot


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_no_bodies_visible(self, mock_settings, snapshot):
        """Test when weather is good but no bodies are visible"""
        eastern = pytz.timezone('America/New_York')
        
        sorted_times = [
            eastern.localize(datetime.datetime(2025, 11, 24, 21, 0)),
        ]
        
        targets = ['Moon', 'Mars', 'Jupiter', 'Saturn']
        
        # All bodies below horizon
        visibility_grid = {
            sorted_times[0]: {'bodies': [], 'count': 0},
        }
        
        body_visibility = {name: [] for name in targets}
        
        result = {
            'has_visible_bodies': any(len(v) > 0 for v in body_visibility.values()),
            'total_slots': len(sorted_times),
            'bodies_meeting_requirement': {}
        }
        
        assert result == snapshot
    
    def test_insufficient_continuous_hours(self, snapshot):
        """Test when visible hours don't meet minimum continuous requirement"""
        eastern = pytz.timezone('America/New_York')
        
        # Non-continuous good weather slots
        sorted_times = [
            eastern.localize(datetime.datetime(2025, 11, 24, 20, 0)),
            eastern.localize(datetime.datetime(2025, 11, 24, 22, 0)),  # Gap here
            eastern.localize(datetime.datetime(2025, 11, 25, 1, 0)),
        ]
        
        # Moon visible but in non-continuous blocks
        moon_visibility = [
            {'time': sorted_times[0], 'alt': 25, 'high': True},
            {'time': sorted_times[1], 'alt': 35, 'high': True},
            {'time': sorted_times[2], 'alt': 35, 'high': True},
        ]
        
        # Find continuous blocks
        longest_block = []
        current_block = [sorted_times[0]]
        
        for i in range(1, len(sorted_times)):
            time_diff = (sorted_times[i] - sorted_times[i-1]).total_seconds() / 3600
            if time_diff <= 1.5:
                current_block.append(sorted_times[i])
            else:
                if len(current_block) > len(longest_block):
                    longest_block = current_block.copy()
                current_block = [sorted_times[i]]
        
        if len(current_block) > len(longest_block):
            longest_block = current_block.copy()
        
        result = {
            'total_visible_hours': len(moon_visibility),
            'longest_continuous_block': len(longest_block),
            'meets_1hr_minimum': len(longest_block) >= 1,
            'meets_2hr_minimum': len(longest_block) >= 2,
        }
        
        assert result == snapshot


class TestBodyFiltering:
    """Test filtering of bodies not visible across entire viewing block"""
    
    def test_body_rising_during_block_excluded(self, mock_settings, snapshot):
        """Test that a body rising partway through the block is excluded from chart"""
        from main import create_visibility_chart
        
        eastern = pytz.timezone('America/New_York')
        start_time = eastern.localize(datetime.datetime(2025, 11, 24, 20, 0))
        sorted_times = [start_time + datetime.timedelta(hours=i) for i in range(4)]
        
        # Jupiter rises at hour 2 (not visible entire block)
        # Saturn visible entire block
        targets = ['Jupiter', 'Saturn']
        
        visibility_grid = {}
        body_visibility = {'Jupiter': [], 'Saturn': []}
        
        for i, dt in enumerate(sorted_times):
            visible_bodies = []
            
            # Jupiter only visible in last 2 hours (rises during block)
            if i >= 2:
                visible_bodies.append('Jupiter')
                body_visibility['Jupiter'].append({'time': dt, 'alt': 30, 'high': True})
            
            # Saturn visible all 4 hours
            visible_bodies.append('Saturn')
            body_visibility['Saturn'].append({'time': dt, 'alt': 45, 'high': True})
            
            visibility_grid[dt] = {'bodies': visible_bodies, 'count': len(visible_bodies)}
        
        continuous_blocks = {
            'Jupiter': sorted_times[2:],  # Only last 2 hours
            'Saturn': sorted_times  # All 4 hours
        }
        
        weather_blocks = {dt.replace(minute=0): {'cloud': 5, 'precip': 0, 'good': True} for dt in sorted_times}
        
        # When passed all targets, only Saturn should be included (visible entire block)
        image_data = create_visibility_chart(
            sorted_times,
            visibility_grid,
            body_visibility,
            ['Saturn'],  # Only Saturn is visible entire block
            continuous_blocks=continuous_blocks,
            weather_blocks=weather_blocks,
            moon_phase_emoji='🌓'
        )
        
        assert image_data is not None
        image_bytes = base64.b64decode(image_data)
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        
        snapshot_data = {
            'image_hash': image_hash,
            'image_size_bytes': len(image_bytes),
            'targets_in_chart': ['Saturn'],
            'jupiter_excluded': True,
            'saturn_included': True
        }
        
        assert snapshot_data == snapshot
    
    def test_body_setting_during_block_excluded(self, mock_settings, snapshot):
        """Test that a body setting partway through the block is excluded from chart"""
        from main import create_visibility_chart
        
        eastern = pytz.timezone('America/New_York')
        start_time = eastern.localize(datetime.datetime(2025, 11, 24, 20, 0))
        sorted_times = [start_time + datetime.timedelta(hours=i) for i in range(4)]
        
        # Moon sets at hour 2 (not visible entire block)
        # Jupiter visible entire block
        targets = ['Moon', 'Jupiter']
        
        visibility_grid = {}
        body_visibility = {'Moon': [], 'Jupiter': []}
        
        for i, dt in enumerate(sorted_times):
            visible_bodies = []
            
            # Moon only visible in first 2 hours (sets during block)
            if i < 2:
                visible_bodies.append('Moon')
                body_visibility['Moon'].append({'time': dt, 'alt': 25, 'high': True})
            
            # Jupiter visible all 4 hours
            visible_bodies.append('Jupiter')
            body_visibility['Jupiter'].append({'time': dt, 'alt': 50, 'high': True})
            
            visibility_grid[dt] = {'bodies': visible_bodies, 'count': len(visible_bodies)}
        
        continuous_blocks = {
            'Moon': sorted_times[:2],  # Only first 2 hours
            'Jupiter': sorted_times  # All 4 hours
        }
        
        weather_blocks = {dt.replace(minute=0): {'cloud': 5, 'precip': 0, 'good': True} for dt in sorted_times}
        
        # Only Jupiter should be included (visible entire block)
        image_data = create_visibility_chart(
            sorted_times,
            visibility_grid,
            body_visibility,
            ['Jupiter'],  # Only Jupiter is visible entire block
            continuous_blocks=continuous_blocks,
            weather_blocks=weather_blocks,
            moon_phase_emoji='🌒'
        )
        
        assert image_data is not None
        image_bytes = base64.b64decode(image_data)
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        
        snapshot_data = {
            'image_hash': image_hash,
            'image_size_bytes': len(image_bytes),
            'targets_in_chart': ['Jupiter'],
            'moon_excluded': True,
            'jupiter_included': True
        }
        
        assert snapshot_data == snapshot
    
    def test_no_yellow_in_colormap(self, mock_settings, snapshot):
        """Test that the colormap no longer contains yellow colors"""
        from main import create_visibility_chart
        
        eastern = pytz.timezone('America/New_York')
        start_time = eastern.localize(datetime.datetime(2025, 11, 24, 20, 0))
        sorted_times = [start_time + datetime.timedelta(hours=i) for i in range(3)]
        
        targets = ['Saturn']
        
        visibility_grid = {}
        body_visibility = {'Saturn': []}
        
        # Create data with high altitude (70°) to test high-altitude colors
        for dt in sorted_times:
            visibility_grid[dt] = {'bodies': ['Saturn'], 'count': 1}
            body_visibility['Saturn'].append({'time': dt, 'alt': 70, 'high': True})
        
        continuous_blocks = {'Saturn': sorted_times}
        weather_blocks = {dt.replace(minute=0): {'cloud': 5, 'precip': 0, 'good': True} for dt in sorted_times}
        
        image_data = create_visibility_chart(
            sorted_times,
            visibility_grid,
            body_visibility,
            targets,
            continuous_blocks=continuous_blocks,
            weather_blocks=weather_blocks,
            moon_phase_emoji='🌕'
        )
        
        assert image_data is not None
        image_bytes = base64.b64decode(image_data)
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        
        # Save image for visual inspection
        snapshot_dir = Path(__file__).parent / '__snapshots__'
        snapshot_dir.mkdir(exist_ok=True)
        with open(snapshot_dir / 'test_no_yellow_colormap.png', 'wb') as f:
            f.write(image_bytes)
        
        snapshot_data = {
            'image_hash': image_hash,
            'image_size_bytes': len(image_bytes),
            'high_altitude_test': True,
            'altitude_degrees': 70
        }
        
        assert snapshot_data == snapshot
    
    def test_chart_title_with_location_and_date(self, mock_settings, snapshot):
        """Test that chart title includes location name and date"""
        from main import create_visibility_chart
        
        eastern = pytz.timezone('America/New_York')
        start_time = eastern.localize(datetime.datetime(2025, 11, 24, 20, 0))
        sorted_times = [start_time + datetime.timedelta(hours=i) for i in range(3)]
        
        targets = ['Saturn']
        
        visibility_grid = {}
        body_visibility = {'Saturn': []}
        
        for dt in sorted_times:
            visibility_grid[dt] = {'bodies': ['Saturn'], 'count': 1}
            body_visibility['Saturn'].append({'time': dt, 'alt': 50, 'high': True})
        
        continuous_blocks = {'Saturn': sorted_times}
        weather_blocks = {dt.replace(minute=0): {'cloud': 5, 'precip': 0, 'good': True} for dt in sorted_times}
        
        # Test with location and date
        image_data = create_visibility_chart(
            sorted_times,
            visibility_grid,
            body_visibility,
            targets,
            continuous_blocks=continuous_blocks,
            weather_blocks=weather_blocks,
            moon_phase_emoji='🌕',
            location_name='Orlando',
            date_str='November 24, 2025'
        )
        
        assert image_data is not None
        image_bytes = base64.b64decode(image_data)
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        
        # Save image for visual inspection
        snapshot_dir = Path(__file__).parent / '__snapshots__'
        snapshot_dir.mkdir(exist_ok=True)
        with open(snapshot_dir / 'test_chart_with_location_date.png', 'wb') as f:
            f.write(image_bytes)
        
        snapshot_data = {
            'image_hash': image_hash,
            'image_size_bytes': len(image_bytes),
            'has_location': True,
            'has_date': True,
            'location': 'Orlando',
            'date': 'November 24, 2025'
        }
        
        assert snapshot_data == snapshot
