import pytest
import datetime
import pytz
from unittest.mock import patch, MagicMock
import base64
from io import BytesIO
from pathlib import Path

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
        
        # Create test data matching good weather scenario
        sorted_times = [
            eastern.localize(datetime.datetime(2025, 11, 24, 20, 0)),
            eastern.localize(datetime.datetime(2025, 11, 24, 21, 0)),
            eastern.localize(datetime.datetime(2025, 11, 24, 22, 0)),
            eastern.localize(datetime.datetime(2025, 11, 24, 23, 0)),
            eastern.localize(datetime.datetime(2025, 11, 25, 0, 0)),
            eastern.localize(datetime.datetime(2025, 11, 25, 1, 0)),
        ]
        
        targets = ['Moon', 'Mars', 'Jupiter', 'Saturn']
        
        # Build visibility grid
        visibility_grid = {}
        body_visibility = {name: [] for name in targets}
        
        for dt in sorted_times:
            hour = dt.hour
            visible_bodies = []
            
            for body_name in targets:
                alt = mock_body_positions(body_name, hour)
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
        
        # Generate chart
        image_data = create_visibility_chart(sorted_times, visibility_grid, body_visibility, targets)
        
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
        
        # Snapshot the visibility data structure
        snapshot_data = {
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
        
        # Only good weather times (cloud < 15%, precip < 5%)
        # From mock_weather_data_mixed: hours 20, 22, 0, 1, 3
        sorted_times = [
            eastern.localize(datetime.datetime(2025, 11, 24, 20, 0)),
            eastern.localize(datetime.datetime(2025, 11, 24, 22, 0)),
            eastern.localize(datetime.datetime(2025, 11, 25, 0, 0)),
            eastern.localize(datetime.datetime(2025, 11, 25, 1, 0)),
            eastern.localize(datetime.datetime(2025, 11, 25, 3, 0)),
        ]
        
        targets = ['Moon', 'Mars', 'Jupiter', 'Saturn']
        
        # Build visibility grid
        visibility_grid = {}
        body_visibility = {name: [] for name in targets}
        
        for dt in sorted_times:
            hour = dt.hour
            visible_bodies = []
            
            for body_name in targets:
                alt = mock_body_positions(body_name, hour)
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
        
        # Generate chart
        image_data = create_visibility_chart(sorted_times, visibility_grid, body_visibility, targets)
        
        # Verify image was created
        assert image_data is not None
        
        # Save the actual PNG image for inspection
        snapshot_dir = Path(__file__).parent / '__snapshots__'
        snapshot_dir.mkdir(exist_ok=True)
        image_path = snapshot_dir / 'test_mixed_conditions_chart.png'
        image_bytes = base64.b64decode(image_data)
        with open(image_path, 'wb') as f:
            f.write(image_bytes)
        
        # Snapshot the visibility data
        snapshot_data = {
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
