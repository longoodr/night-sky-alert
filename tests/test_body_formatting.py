import pytest
from main import format_body_list


class TestBodyListFormatting:
    """Test the format_body_list function for various scenarios"""
    
    @pytest.mark.parametrize("bodies,total,expected,description", [
        # Single body scenarios
        (['Moon'], 1, 'Moon', "single body - all visible"),
        (['Moon'], 3, 'including Moon', "single body - partial"),
        
        # Two body scenarios
        (['Jupiter', 'Saturn'], 2, 'Jupiter and Saturn', "two bodies - all visible"),
        (['Jupiter', 'Saturn'], 4, 'including Jupiter and Saturn', "two bodies - partial"),
        
        # Three body scenarios
        (['Jupiter', 'Mars', 'Saturn'], 3, 'Jupiter, Mars, and Saturn', "three bodies - all visible"),
        (['Jupiter', 'Mars', 'Saturn'], 4, 'including Jupiter, Mars, and Saturn', "three bodies - partial"),
        
        # Four body scenarios
        (['Jupiter', 'Mars', 'Moon', 'Saturn'], 4, 'Jupiter, Mars, Moon, and Saturn', "four bodies - all visible"),
        
        # Edge cases
        ([], 4, '', "empty list"),
        (['Mars', 'Moon', 'Saturn'], 3, 'Mars, Moon, and Saturn', "alphabetical ordering maintained"),
    ])
    def test_body_list_formatting(self, bodies, total, expected, description):
        """Test various body list formatting scenarios"""
        result = format_body_list(bodies, total_bodies_tracked=total)
        assert result == expected, f"Failed: {description}"
    
    def test_oxford_comma_present(self):
        """Verify Oxford comma is used in lists of 3+"""
        result = format_body_list(['A', 'B', 'C'], total_bodies_tracked=3)
        assert ', and' in result, "Oxford comma should be present in 3+ item lists"
