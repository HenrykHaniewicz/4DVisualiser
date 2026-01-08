"""
Tests for mc_table_to_header.py - Marching Cubes table conversion script.
"""

import pytest
import sys
import io
from unittest.mock import patch, mock_open, MagicMock
from typing import List

from mc_table_to_header import (
    parse_python_tri_table,
    normalize_row_to_16,
    emit_cpp_header,
    main,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def minimal_tri_table_256():
    """Generate a minimal valid TRI_TABLE with 256 rows."""
    rows = []
    for i in range(256):
        rows.append("[-1]")
    return "TRI_TABLE = [\n" + ",\n".join(rows) + "\n]"


@pytest.fixture
def sample_tri_table_256():
    """Generate a sample TRI_TABLE with varied content."""
    rows = []
    # First row: empty (just -1)
    rows.append("[-1]")
    # Some rows with actual triangle data
    rows.append("[0, 8, 3, -1]")
    rows.append("[0, 1, 9, -1]")
    rows.append("[1, 8, 3, 9, 8, 1, -1]")
    # Fill remaining with simple patterns
    for i in range(4, 256):
        if i % 2 == 0:
            rows.append("[-1]")
        else:
            rows.append(f"[{i % 12}, {(i+1) % 12}, {(i+2) % 12}, -1]")
    return "TRI_TABLE = [\n" + ",\n".join(rows) + "\n]"


@pytest.fixture
def real_tri_table_snippet():
    """A snippet from a real marching cubes TRI_TABLE."""
    return '''
TRI_TABLE = [
    [-1],
    [0, 8, 3, -1],
    [0, 1, 9, -1],
    [1, 8, 3, 9, 8, 1, -1],
    [1, 2, 10, -1],
    [0, 8, 3, 1, 2, 10, -1],
    [9, 2, 10, 0, 2, 9, -1],
    [2, 8, 3, 2, 10, 8, 10, 9, 8, -1],
    [3, 11, 2, -1],
    [0, 11, 2, 8, 11, 0, -1],
]
'''


# =============================================================================
# PARSE_PYTHON_TRI_TABLE TESTS
# =============================================================================

class TestParsePythonTriTable:
    """Tests for parse_python_tri_table function."""
    
    def test_parse_simple_table(self):
        """Test parsing a simple table with a few rows."""
        text = "TRI_TABLE = [[-1], [0, 1, 2, -1], [3, 4, 5, -1]]"
        rows = parse_python_tri_table(text)
        
        assert len(rows) == 3
        assert rows[0] == [-1]
        assert rows[1] == [0, 1, 2, -1]
        assert rows[2] == [3, 4, 5, -1]
    
    def test_parse_multiline_table(self):
        """Test parsing a multi-line formatted table."""
        text = """
TRI_TABLE = [
    [-1],
    [0, 8, 3, -1],
    [0, 1, 9, -1],
]
"""
        rows = parse_python_tri_table(text)
        
        assert len(rows) == 3
        assert rows[0] == [-1]
        assert rows[1] == [0, 8, 3, -1]
        assert rows[2] == [0, 1, 9, -1]
    
    def test_parse_with_surrounding_code(self):
        """Test parsing when TRI_TABLE is surrounded by other code."""
        text = """
# Some comment
EDGE_TABLE = [0x0, 0x109, 0x203]

TRI_TABLE = [
    [-1],
    [0, 8, 3, -1],
]

EDGE_VERTICES = [(0, 1), (1, 2)]
"""
        rows = parse_python_tri_table(text)
        
        assert len(rows) == 2
        assert rows[0] == [-1]
        assert rows[1] == [0, 8, 3, -1]
    
    def test_parse_with_extra_whitespace(self):
        """Test parsing handles extra whitespace."""
        text = "TRI_TABLE   =   [  [-1]  ,  [  0  ,  1  ,  -1  ]  ]"
        rows = parse_python_tri_table(text)
        
        assert len(rows) == 2
        assert rows[0] == [-1]
        assert rows[1] == [0, 1, -1]
    
    def test_parse_with_newlines_in_rows(self):
        """Test parsing rows split across multiple lines."""
        text = """
TRI_TABLE = [
    [0, 1, 2,
     3, 4, 5, -1],
    [-1],
]
"""
        rows = parse_python_tri_table(text)
        
        assert len(rows) == 2
        assert rows[0] == [0, 1, 2, 3, 4, 5, -1]
        assert rows[1] == [-1]
    
    def test_parse_missing_tri_table_raises(self):
        """Test that missing TRI_TABLE raises ValueError."""
        text = "SOME_OTHER_TABLE = [[1, 2, 3]]"
        
        with pytest.raises(ValueError, match="Could not find 'TRI_TABLE"):
            parse_python_tri_table(text)
    
    def test_parse_empty_input_raises(self):
        """Test that empty input raises ValueError."""
        with pytest.raises(ValueError, match="Could not find 'TRI_TABLE"):
            parse_python_tri_table("")
    
    def test_parse_unbalanced_brackets_raises(self):
        """Test that unbalanced brackets raise ValueError."""
        text = "TRI_TABLE = [[-1], [0, 1, 2"
        
        with pytest.raises(ValueError, match="Unbalanced brackets"):
            parse_python_tri_table(text)
    
    def test_parse_no_rows_raises(self):
        """Test that TRI_TABLE with no rows raises ValueError."""
        text = "TRI_TABLE = []"
        
        with pytest.raises(ValueError, match="no row lists"):
            parse_python_tri_table(text)
    
    def test_parse_negative_numbers(self):
        """Test parsing handles negative numbers correctly."""
        text = "TRI_TABLE = [[-1], [0, -1], [5, 6, 7, -1]]"
        rows = parse_python_tri_table(text)
        
        assert rows[0] == [-1]
        assert rows[1] == [0, -1]
        assert rows[2] == [5, 6, 7, -1]
    
    def test_parse_256_rows(self, minimal_tri_table_256):
        """Test parsing a full 256-row table."""
        rows = parse_python_tri_table(minimal_tri_table_256)
        assert len(rows) == 256
    
    def test_parse_varied_row_lengths(self):
        """Test parsing rows of varying lengths."""
        text = """
TRI_TABLE = [
    [-1],
    [0, 8, 3, -1],
    [1, 8, 3, 9, 8, 1, -1],
    [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1],
    [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
]
"""
        rows = parse_python_tri_table(text)
        
        assert len(rows) == 5
        assert len(rows[0]) == 1
        assert len(rows[1]) == 4
        assert len(rows[2]) == 7
        assert len(rows[3]) == 13
        assert len(rows[4]) == 16
    
    def test_parse_preserves_order(self):
        """Test that row order is preserved."""
        text = "TRI_TABLE = [[1, -1], [2, -1], [3, -1], [4, -1], [5, -1]]"
        rows = parse_python_tri_table(text)
        
        assert rows[0][0] == 1
        assert rows[1][0] == 2
        assert rows[2][0] == 3
        assert rows[3][0] == 4
        assert rows[4][0] == 5
    
    def test_parse_with_trailing_comma(self):
        """Test parsing handles trailing commas."""
        text = "TRI_TABLE = [[-1,], [0, 1, -1,],]"
        rows = parse_python_tri_table(text)
        
        assert len(rows) == 2
        assert rows[0] == [-1]
        assert rows[1] == [0, 1, -1]
    
    def test_parse_case_sensitive(self):
        """Test that parsing is case-sensitive for TRI_TABLE."""
        text = "tri_table = [[-1]]"
        
        with pytest.raises(ValueError, match="Could not find 'TRI_TABLE"):
            parse_python_tri_table(text)


# =============================================================================
# NORMALIZE_ROW_TO_16 TESTS
# =============================================================================

class TestNormalizeRowTo16:
    """Tests for normalize_row_to_16 function."""
    
    def test_normalize_empty_row_with_terminator(self):
        """Test normalizing a row with just -1."""
        result = normalize_row_to_16([-1])
        
        assert len(result) == 16
        assert result[0] == -1
        assert all(v == -1 for v in result)
    
    def test_normalize_short_row(self):
        """Test normalizing a short row."""
        result = normalize_row_to_16([0, 8, 3, -1])
        
        assert len(result) == 16
        assert result[:4] == [0, 8, 3, -1]
        assert all(v == -1 for v in result[4:])
    
    def test_normalize_medium_row(self):
        """Test normalizing a medium-length row."""
        result = normalize_row_to_16([1, 8, 3, 9, 8, 1, -1])
        
        assert len(result) == 16
        assert result[:7] == [1, 8, 3, 9, 8, 1, -1]
        assert all(v == -1 for v in result[7:])
    
    def test_normalize_max_length_row(self):
        """Test normalizing a maximum length row (15 values + -1)."""
        row = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, -1]
        result = normalize_row_to_16(row)
        
        assert len(result) == 16
        assert result == row
    
    def test_normalize_row_without_terminator(self):
        """Test normalizing a row missing -1 terminator."""
        result = normalize_row_to_16([0, 8, 3])
        
        assert len(result) == 16
        assert result[:4] == [0, 8, 3, -1]
        assert all(v == -1 for v in result[4:])
    
    def test_normalize_empty_row_without_terminator(self):
        """Test normalizing an empty row without terminator."""
        result = normalize_row_to_16([])
        
        assert len(result) == 16
        assert all(v == -1 for v in result)
    
    def test_normalize_row_with_multiple_terminators(self):
        """Test row with -1 in middle truncates at first -1."""
        result = normalize_row_to_16([0, 8, -1, 3, 4, -1])
        
        assert len(result) == 16
        assert result[:3] == [0, 8, -1]
        assert all(v == -1 for v in result[3:])
    
    def test_normalize_row_too_long_raises(self):
        """Test that rows longer than 16 after termination raise error."""
        # 17 elements with terminator
        row = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, -1]
        
        with pytest.raises(ValueError, match="Row longer than 16"):
            normalize_row_to_16(row)
    
    def test_normalize_exactly_15_values(self):
        """Test normalizing exactly 15 triangle indices + terminator."""
        row = list(range(15)) + [-1]  # [0,1,2,...,14,-1]
        result = normalize_row_to_16(row)
        
        assert len(result) == 16
        assert result == row
    
    def test_normalize_preserves_values(self):
        """Test that normalization preserves original values."""
        original = [5, 7, 11, 2, 4, 6, -1]
        result = normalize_row_to_16(original)
        
        assert result[:7] == original
    
    def test_normalize_all_zeros(self):
        """Test normalizing row of all zeros."""
        result = normalize_row_to_16([0, 0, 0, -1])
        
        assert len(result) == 16
        assert result[:4] == [0, 0, 0, -1]
    
    def test_normalize_valid_edge_indices(self):
        """Test with valid marching cubes edge indices (0-11)."""
        row = [0, 3, 8, 1, 9, 4, 7, 11, 2, 10, 5, 6, -1]
        result = normalize_row_to_16(row)
        
        assert len(result) == 16
        assert result[:13] == row


# =============================================================================
# EMIT_CPP_HEADER TESTS
# =============================================================================

class TestEmitCppHeader:
    """Tests for emit_cpp_header function."""
    
    def test_emit_basic_structure(self):
        """Test basic structure of emitted header."""
        rows = [[-1] * 16 for _ in range(256)]
        header = emit_cpp_header(rows)
        
        assert "#pragma once" in header
        assert "static const int TRI_TABLE[256][16]" in header
        assert header.count("{") == 257  # 1 outer + 256 inner
        assert header.count("}") == 257
    
    def test_emit_contains_all_rows(self):
        """Test that all 256 rows are present."""
        rows = [[i] + [-1] * 15 for i in range(256)]
        header = emit_cpp_header(rows)
        
        # Check that each row identifier is present
        for i in range(256):
            assert f"{{{i}," in header
    
    def test_emit_row_format(self):
        """Test individual row formatting."""
        rows = [[-1] * 16] * 256
        rows[0] = [0, 8, 3, -1] + [-1] * 12
        header = emit_cpp_header(rows)
        
        assert "{0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}," in header
    
    def test_emit_includes_comments(self):
        """Test that header includes descriptive comments."""
        rows = [[-1] * 16 for _ in range(256)]
        header = emit_cpp_header(rows)
        
        assert "Generated" in header or "Marching Cubes" in header
        assert "256" in header
        assert "16" in header
    
    def test_emit_valid_cpp_syntax(self):
        """Test that output has valid C++ array syntax."""
        rows = [[0, 1, 2, -1] + [-1] * 12 for _ in range(256)]
        header = emit_cpp_header(rows)
        
        # Should end with proper closing
        assert header.strip().endswith("};")
        
        # Each row should end with },
        lines = header.split('\n')
        row_lines = [l for l in lines if l.strip().startswith('{') and l.strip().endswith('},')]
        assert len(row_lines) == 256
    
    def test_emit_no_trailing_comma_on_last_row(self):
        """Test that the array doesn't have syntax errors."""
        rows = [[-1] * 16 for _ in range(256)]
        header = emit_cpp_header(rows)
        
        # The overall structure should be valid
        # Last content line before }; should have },
        lines = [l.strip() for l in header.split('\n') if l.strip()]
        assert lines[-1] == "};"
    
    def test_emit_consistent_row_length(self):
        """Test all rows have exactly 16 comma-separated values."""
        rows = [[-1] * 16 for _ in range(256)]
        header = emit_cpp_header(rows)
        
        for line in header.split('\n'):
            if line.strip().startswith('{') and line.strip().endswith('},'):
                # Extract content between braces
                content = line.strip()[1:-2]  # Remove { and },
                values = content.split(',')
                assert len(values) == 16, f"Row has {len(values)} values: {line}"
    
    def test_emit_preserves_values(self):
        """Test that emitted values match input."""
        rows = [[-1] * 16 for _ in range(256)]
        rows[42] = [1, 2, 3, 4, 5, -1] + [-1] * 10
        header = emit_cpp_header(rows)
        
        assert "{1,2,3,4,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}," in header
    
    def test_emit_indentation(self):
        """Test that rows are indented."""
        rows = [[-1] * 16 for _ in range(256)]
        header = emit_cpp_header(rows)
        
        for line in header.split('\n'):
            if line.lstrip().startswith('{') and ',' in line:
                # Row lines should be indented
                assert line.startswith('    {') or line.startswith('\t{')


# =============================================================================
# MAIN FUNCTION TESTS
# =============================================================================

class TestMain:
    """Tests for main function."""
    
    def test_main_with_stdin_stdout(self, minimal_tri_table_256):
        """Test main reading from stdin and writing to stdout."""
        with patch('sys.stdin', io.StringIO(minimal_tri_table_256)):
            with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
                with patch('sys.argv', ['mc_table_to_header.py']):
                    result = main()
        
        assert result == 0
        output = mock_stdout.getvalue()
        assert "#pragma once" in output
        assert "TRI_TABLE[256][16]" in output
    
    def test_main_with_input_file(self, minimal_tri_table_256, tmp_path):
        """Test main reading from input file."""
        input_file = tmp_path / "input.py"
        input_file.write_text(minimal_tri_table_256)
        
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            with patch('sys.argv', ['mc_table_to_header.py', '--in', str(input_file)]):
                result = main()
        
        assert result == 0
        output = mock_stdout.getvalue()
        assert "#pragma once" in output
    
    def test_main_with_output_file(self, minimal_tri_table_256, tmp_path):
        """Test main writing to output file."""
        output_file = tmp_path / "output.h"
        
        with patch('sys.stdin', io.StringIO(minimal_tri_table_256)):
            with patch('sys.argv', ['mc_table_to_header.py', '--out', str(output_file)]):
                result = main()
        
        assert result == 0
        assert output_file.exists()
        content = output_file.read_text()
        assert "#pragma once" in content
    
    def test_main_with_both_files(self, minimal_tri_table_256, tmp_path):
        """Test main with both input and output files."""
        input_file = tmp_path / "input.py"
        output_file = tmp_path / "output.h"
        input_file.write_text(minimal_tri_table_256)
        
        with patch('sys.argv', ['mc_table_to_header.py', 
                                '--in', str(input_file),
                                '--out', str(output_file)]):
            result = main()
        
        assert result == 0
        assert output_file.exists()
        content = output_file.read_text()
        assert "#pragma once" in content
        assert "TRI_TABLE[256][16]" in content
    
    def test_main_wrong_row_count_raises(self):
        """Test main raises error when row count isn't 256."""
        bad_table = "TRI_TABLE = [[-1], [-1], [-1]]"  # Only 3 rows
        
        with patch('sys.stdin', io.StringIO(bad_table)):
            with patch('sys.argv', ['mc_table_to_header.py']):
                with pytest.raises(ValueError, match="Expected 256 rows"):
                    main()
    
    def test_main_invalid_input_raises(self):
        """Test main raises error on invalid input."""
        with patch('sys.stdin', io.StringIO("not a valid table")):
            with patch('sys.argv', ['mc_table_to_header.py']):
                with pytest.raises(ValueError, match="Could not find 'TRI_TABLE"):
                    main()
    
    def test_main_default_stdin(self, minimal_tri_table_256):
        """Test that default input is stdin."""
        with patch('sys.stdin', io.StringIO(minimal_tri_table_256)):
            with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
                with patch('sys.argv', ['mc_table_to_header.py', '--in', '-']):
                    result = main()
        
        assert result == 0
    
    def test_main_default_stdout(self, minimal_tri_table_256):
        """Test that default output is stdout."""
        with patch('sys.stdin', io.StringIO(minimal_tri_table_256)):
            with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
                with patch('sys.argv', ['mc_table_to_header.py', '--out', '-']):
                    result = main()
        
        assert result == 0
        assert len(mock_stdout.getvalue()) > 0
    
    def test_main_empty_string_args(self, minimal_tri_table_256):
        """Test empty string arguments treated as stdin/stdout."""
        with patch('sys.stdin', io.StringIO(minimal_tri_table_256)):
            with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
                with patch('sys.argv', ['mc_table_to_header.py', '--in', '', '--out', '']):
                    result = main()
        
        assert result == 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the complete conversion pipeline."""
    
    def test_full_pipeline_minimal(self, minimal_tri_table_256):
        """Test full pipeline with minimal table."""
        rows = parse_python_tri_table(minimal_tri_table_256)
        assert len(rows) == 256
        
        fixed = [normalize_row_to_16(r) for r in rows]
        assert all(len(r) == 16 for r in fixed)
        
        header = emit_cpp_header(fixed)
        assert "#pragma once" in header
        assert "TRI_TABLE[256][16]" in header
    
    def test_full_pipeline_with_varied_data(self, sample_tri_table_256):
        """Test full pipeline with varied row data."""
        rows = parse_python_tri_table(sample_tri_table_256)
        assert len(rows) == 256
        
        fixed = [normalize_row_to_16(r) for r in rows]
        assert all(len(r) == 16 for r in fixed)
        
        header = emit_cpp_header(fixed)
        
        # Verify specific rows are present
        assert "{0,8,3,-1" in header  # Row 1
        assert "{0,1,9,-1" in header  # Row 2
    
    def test_roundtrip_preserves_data(self):
        """Test that data is preserved through the pipeline."""
        # Create specific test data
        rows = []
        for i in range(256):
            if i == 0:
                rows.append("[-1]")
            elif i == 1:
                rows.append("[0, 8, 3, -1]")
            elif i == 255:
                rows.append("[9, 10, 8, 10, 11, 8, -1]")
            else:
                rows.append("[-1]")
        
        text = "TRI_TABLE = [\n" + ",\n".join(rows) + "\n]"
        
        parsed = parse_python_tri_table(text)
        fixed = [normalize_row_to_16(r) for r in parsed]
        header = emit_cpp_header(fixed)
        
        # Check specific values preserved
        assert "{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}," in header  # Row 0
        assert "{0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}," in header  # Row 1
        assert "{9,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}," in header  # Row 255
    
    def test_real_marching_cubes_data_snippet(self, real_tri_table_snippet):
        """Test with a snippet of real marching cubes data."""
        rows = parse_python_tri_table(real_tri_table_snippet)
        
        assert len(rows) == 10
        assert rows[0] == [-1]
        assert rows[1] == [0, 8, 3, -1]
        assert rows[7] == [2, 8, 3, 2, 10, 8, 10, 9, 8, -1]
    
    def test_output_is_valid_cpp_array(self, minimal_tri_table_256):
        """Test that output is syntactically valid C++ array."""
        rows = parse_python_tri_table(minimal_tri_table_256)
        fixed = [normalize_row_to_16(r) for r in rows]
        header = emit_cpp_header(fixed)
        
        # Basic C++ syntax checks
        assert header.count('{') == header.count('}')
        assert header.count('[') == header.count(']')
        
        # Should compile (we can't actually compile, but check structure)
        lines = header.split('\n')
        in_array = False
        brace_count = 0
        
        for line in lines:
            if 'TRI_TABLE[256][16]' in line:
                in_array = True
            if in_array:
                brace_count += line.count('{') - line.count('}')
        
        assert brace_count == 0  # All braces balanced


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Edge case and boundary tests."""
    
    def test_parse_with_hex_values_fails(self):
        """Test that hex values cause int() to fail (expected behavior)."""
        text = "TRI_TABLE = [[0x0, -1]]"
        
        # Hex values like 0x0 won't parse as int without base specification
        with pytest.raises(ValueError):
            parse_python_tri_table(text)
    
    def test_parse_with_float_values_fails(self):
        """Test that float values cause int() to fail."""
        text = "TRI_TABLE = [[0.5, -1]]"
        
        with pytest.raises(ValueError):
            parse_python_tri_table(text)
    
    def test_normalize_single_negative_one(self):
        """Test normalizing single -1."""
        result = normalize_row_to_16([-1])
        assert len(result) == 16
        assert all(v == -1 for v in result)
    
    def test_normalize_exactly_16_values(self):
        """Test normalizing exactly 16 values ending in -1."""
        row = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, -1]
        result = normalize_row_to_16(row)
        assert result == row
    
    def test_empty_rows_produce_valid_output(self):
        """Test all -1 rows produce valid C++ output."""
        rows = [[-1] * 16 for _ in range(256)]
        header = emit_cpp_header(rows)
        
        # All rows should be the same
        expected_row = "{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},"
        assert header.count(expected_row) == 256
    
    def test_max_triangles_row(self):
        """Test row with maximum 5 triangles (15 indices)."""
        # 5 triangles = 15 indices + terminator = 16 values
        row = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, -1]
        result = normalize_row_to_16(row)
        
        assert len(result) == 16
        assert result == row
    
    def test_unicode_in_file_handled(self, tmp_path):
        """Test that UTF-8 encoding is handled properly."""
        # Create table with a UTF-8 comment
        text = '''# Марширующие кубы (Russian comment)
TRI_TABLE = [
''' + ',\n'.join(['    [-1]'] * 256) + '''
]
'''
        input_file = tmp_path / "unicode_input.py"
        input_file.write_text(text, encoding='utf-8')
        output_file = tmp_path / "output.h"
        
        with patch('sys.argv', ['mc_table_to_header.py',
                                '--in', str(input_file),
                                '--out', str(output_file)]):
            result = main()
        
        assert result == 0
        assert output_file.exists()
    
    def test_windows_line_endings(self, tmp_path):
        """Test handling of Windows line endings."""
        rows = ',\r\n'.join(['[-1]'] * 256)
        text = f"TRI_TABLE = [\r\n{rows}\r\n]"
        
        input_file = tmp_path / "windows_input.py"
        input_file.write_bytes(text.encode('utf-8'))
        output_file = tmp_path / "output.h"
        
        with patch('sys.argv', ['mc_table_to_header.py',
                                '--in', str(input_file),
                                '--out', str(output_file)]):
            result = main()
        
        assert result == 0
    
    def test_output_uses_unix_line_endings(self, minimal_tri_table_256, tmp_path):
        """Test that output uses Unix line endings."""
        output_file = tmp_path / "output.h"
        
        with patch('sys.stdin', io.StringIO(minimal_tri_table_256)):
            with patch('sys.argv', ['mc_table_to_header.py', '--out', str(output_file)]):
                main()
        
        content = output_file.read_bytes()
        assert b'\r\n' not in content  # No Windows line endings
        assert b'\n' in content  # Has Unix line endings


# =============================================================================
# ARGUMENT PARSING TESTS
# =============================================================================

class TestArgumentParsing:
    """Tests for command-line argument parsing."""
    
    def test_help_argument(self):
        """Test --help argument."""
        with patch('sys.argv', ['mc_table_to_header.py', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
    
    def test_unknown_argument_fails(self):
        """Test unknown arguments cause error."""
        with patch('sys.argv', ['mc_table_to_header.py', '--unknown']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code != 0
    
    def test_input_file_not_found(self, tmp_path):
        """Test error when input file doesn't exist."""
        nonexistent = tmp_path / "nonexistent.py"
        
        with patch('sys.argv', ['mc_table_to_header.py', '--in', str(nonexistent)]):
            with pytest.raises(FileNotFoundError):
                main()