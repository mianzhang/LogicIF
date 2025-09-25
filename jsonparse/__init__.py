"""
JSONParse - A simple Python package for extracting JSON objects from strings and processing JSONL files.
"""

import json
from typing import Union, Optional, List, Dict
from pathlib import Path

__version__ = "0.0.1"


def extract_valid_json(input_str: str, position: str = 'last') -> Optional[Union[Dict, List]]:
    """
    Extract a valid JSON object or array from a string.
    
    Args:
        input_str (str): The input string containing JSON
        position (str): Either 'first' or 'last' to specify which JSON object to extract
        
    Returns:
        Union[Dict, List, None]: The extracted JSON object/array or None if no valid JSON found
        
    Raises:
        ValueError: If position is not 'first' or 'last'
    """
    if position not in ['first', 'last']:
        raise ValueError("Position must be either 'first' or 'last'")
    
    if position == 'first':
        return _extract_first_json(input_str)
    else:
        return _extract_last_json(input_str)


def parse_jsonl_column(
    input_file: Union[str, Path], 
    output_file: Union[str, Path],
    source_column: str,
    target_column: str,
    position: str = 'last'
) -> None:
    """
    Process a JSONL file and extract JSON objects from a specified column.
    
    Args:
        input_file (Union[str, Path]): Path to the input JSONL file
        output_file (Union[str, Path]): Path to the output JSONL file
        source_column (str): Name of the column containing text with JSON
        target_column (str): Name of the new column for extracted JSON
        position (str): Either 'first' or 'last' to specify which JSON object to extract
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If source_column doesn't exist in the data or position is invalid
        Exception: For other processing errors
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if position not in ['first', 'last']:
        raise ValueError("Position must be either 'first' or 'last'")
    
    try:
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        processed_count = 0
        source_column_found = False
        
        # Process the JSONL file line by line
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                try:
                    # Parse the JSON record
                    record = json.loads(line)
                    
                    # Check if source column exists in this record
                    if source_column not in record:
                        if not source_column_found:
                            # Check if any record has the source column
                            temp_found = _check_source_column_exists(input_path, source_column)
                            if not temp_found:
                                raise ValueError(f"Source column '{source_column}' not found in any records. Available columns in first record: {list(record.keys())}")
                        # Skip this record if it doesn't have the source column
                        print(f"Warning: Source column '{source_column}' not found in record at line {line_num}")
                        record[target_column] = None
                    else:
                        source_column_found = True
                        text = record[source_column]
                        
                        # Extract JSON from the text
                        if text is None or text == "":
                            extracted_json = None
                        else:
                            try:
                                extracted_json = extract_valid_json(str(text), position=position)
                            except Exception as e:
                                print(f"Warning: Error processing record at line {line_num}: {e}")
                                extracted_json = None
                        
                        # Add the extracted JSON to the record
                        record[target_column] = extracted_json
                    
                    # Write the updated record to output file
                    json.dump(record, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    processed_count += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON at line {line_num}: {e}")
                    continue
        
        if processed_count == 0:
            raise ValueError("No valid JSON records found in input file")
        
        print(f"Successfully processed {processed_count} records from {input_path} to {output_path}")
        print(f"Added column '{target_column}' with extracted JSON objects (position: {position})")
        
    except Exception as e:
        raise Exception(f"Error processing JSONL file: {e}")


# Helper functions
def _extract_first_json(input_str: str) -> Optional[Union[Dict, List]]:
    """Extract the first valid JSON object or array from a string."""
    stack = []
    start_index = None

    for i, char in enumerate(input_str):
        if char == '{' or char == '[':
            stack.append(char)
            if len(stack) == 1:
                # Mark the start of a potential JSON object or array
                start_index = i
        elif char == '}' or char == ']':
            if stack:
                if (char == '}' and stack[-1] == '{') or (char == ']' and stack[-1] == '['):
                    stack.pop()
                if len(stack) == 0 and start_index is not None:
                    # Attempt to parse when we find a closing brace or bracket
                    try:
                        json_obj = json.loads(input_str[start_index:i + 1])
                        return json_obj  # Return the first successfully parsed JSON
                    except json.JSONDecodeError:
                        # Reset start_index if parsing fails
                        start_index = None

    # If no valid JSON found
    return None


def _extract_last_json(input_str: str) -> Optional[Union[Dict, List]]:
    """Extract the last valid JSON object or array from a string."""
    valid_jsons = []
    stack = []
    start_index = None

    for i, char in enumerate(input_str):
        if char == '{' or char == '[':
            stack.append(char)
            if len(stack) == 1:
                # Mark the start of a potential JSON object or array
                start_index = i
        elif char == '}' or char == ']':
            if stack:
                if (char == '}' and stack[-1] == '{') or (char == ']' and stack[-1] == '['):
                    stack.pop()
                if len(stack) == 0 and start_index is not None:
                    # Attempt to parse when we find a closing brace or bracket
                    try:
                        json_obj = json.loads(input_str[start_index:i + 1])
                        valid_jsons.append(json_obj)  # Collect all valid JSONs
                    except json.JSONDecodeError:
                        pass  # Continue searching
                    finally:
                        start_index = None

    # Return the last valid JSON found, or None if no valid JSON found
    return valid_jsons[-1] if valid_jsons else None


def _check_source_column_exists(input_path: Path, source_column: str) -> bool:
    """
    Check if the source column exists in any record in the JSONL file.
    
    Args:
        input_path (Path): Path to the input JSONL file
        source_column (str): Name of the source column to check
        
    Returns:
        bool: True if the column exists in at least one record
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if source_column in record:
                        return True
                except json.JSONDecodeError:
                    continue
        return False
    except Exception:
        return False 