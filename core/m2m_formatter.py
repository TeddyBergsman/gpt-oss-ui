"""M2M format parser and markdown formatter."""

from typing import Dict, List, Union, Any
import re


def parse_m2m_output(text: str) -> Dict[str, Any]:
    """
    Parse M2M format output into a structured dictionary.
    
    Format: key:value|key:value|key:list,of,values
    """
    if not text or '|' not in text and ':' not in text:
        return {}
    
    # Clean the text - remove any trailing newlines or spaces
    text = text.strip()
    
    # Handle multi-line M2M outputs (concatenate them)
    lines = text.split('\n')
    combined_text = ''.join(lines)
    
    # Parse key:value pairs
    data = {}
    pairs = combined_text.split('|')
    
    for pair in pairs:
        if ':' not in pair:
            continue
            
        key, value = pair.split(':', 1)
        key = key.strip()
        value = value.strip()
        
        # First handle nested keys (parent.child:value)
        if '.' in key:
            parts = key.split('.')
            current = data
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Now handle the value for the nested key
            if ',' in value:
                # Parse as list
                items = [item.strip() for item in value.split(',')]
                current[parts[-1]] = items
            else:
                current[parts[-1]] = value
        else:
            # Handle non-nested keys
            if ',' in value:
                # Parse as list
                items = [item.strip() for item in value.split(',')]
                data[key] = items
            else:
                data[key] = value
    
    return data


def format_value_as_title(value: str) -> str:
    """Convert underscore_separated_value to Title Case."""
    return ' '.join(word.capitalize() for word in value.split('_'))


def format_m2m_to_markdown(data: Dict[str, Any]) -> str:
    """
    Convert parsed M2M data to formatted markdown with minimal added text.
    """
    if not data:
        return ""
    
    md_lines = []
    
    # Extract main title (look for 'name' or 'title' key)
    title = None
    for key in ['name', 'title', 'subject', 'topic']:
        if key in data:
            title = format_value_as_title(str(data[key]))
            break
    
    if title:
        md_lines.append(f"# {title}\n")
    
    # Group data by type
    metadata = {}
    lists = {}
    nested_data = {}
    
    for key, value in data.items():
        if key in ['name', 'title', 'subject', 'topic']:
            continue  # Already used for title
        elif isinstance(value, list):
            lists[key] = value
        elif isinstance(value, dict):
            nested_data[key] = value
        else:
            metadata[key] = value
    
    # Format metadata section (without extra header)
    if metadata:
        for key, value in metadata.items():
            formatted_key = format_value_as_title(key)
            formatted_value = format_value_as_title(str(value))
            md_lines.append(f"**{formatted_key}:** {formatted_value}  ")
        md_lines.append("")  # Empty line
    
    # Format list sections (minimal - just show key and items)
    for key, items in lists.items():
        formatted_key = format_value_as_title(key)
        md_lines.append(f"**{formatted_key}:**")
        md_lines.append("")  # Blank line needed for markdown to recognize the list
        for item in items:
            formatted_item = format_value_as_title(item)
            md_lines.append(f"- {formatted_item}")
        md_lines.append("")  # Empty line
    
    # Format nested data sections (minimal)
    for key, nested in nested_data.items():
        # Instead of nested structure, format as "Parent; Child"
        for subkey, subvalue in nested.items():
            formatted_full_key = f"{format_value_as_title(key)}; {format_value_as_title(subkey)}"
            if isinstance(subvalue, list):
                md_lines.append(f"**{formatted_full_key}:**")
                md_lines.append("")  # Blank line for list
                for item in subvalue:
                    md_lines.append(f"- {format_value_as_title(item)}")
                md_lines.append("")
            else:
                formatted_value = format_value_as_title(str(subvalue))
                md_lines.append(f"**{formatted_full_key}:** {formatted_value}  ")
        md_lines.append("")
    
    return '\n'.join(md_lines).strip()


def is_m2m_format(text: str) -> bool:
    """Check if text appears to be in M2M format."""
    # Basic heuristic: contains key:value pairs separated by |
    # and uses underscores instead of spaces
    if not text:
        return False
    
    # Check for M2M patterns
    has_colon = ':' in text
    has_pipe = '|' in text
    has_underscore = '_' in text
    no_spaces_in_values = not bool(re.search(r':\s*[^|]*\s+[^|]*', text))
    
    return has_colon and (has_pipe or has_underscore) and no_spaces_in_values


def debug_print_parsed_data(data: Dict[str, Any]) -> None:
    """Print parsed M2M data structure for debugging."""
    print("\n=== M2M Parsed Data Structure ===")
    for key, value in data.items():
        if isinstance(value, list):
            print(f"{key}: [list with {len(value)} items]")
            for i, item in enumerate(value[:3]):  # Show first 3 items
                print(f"  - {item}")
            if len(value) > 3:
                print(f"  ... and {len(value) - 3} more items")
        elif isinstance(value, dict):
            print(f"{key}: [nested dict]")
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue}")
        else:
            print(f"{key}: {value}")
    print("=== End Parsed Data ===\n")
