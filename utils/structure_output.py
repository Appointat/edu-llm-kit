import json

def extract_json_from_string(input_str: str) -> dict:
    r"""Extract the the first JSON from a string, and returns it as a Python
    dictionary.

    Args:
        string (str): The string to extract JSON from.

    Returns:
        dict: The first JSON object found in the string as a Python dictionary.
    """
    input_str = input_str.replace('\\', '\\\\')  # escaping backslashes first

    in_quotes = False
    in_code_block = False
    escaped = False
    depth = 0
    start_index = -1
    clean_input = []

    i = 0
    while i < len(input_str):
        char = input_str[i]

        # Check for code block start or end
        if (
            input_str[i : i + 3] == '```' and input_str[i + 3 : i + 7] != 'json'
        ):  # assuming ``` as code block delimiter
            in_code_block = not in_code_block
            i += 3  # Skip the next two characters as well
            continue

        if char == '"' and not escaped and not in_code_block:
            in_quotes = not in_quotes

        if in_quotes or in_code_block:
            if char == '\\' and not escaped:
                escaped = True
            elif escaped:
                escaped = False
            else:
                if char == '\n':
                    clean_input.append('\\n')
                elif char == '"' and in_code_block:
                    # Escape quotes only inside code blocks
                    clean_input.append('\\"')
                else:
                    clean_input.append(char)
        else:
            clean_input.append(char)

        if char == '{' and not in_quotes and not in_code_block:
            depth += 1
            if depth == 1:
                start_index = i  # mark the start of a JSON object
        elif char == '}' and not in_quotes and not in_code_block:
            depth -= 1
            if depth == 0 and start_index != -1:
                cleaned_str = ''.join(clean_input[start_index : i + 1])
                try:
                    return json.loads(cleaned_str)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        "Failed to decode JSON object:\n"
                        + cleaned_str
                        + "\n"
                        + str(e)
                    ) from e

        i += 1

    raise ValueError("No complete JSON object found:\n" + ''.join(clean_input))
