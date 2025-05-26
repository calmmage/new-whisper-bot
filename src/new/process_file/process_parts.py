# given a bunch of audio parts, parse them and merge the results
from src.new.process_file.process_part import process_part


def process_parts(parts: list):
    parsed_parts = []
    for part in parts:
        parsed_parts.append(process_part(part))

    # return parsed_parts
    # merge parts
    result = ""

    return result
