"""
inference/braille_decoder.py

Simple Braille decoder for 26-class (a-z) system.
No states, no capitalization, no numbers.
"""

from typing import List, Optional

class BrailleDecoder:
    """
    Direct mapping from integer classes (0-25) to lowercase letters (a-z).
    """

    def __init__(self, grade: int = 1):
        # Grade argument kept for compatibility with existing code
        self.idx_to_char = {i: chr(ord('a') + i) for i in range(26)}

    def reset(self) -> None:
        pass

    def decode_row(self, pattern_sequence: List[Optional[int]]) -> str:
        """
        Decode one row of pattern_ints (with None = word space sentinels).
        """
        result = []
        for pattern in pattern_sequence:
            if pattern is None:
                result.append(" ")
            else:
                char = self.idx_to_char.get(pattern, "?")
                result.append(char)

        return "".join(result)

    def decode_page(self, rows: List[List[Optional[int]]]) -> str:
        """
        Decode a full page of rows.
        """
        self.reset()
        decoded_rows = []
        for row in rows:
            line = self.decode_row(row)
            line = line.rstrip()
            if line:  # skip blank rows
                decoded_rows.append(line)
        return "\n".join(decoded_rows)
