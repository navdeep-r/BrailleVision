"""
inference/braille_decoder.py

Stateful Braille decoder implementing UEB Grade 1 and Grade 2 contractions.

Bit convention (module-level):
  bit 0 = dot 1 (top-left)    bit 3 = dot 4 (top-right)
  bit 1 = dot 2 (mid-left)    bit 4 = dot 5 (mid-right)
  bit 2 = dot 3 (bot-left)    bit 5 = dot 6 (bot-right)

States: "normal", "capital_single", "capital_word", "number"

WHY a state machine is necessary:
  Capital and number indicators make Braille stateful. A static lookup table
  produces wrong output for every uppercase word and every number sequence.
"""

from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Indicator constants
# ---------------------------------------------------------------------------

CAPITAL_INDICATOR = 0b100000   # dot 6 alone
NUMBER_INDICATOR  = 0b111100   # dots 3, 4, 5, 6
LETTER_INDICATOR  = 0b010000   # dot 5 alone


# ---------------------------------------------------------------------------
# Lookup table builders
# ---------------------------------------------------------------------------

def build_grade1_table() -> Dict[int, str]:
    """Complete UEB Grade 1 character mapping."""
    return {
        0b000001: "a", 0b000011: "b", 0b001001: "c", 0b011001: "d", 0b010001: "e",
        0b001011: "f", 0b011011: "g", 0b010011: "h", 0b001010: "i", 0b011010: "j",
        0b000101: "k", 0b000111: "l", 0b001101: "m", 0b011101: "n", 0b010101: "o",
        0b001111: "p", 0b011111: "q", 0b010111: "r", 0b001110: "s", 0b011110: "t",
        0b100101: "u", 0b100111: "v", 0b111010: "w", 0b101101: "x", 0b111101: "y",
        0b110101: "z",
        0b000010: ",",
        0b000110: ";",
        0b010010: ":",
        0b110010: ".",
        0b010110: "!",
        0b100110: "?",
        0b100100: "-",
        0b000100: "'",
        0b000000: " ",
    }


def build_number_table() -> Dict[int, str]:
    """Braille number patterns (same as a-j in Grade 1)."""
    return {
        0b000001: "1",
        0b000011: "2",
        0b001001: "3",
        0b011001: "4",
        0b010001: "5",
        0b001011: "6",
        0b011011: "7",
        0b010011: "8",
        0b001010: "9",
        0b011010: "0",
    }


def build_grade2_table() -> Dict[int, str]:
    """
    Partial UEB Grade 2 whole-word contractions.
    Only applied when the pattern appears as a standalone word (surrounded by spaces/boundaries).
    """
    return {
        0b000011: "but",
        0b001001: "can",
        0b011001: "do",
        0b010001: "every",
        0b001011: "from",
        0b011011: "go",
        0b010011: "have",
        0b011010: "just",
        0b000101: "knowledge",
        0b000111: "like",
        0b001101: "more",
        0b011101: "not",
        0b001111: "people",
        0b011111: "quite",
        0b010111: "rather",
        0b001110: "so",
        0b011110: "that",
        0b100101: "us",
        0b100111: "very",
        0b111010: "will",
        0b111101: "you",
        0b110101: "as",
        0b001010: "in",
    }


# ---------------------------------------------------------------------------
# Decoder class
# ---------------------------------------------------------------------------

class BrailleDecoder:
    """
    Stateful Braille decoder.

    States:
      "normal"         — default, lower-case letters
      "capital_single" — next letter is uppercase, then revert to normal
      "capital_word"   — all letters uppercase until space
      "number"         — number mode until letter indicator or space
    """

    def __init__(self, grade: int = 2):
        self.grade = grade
        self.g1_table  = build_grade1_table()
        self.num_table = build_number_table()
        self.g2_table  = build_grade2_table() if grade == 2 else {}
        self.mode = "normal"

    def reset(self) -> None:
        """Reset state for a fresh page."""
        self.mode = "normal"

    def _apply_capitalization(self, char: str) -> str:
        """Apply capitalization based on current mode."""
        if self.mode == "capital_single":
            self.mode = "normal"
            return char.upper() if char.isalpha() else char
        elif self.mode == "capital_word":
            return char.upper() if char.isalpha() else char
        return char

    def _decode_single_cell(self, pattern: int) -> Optional[str]:
        """
        Decode one cell pattern given the current state machine mode.
        Returns a character, or None if the pattern is an indicator cell.
        """
        # --- Indicator checks ---
        if pattern == CAPITAL_INDICATOR:
            if self.mode == "normal":
                self.mode = "capital_single"
            elif self.mode == "capital_single":
                self.mode = "capital_word"
            # In number mode: capital indicator has no effect
            return None  # not a printable character

        if pattern == NUMBER_INDICATOR:
            self.mode = "number"
            return None

        if pattern == LETTER_INDICATOR:
            if self.mode == "number":
                self.mode = "normal"
            return None

        # --- Content lookup ---
        if self.mode == "number":
            char = self.num_table.get(pattern)
            if char is not None:
                return char
            # Pattern not in number table → exit number mode, fall through to Grade 1
            self.mode = "normal"

        char = self.g1_table.get(pattern, "?")
        return self._apply_capitalization(char)

    def decode_row(self, pattern_sequence: List[Optional[int]]) -> str:
        """
        Decode one row of pattern_ints (with None = word space sentinels).
        """
        result = []
        n = len(pattern_sequence)

        for i, pattern in enumerate(pattern_sequence):
            if pattern is None:
                result.append(" ")
                # Space resets certain modes
                if self.mode == "capital_word":
                    self.mode = "normal"
                if self.mode == "number":
                    self.mode = "normal"
                continue

            # Grade 2 standalone-word contraction check (before Grade 1)
            if self.grade == 2 and pattern in self.g2_table:
                prev_is_boundary = (i == 0 or pattern_sequence[i - 1] is None)
                next_is_boundary = (i == n - 1 or pattern_sequence[i + 1] is None)
                if prev_is_boundary and next_is_boundary:
                    word = self.g2_table[pattern]
                    word = self._apply_capitalization(word)
                    result.append(word)
                    continue

            # Grade 1 / indicator decode
            char = self._decode_single_cell(pattern)
            if char is not None:
                result.append(char)

        return "".join(result)

    def decode_page(self, rows: List[List[Optional[int]]]) -> str:
        """
        Decode a full page of rows.
        Resets state at the start of every page.
        """
        self.reset()
        decoded_rows = []
        for row in rows:
            line = self.decode_row(row)
            line = line.rstrip()
            if line:  # skip blank rows
                decoded_rows.append(line)
        return "\n".join(decoded_rows)
