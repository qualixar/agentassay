# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Layer 3: Steganographic watermarking for AgentAssay outputs.

Embeds invisible attribution in text using zero-width Unicode characters.
The watermark is resilient to whitespace changes and text modifications,
providing a covert provenance signal.

Uses three zero-width characters:
    - U+200B (zero-width space)       -> binary 0
    - U+200C (zero-width non-joiner)  -> binary 1
    - U+200D (zero-width joiner)      -> separator

The watermark encodes a fixed payload: "qualixar:agentassay"
"""

from __future__ import annotations

import re
from typing import Any


class QualixarWatermark:
    """Embeds invisible attribution in text outputs.

    Uses zero-width Unicode characters to encode a watermark payload
    that survives copy-paste and text editing. The watermark is embedded
    after the first paragraph break or at the end of the text.

    Example
    -------
    >>> wm = QualixarWatermark()
    >>> marked = wm.embed("Hello world\\n\\nThis is a test.")
    >>> detected = wm.detect(marked)
    >>> assert detected is not None
    >>> assert detected["product"] == "AgentAssay"
    """

    # Zero-width characters for encoding
    ZWC_0 = "\u200b"  # zero-width space
    ZWC_1 = "\u200c"  # zero-width non-joiner
    ZWC_SEP = "\u200d"  # zero-width joiner (separator)

    # Watermark payload
    MARKER = "qualixar:agentassay"

    def embed(self, text: str) -> str:
        """Embed watermark in text.

        The watermark is inserted after the first paragraph break
        (double newline) or at the end if no paragraph break exists.

        Parameters
        ----------
        text
            The text to watermark.

        Returns
        -------
        str
            Text with embedded watermark.
        """
        # Encode marker to zero-width characters
        zwc_string = self._encode_to_zwc(self.MARKER)

        # Find first paragraph break
        match = re.search(r"\n\n", text)
        if match:
            # Insert after first paragraph
            pos = match.end()
            return text[:pos] + zwc_string + text[pos:]
        else:
            # Append at end
            return text + zwc_string

    def detect(self, text: str) -> dict[str, Any] | None:
        """Detect and extract watermark from text.

        Parameters
        ----------
        text
            Text potentially containing a watermark.

        Returns
        -------
        dict or None
            Metadata dict if watermark is detected, ``None`` otherwise.
            The dict contains:
                - product: "AgentAssay"
                - marker: The decoded marker string
                - qualixar: True
        """
        # Extract all zero-width sequences
        zwc_sequences = self._extract_zwc_sequences(text)

        # Try to decode each sequence
        for zwc_seq in zwc_sequences:
            decoded = self._decode_from_zwc(zwc_seq)
            if decoded == self.MARKER:
                return {
                    "product": "AgentAssay",
                    "marker": decoded,
                    "qualixar": True,
                }

        return None

    def _encode_to_zwc(self, message: str) -> str:
        """Encode string to zero-width characters.

        Converts each character to its binary representation and maps:
            - '0' -> U+200B
            - '1' -> U+200C
            - byte boundary -> U+200D

        Parameters
        ----------
        message
            String to encode.

        Returns
        -------
        str
            Zero-width character string.
        """
        zwc_parts = []

        for char in message:
            # Convert char to 8-bit binary
            binary = format(ord(char), "08b")

            # Map each bit to ZWC
            for bit in binary:
                if bit == "0":
                    zwc_parts.append(self.ZWC_0)
                else:
                    zwc_parts.append(self.ZWC_1)

            # Add separator after each byte
            zwc_parts.append(self.ZWC_SEP)

        return "".join(zwc_parts)

    def _decode_from_zwc(self, zwc_string: str) -> str | None:
        """Decode zero-width characters back to string.

        Parameters
        ----------
        zwc_string
            String of zero-width characters.

        Returns
        -------
        str or None
            Decoded string, or ``None`` if decoding fails.
        """
        if not zwc_string:
            return None

        try:
            # Split by separator to get individual bytes
            byte_parts = zwc_string.split(self.ZWC_SEP)

            chars = []
            for byte_part in byte_parts:
                if not byte_part:
                    continue

                # Convert ZWC back to binary
                binary = ""
                for char in byte_part:
                    if char == self.ZWC_0:
                        binary += "0"
                    elif char == self.ZWC_1:
                        binary += "1"
                    else:
                        # Invalid character
                        return None

                # Convert binary to character
                if len(binary) == 8:
                    code_point = int(binary, 2)
                    chars.append(chr(code_point))

            return "".join(chars)

        except (ValueError, OverflowError):
            return None

    def _extract_zwc_sequences(self, text: str) -> list[str]:
        """Extract all contiguous zero-width character sequences from text.

        Parameters
        ----------
        text
            Text to search.

        Returns
        -------
        list of str
            List of zero-width character sequences found.
        """
        # Pattern matching any of our zero-width characters
        zwc_pattern = f"[{self.ZWC_0}{self.ZWC_1}{self.ZWC_SEP}]+"
        matches = re.findall(zwc_pattern, text)
        return matches
