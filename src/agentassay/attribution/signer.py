# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Layer 2: Cryptographic signing for AgentAssay outputs.

Provides HMAC-SHA256 signatures for content authentication and
tamper detection. The private key is stored in ~/.agentassay/signing.key
and is auto-generated on first use.

All signed outputs include author, product, license, and timestamp
metadata along with the cryptographic signature.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class QualixarSigner:
    """Signs AgentAssay outputs with HMAC-SHA256 for provenance verification.

    The signer uses a private key stored in the user's home directory to
    generate cryptographic signatures. If no key exists, one is auto-generated
    on first use.

    Signed metadata includes:
        - author: Varun Pratap Bhardwaj
        - author_url: https://varunpratap.com
        - product: AgentAssay
        - product_url: https://qualixar.com
        - license: Apache-2.0
        - timestamp: ISO 8601 UTC
        - content_hash: SHA-256 of content
        - signature: HMAC-SHA256(content_hash, private_key)

    Example
    -------
    >>> signer = QualixarSigner()
    >>> metadata = signer.sign("My report content")
    >>> assert signer.verify("My report content", metadata)
    """

    DEFAULT_KEY_PATH = Path.home() / ".agentassay" / "signing.key"

    def __init__(self, private_key: str | None = None) -> None:
        """Initialize the signer with a private key.

        Parameters
        ----------
        private_key
            Hex-encoded private key. If ``None``, loads from
            ``~/.agentassay/signing.key`` or generates a new one.
        """
        if private_key is not None:
            self._key_bytes = bytes.fromhex(private_key)
        else:
            self._key_bytes = self._load_or_generate_key()

    def sign(self, content: str) -> dict[str, Any]:
        """Sign content and return metadata dict with signature.

        Parameters
        ----------
        content
            The content to sign (typically a report or output).

        Returns
        -------
        dict
            Metadata dict with all attribution fields plus signature.
        """
        # Compute content hash
        content_bytes = content.encode("utf-8")
        content_hash = hashlib.sha256(content_bytes).hexdigest()

        # Generate signature: HMAC-SHA256(content_hash, key)
        signature = hmac.new(
            self._key_bytes,
            content_hash.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        # Build metadata
        metadata = {
            "author": "Varun Pratap Bhardwaj",
            "author_url": "https://varunpratap.com",
            "product": "AgentAssay",
            "product_url": "https://qualixar.com",
            "license": "Apache-2.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content_hash": content_hash,
            "signature": signature,
        }

        return metadata

    def verify(self, content: str, signature_metadata: dict[str, Any]) -> bool:
        """Verify a signed output.

        Parameters
        ----------
        content
            The original content.
        signature_metadata
            The metadata dict returned by ``sign()``.

        Returns
        -------
        bool
            ``True`` if signature is valid and content matches,
            ``False`` otherwise.
        """
        # Recompute content hash
        content_bytes = content.encode("utf-8")
        content_hash = hashlib.sha256(content_bytes).hexdigest()

        # Check if content hash matches
        stored_hash = signature_metadata.get("content_hash")
        if stored_hash != content_hash:
            return False

        # Recompute signature
        expected_sig = hmac.new(
            self._key_bytes,
            content_hash.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        # Compare signatures (constant-time comparison)
        stored_sig = signature_metadata.get("signature", "")
        return hmac.compare_digest(expected_sig, stored_sig)

    def export_key(self) -> str:
        """Export the private key as a hex string.

        Returns
        -------
        str
            Hex-encoded private key.

        Warning
        -------
        Keep this key secure. Anyone with this key can forge signatures.
        """
        return self._key_bytes.hex()

    def _load_or_generate_key(self) -> bytes:
        """Load key from disk or generate a new one.

        Returns
        -------
        bytes
            The private key (32 bytes).
        """
        if self.DEFAULT_KEY_PATH.exists():
            # Load existing key
            key_hex = self.DEFAULT_KEY_PATH.read_text(encoding="utf-8").strip()
            return bytes.fromhex(key_hex)
        else:
            # Generate new key
            key_bytes = secrets.token_bytes(32)
            self.DEFAULT_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
            self.DEFAULT_KEY_PATH.write_text(
                key_bytes.hex(), encoding="utf-8"
            )
            # Secure permissions (readable only by owner)
            self.DEFAULT_KEY_PATH.chmod(0o600)
            return key_bytes
