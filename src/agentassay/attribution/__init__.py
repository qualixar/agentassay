# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Qualixar 3-Layer Attribution System for AgentAssay.

This module provides cryptographic signing, steganographic watermarking,
and visible attribution for all AgentAssay outputs to ensure proper
provenance tracking and IP protection.

Layer 1: Visible Attribution
    - Source file headers
    - CLI version output
    - Report footers
    - JSON metadata

Layer 2: Cryptographic Signing (QualixarSigner)
    - HMAC-SHA256 signatures
    - Author and product metadata
    - Tamper detection

Layer 3: Steganographic Watermark (QualixarWatermark)
    - Zero-width Unicode embedding
    - Invisible attribution
    - Resilient to text modifications

Example
-------
>>> from agentassay.attribution import QualixarSigner, QualixarWatermark
>>> signer = QualixarSigner()
>>> metadata = signer.sign("My report content")
>>> watermark = QualixarWatermark()
>>> marked_text = watermark.embed("My report content")
"""

from __future__ import annotations

from agentassay.attribution.signer import QualixarSigner
from agentassay.attribution.watermark import QualixarWatermark

__all__ = [
    "QualixarSigner",
    "QualixarWatermark",
]
