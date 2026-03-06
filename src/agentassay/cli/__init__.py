# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""AgentAssay CLI — command-line interface exports.

Provides the ``cli`` Click group as the public entry point.

Usage:
    Registered as a console script in pyproject.toml::

        [project.scripts]
        agentassay = "agentassay.cli.main:cli"

    Or invoke directly::

        python -m agentassay.cli.main
"""

from agentassay.cli.main import cli

__all__ = ["cli"]
