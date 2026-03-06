# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Reporting module for AgentAssay.

Provides three complementary output formats:

- **ConsoleReporter**: Rich-based terminal output with coloured tables,
  panels, and progress bars. Ideal for interactive use and CI log output.
- **HTMLReporter**: Self-contained HTML reports with embedded CSS (dark
  theme). No external dependencies at render time. Ideal for sharing
  and archiving.
- **JSONExporter**: Structured JSON export for programmatic consumption.
  Uses Pydantic ``model_dump(mode="json")`` for type-safe serialisation.

Example
-------
>>> from agentassay.reporting import ConsoleReporter, HTMLReporter, JSONExporter
>>>
>>> # Terminal output
>>> console = ConsoleReporter(verbose=True)
>>> console.print_verdict(verdict)
>>> console.print_coverage(coverage)
>>>
>>> # HTML report
>>> html = HTMLReporter()
>>> html.save_report(data, "report.html")
>>>
>>> # JSON export
>>> JSONExporter.save(data, "report.json")
"""

from agentassay.reporting.console import ConsoleReporter
from agentassay.reporting.html import HTMLReporter
from agentassay.reporting.json_export import JSONExporter

__all__ = [
    "ConsoleReporter",
    "HTMLReporter",
    "JSONExporter",
]
