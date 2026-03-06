# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | Yes                |

## Reporting a Vulnerability

If you discover a security vulnerability in AgentAssay, please report it responsibly.

### How to Report

1. **Do NOT open a public GitHub issue.** Security vulnerabilities must be reported privately.
2. Email: Send a detailed report to the project maintainer via GitHub's private vulnerability reporting feature, or contact the repository owner directly.
3. Include the following in your report:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment:** Within 48 hours of your report.
- **Assessment:** We will evaluate the severity and impact within 5 business days.
- **Fix timeline:** Critical vulnerabilities will be patched within 7 days. Other issues will be addressed in the next scheduled release.
- **Disclosure:** We will coordinate with you on responsible disclosure timing.

### Scope

The following are in scope for security reports:

- Code safety vulnerabilities in the condition parser or contract evaluation engine
- Dependency vulnerabilities that affect AgentAssay directly
- Information disclosure through error messages or reports
- Unsafe handling of agent traces or configuration files

### Out of Scope

- Vulnerabilities in agent frameworks that AgentAssay integrates with (report those to the respective framework maintainers)
- Issues that require physical access to the machine running AgentAssay
- Denial of service through intentionally large inputs (this is a testing tool, not a production service)

## Security Design Principles

- **No dynamic code evaluation.** The contract condition parser uses safe pattern matching, not arbitrary code evaluation.
- **Input validation.** All configuration and data models enforce strict validation and type constraints.
- **Minimal permissions.** AgentAssay does not require network access, filesystem write access (beyond report output), or elevated privileges.
- **Dependency minimization.** Core functionality uses a small set of well-maintained dependencies.
