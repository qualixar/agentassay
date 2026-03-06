# Changelog

All notable changes to AgentAssay will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-06

### Added

- Stochastic verdict system with three-valued outcomes (PASS, FAIL, INCONCLUSIVE)
- Sequential probability ratio test for adaptive sampling and early stopping
- Five-dimensional agent coverage metrics (tool, path, state, boundary, model)
- Twelve mutation operators across four categories (prompt, tool, model, context)
- Seven metamorphic testing relations across four families
- Contract oracle integration for behavioral specification checking
- pytest plugin with statistical assertions and custom markers
- CLI with five commands: run, compare, mutate, coverage, report
- Six framework adapters for popular agent frameworks
- Console, HTML, and JSON reporting
- Deployment gate system for CI/CD integration
- Confidence interval estimation with multiple methods
- Effect size computation for regression quantification
- Sample size determination for test planning
