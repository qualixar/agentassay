#!/usr/bin/env bash
# ============================================================================
# AgentAssay Pre-Launch Verification Script
# Run this BEFORE pushing to GitHub or publishing to PyPI
#
# Usage: bash scripts/pre-launch-verify.sh
#
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# ============================================================================

set -e

PYTHON="/opt/homebrew/Caskroom/miniforge/base/bin/python3"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
PASS=0
FAIL=0
WARN=0

echo "============================================"
echo "  AgentAssay Pre-Launch Verification"
echo "  $(date)"
echo "============================================"
echo ""

# ---- Check 1: Python version ----
echo -n "1. Python version... "
PY_VER=$($PYTHON --version 2>&1)
if [[ "$PY_VER" == *"3.12"* ]] || [[ "$PY_VER" == *"3.11"* ]] || [[ "$PY_VER" == *"3.10"* ]]; then
    echo -e "${GREEN}OK${NC} ($PY_VER)"
    ((PASS++))
else
    echo -e "${YELLOW}WARN${NC} ($PY_VER — expected 3.10-3.12)"
    ((WARN++))
fi

# ---- Check 2: Package installs ----
echo -n "2. Package install (editable)... "
if $PYTHON -m pip install -e ".[dev]" --quiet 2>/dev/null; then
    echo -e "${GREEN}OK${NC}"
    ((PASS++))
else
    echo -e "${RED}FAIL${NC}"
    ((FAIL++))
fi

# ---- Check 3: All 14 modules import ----
echo -n "3. Module imports (14 modules)... "
IMPORT_RESULT=$($PYTHON -c "
modules = ['core', 'statistics', 'verdicts', 'coverage', 'mutation', 'metamorphic',
           'contracts', 'efficiency', 'integrations', 'persistence', 'plugin', 'cli', 'reporting', 'dashboard']
failed = []
for mod in modules:
    try:
        __import__(f'agentassay.{mod}')
    except Exception as e:
        failed.append(f'{mod}: {e}')
if failed:
    print('FAIL:' + ';'.join(failed))
else:
    print('OK:14/14')
" 2>&1)
if [[ "$IMPORT_RESULT" == OK* ]]; then
    echo -e "${GREEN}OK${NC} (14/14 modules)"
    ((PASS++))
else
    echo -e "${RED}FAIL${NC} — $IMPORT_RESULT"
    ((FAIL++))
fi

# ---- Check 4: CLI works ----
echo -n "4. CLI --help... "
if agentassay --help >/dev/null 2>&1; then
    echo -e "${GREEN}OK${NC}"
    ((PASS++))
else
    echo -e "${RED}FAIL${NC}"
    ((FAIL++))
fi

echo -n "5. CLI --version... "
CLI_VER=$(agentassay --version 2>&1)
if [[ "$CLI_VER" == *"0.1.0"* ]]; then
    echo -e "${GREEN}OK${NC} ($CLI_VER)"
    ((PASS++))
else
    echo -e "${RED}FAIL${NC} ($CLI_VER)"
    ((FAIL++))
fi

echo -n "6. CLI commands (6 expected)... "
CMD_COUNT=$(agentassay --help 2>&1 | grep -c "^  [a-z]")
if [ "$CMD_COUNT" -ge 6 ]; then
    echo -e "${GREEN}OK${NC} ($CMD_COUNT commands)"
    ((PASS++))
else
    echo -e "${RED}FAIL${NC} ($CMD_COUNT commands, expected 6+)"
    ((FAIL++))
fi

# ---- Check 5: Full test suite ----
echo -n "7. Test suite... "
TEST_OUTPUT=$($PYTHON -m pytest tests/ -q --tb=line 2>&1)
TEST_RESULT=$?
PASSED=$(echo "$TEST_OUTPUT" | grep -Eo '[0-9]+ passed' | grep -Eo '[0-9]+')
FAILED_TESTS=$(echo "$TEST_OUTPUT" | grep -Eo '[0-9]+ failed' | grep -Eo '[0-9]+' || echo "0")
if [ "$TEST_RESULT" -eq 0 ]; then
    echo -e "${GREEN}OK${NC} ($PASSED passed, 0 failed)"
    ((PASS++))
else
    echo -e "${RED}FAIL${NC} ($PASSED passed, $FAILED_TESTS failed)"
    ((FAIL++))
    echo "   Last 5 lines:"
    echo "$TEST_OUTPUT" | tail -5 | sed 's/^/   /'
fi

# ---- Check 6: Package builds ----
echo -n "8. Package build (wheel + sdist)... "
rm -rf dist/ 2>/dev/null
if $PYTHON -m build --quiet 2>/dev/null; then
    WHL=$(ls dist/*.whl 2>/dev/null | head -1)
    SDIST=$(ls dist/*.tar.gz 2>/dev/null | head -1)
    if [ -n "$WHL" ] && [ -n "$SDIST" ]; then
        echo -e "${GREEN}OK${NC} (wheel + sdist)"
        ((PASS++))
    else
        echo -e "${RED}FAIL${NC} (missing artifacts)"
        ((FAIL++))
    fi
else
    echo -e "${RED}FAIL${NC}"
    ((FAIL++))
fi

# ---- Check 7: No secrets in source ----
echo -n "9. No secrets in source... "
# Look for actual API key patterns (sk-proj-, sk-ant-, real tokens), not variable names like sk-agent
SECRET_HITS=$(grep -rlE "sk-proj-[a-zA-Z0-9]{20}|sk-ant-[a-zA-Z0-9]{20}|AKIA[0-9A-Z]{16}|ghp_[a-zA-Z0-9]{36}" src/ tests/ README.md 2>/dev/null | wc -l | tr -d ' ')
API_KEY_HITS=$(grep -rlE "=['\"]sk-[a-zA-Z0-9]{30,}|=['\"]AKIA[0-9A-Z]{16}" src/ tests/ 2>/dev/null | wc -l | tr -d ' ')
if [ "$SECRET_HITS" -eq 0 ] && [ "$API_KEY_HITS" -eq 0 ]; then
    echo -e "${GREEN}OK${NC} (no hardcoded secrets)"
    ((PASS++))
else
    echo -e "${RED}FAIL${NC} (found $SECRET_HITS secret patterns, $API_KEY_HITS API key refs)"
    ((FAIL++))
fi

# ---- Check 8: No .env file ----
echo -n "10. No .env file exposed... "
if [ -f ".env" ]; then
    echo -e "${RED}FAIL${NC} (.env file exists — will be committed!)"
    ((FAIL++))
else
    echo -e "${GREEN}OK${NC}"
    ((PASS++))
fi

# ---- Check 9: .gitignore exists ----
echo -n "11. .gitignore exists... "
if [ -f ".gitignore" ]; then
    PLAN_IGNORED=$(grep -c "^plan/" .gitignore 2>/dev/null || echo "0")
    BACKUP_IGNORED=$(grep -c "^.backup/" .gitignore 2>/dev/null || echo "0")
    ENV_IGNORED=$(grep -c "^.env" .gitignore 2>/dev/null || echo "0")
    if [ "$PLAN_IGNORED" -gt 0 ] && [ "$BACKUP_IGNORED" -gt 0 ] && [ "$ENV_IGNORED" -gt 0 ]; then
        echo -e "${GREEN}OK${NC} (plan/, .backup/, .env all ignored)"
        ((PASS++))
    else
        echo -e "${YELLOW}WARN${NC} (some sensitive dirs may not be ignored)"
        ((WARN++))
    fi
else
    echo -e "${RED}FAIL${NC}"
    ((FAIL++))
fi

# ---- Check 10: Key files exist ----
echo -n "12. Required files exist... "
MISSING=""
for f in README.md LICENSE pyproject.toml CHANGELOG.md SECURITY.md .gitignore; do
    [ ! -f "$f" ] && MISSING="$MISSING $f"
done
if [ -z "$MISSING" ]; then
    echo -e "${GREEN}OK${NC}"
    ((PASS++))
else
    echo -e "${RED}FAIL${NC} (missing:$MISSING)"
    ((FAIL++))
fi

# ---- Check 11: No plan/ or .backup/ dirs ----
echo -n "13. No internal dirs exposed... "
INTERNAL=""
[ -d "plan" ] && INTERNAL="$INTERNAL plan/"
[ -d ".backup" ] && INTERNAL="$INTERNAL .backup/"
if [ -z "$INTERNAL" ]; then
    echo -e "${GREEN}OK${NC}"
    ((PASS++))
else
    echo -e "${YELLOW}WARN${NC} (found:$INTERNAL — ensure .gitignore covers them)"
    ((WARN++))
fi

# ---- Check 12: File size cap ----
echo -n "14. 800-line cap (source files)... "
OVER_800=$(find src/ -name "*.py" -exec wc -l {} + 2>/dev/null | awk '$1 > 800 && !/total/' | wc -l | tr -d ' ')
if [ "$OVER_800" -eq 0 ]; then
    echo -e "${GREEN}OK${NC} (all under 800 lines)"
    ((PASS++))
else
    echo -e "${RED}FAIL${NC} ($OVER_800 files over 800 lines)"
    ((FAIL++))
    find src/ -name "*.py" -exec wc -l {} + 2>/dev/null | awk '$1 > 800 && !/total/' | sed 's/^/   /'
fi

# ---- Check 13: URLs point to qualixar ----
echo -n "15. URLs point to qualixar org... "
OLD_URLS=$(grep -r "varun369/agentassay" pyproject.toml README.md 2>/dev/null | wc -l | tr -d ' ')
if [ "$OLD_URLS" -eq 0 ]; then
    echo -e "${GREEN}OK${NC}"
    ((PASS++))
else
    echo -e "${YELLOW}WARN${NC} ($OLD_URLS references to varun369 — update to qualixar)"
    ((WARN++))
fi

# ---- Check 14: Paper PDF exists ----
echo -n "16. Paper PDF present... "
if [ -f "paper/main.pdf" ]; then
    PDF_SIZE=$(du -h paper/main.pdf | cut -f1)
    echo -e "${GREEN}OK${NC} ($PDF_SIZE)"
    ((PASS++))
else
    echo -e "${YELLOW}WARN${NC} (paper/main.pdf not found)"
    ((WARN++))
fi

# ---- SUMMARY ----
echo ""
echo "============================================"
TOTAL=$((PASS + FAIL + WARN))
echo -e "  Results: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}, ${YELLOW}$WARN warnings${NC} / $TOTAL checks"
echo "============================================"

if [ "$FAIL" -gt 0 ]; then
    echo -e ""
    echo -e "${RED}  LAUNCH BLOCKED — Fix $FAIL failure(s) before pushing to GitHub${NC}"
    echo ""
    exit 1
elif [ "$WARN" -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}  LAUNCH OK WITH WARNINGS — Review $WARN warning(s) before pushing${NC}"
    echo ""
    exit 0
else
    echo ""
    echo -e "${GREEN}  ALL CLEAR — Ready for GitHub push and PyPI publish${NC}"
    echo ""
    exit 0
fi
