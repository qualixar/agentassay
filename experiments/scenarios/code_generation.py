"""Code generation agent scenario for AgentAssay experiments.

AI coding assistant agent that takes a problem specification, writes
Python code, executes it, runs tests, and iterates until tests pass.
Models the modern AI coding workflow (vibe coding with verification).

Agent workflow:
    1. Receive coding problem specification
    2. Write solution code to a file
    3. Write test cases
    4. Execute tests
    5. If tests fail, read errors, fix code, re-run (up to 3 iterations)
    6. Return final code and test results

Regression injection points:
    - Remove "write tests" instruction (skip verification)
    - Limit to 1 tool call (no iteration / debugging loop)
    - Add misleading code example (hallucination trigger)
    - Disable file reading (cannot inspect errors)
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Callable
from typing import Any

from agentassay.core.models import (
    ExecutionTrace,
    StepTrace,
    TestScenario,
)


# ===================================================================
# Virtual File System (in-memory)
# ===================================================================

_VIRTUAL_FS: dict[str, str] = {}


def _reset_fs() -> None:
    """Reset virtual file system between runs."""
    _VIRTUAL_FS.clear()


# ===================================================================
# Predefined Solutions & Tests (for deterministic mock)
# ===================================================================

_SOLUTIONS: dict[str, dict[str, str]] = {
    "fizzbuzz": {
        "solution": textwrap.dedent("""\
            def fizzbuzz(n: int) -> list[str]:
                result = []
                for i in range(1, n + 1):
                    if i % 15 == 0:
                        result.append("FizzBuzz")
                    elif i % 3 == 0:
                        result.append("Fizz")
                    elif i % 5 == 0:
                        result.append("Buzz")
                    else:
                        result.append(str(i))
                return result
        """),
        "tests": textwrap.dedent("""\
            from solution import fizzbuzz

            def test_fizzbuzz_15():
                result = fizzbuzz(15)
                assert result[0] == "1"
                assert result[2] == "Fizz"
                assert result[4] == "Buzz"
                assert result[14] == "FizzBuzz"
                assert len(result) == 15

            def test_fizzbuzz_1():
                assert fizzbuzz(1) == ["1"]

            def test_fizzbuzz_edge():
                assert fizzbuzz(0) == []
        """),
        "expected_output": "3 passed",
    },
    "palindrome": {
        "solution": textwrap.dedent("""\
            def is_palindrome(s: str) -> bool:
                cleaned = ''.join(c.lower() for c in s if c.isalnum())
                return cleaned == cleaned[::-1]
        """),
        "tests": textwrap.dedent("""\
            from solution import is_palindrome

            def test_simple_palindrome():
                assert is_palindrome("racecar") is True

            def test_not_palindrome():
                assert is_palindrome("hello") is False

            def test_with_spaces():
                assert is_palindrome("A man a plan a canal Panama") is True

            def test_with_punctuation():
                assert is_palindrome("Was it a car or a cat I saw?") is True

            def test_empty_string():
                assert is_palindrome("") is True

            def test_single_char():
                assert is_palindrome("a") is True
        """),
        "expected_output": "6 passed",
    },
    "binary_search": {
        "solution": textwrap.dedent("""\
            def binary_search(arr: list[int], target: int) -> int:
                left, right = 0, len(arr) - 1
                while left <= right:
                    mid = (left + right) // 2
                    if arr[mid] == target:
                        return mid
                    elif arr[mid] < target:
                        left = mid + 1
                    else:
                        right = mid - 1
                return -1
        """),
        "tests": textwrap.dedent("""\
            from solution import binary_search

            def test_found():
                assert binary_search([1, 3, 5, 7, 9], 5) == 2

            def test_not_found():
                assert binary_search([1, 3, 5, 7, 9], 4) == -1

            def test_first_element():
                assert binary_search([1, 3, 5], 1) == 0

            def test_last_element():
                assert binary_search([1, 3, 5], 5) == 2

            def test_empty():
                assert binary_search([], 1) == -1

            def test_single():
                assert binary_search([42], 42) == 0
        """),
        "expected_output": "6 passed",
    },
    "linked_list": {
        "solution": textwrap.dedent("""\
            class Node:
                def __init__(self, val: int, next_node: 'Node | None' = None):
                    self.val = val
                    self.next = next_node

            class LinkedList:
                def __init__(self):
                    self.head: Node | None = None

                def append(self, val: int) -> None:
                    if self.head is None:
                        self.head = Node(val)
                        return
                    current = self.head
                    while current.next:
                        current = current.next
                    current.next = Node(val)

                def to_list(self) -> list[int]:
                    result = []
                    current = self.head
                    while current:
                        result.append(current.val)
                        current = current.next
                    return result

                def reverse(self) -> None:
                    prev = None
                    current = self.head
                    while current:
                        next_node = current.next
                        current.next = prev
                        prev = current
                        current = next_node
                    self.head = prev

                def __len__(self) -> int:
                    count = 0
                    current = self.head
                    while current:
                        count += 1
                        current = current.next
                    return count
        """),
        "tests": textwrap.dedent("""\
            from solution import LinkedList

            def test_append_and_list():
                ll = LinkedList()
                ll.append(1)
                ll.append(2)
                ll.append(3)
                assert ll.to_list() == [1, 2, 3]

            def test_reverse():
                ll = LinkedList()
                ll.append(1)
                ll.append(2)
                ll.append(3)
                ll.reverse()
                assert ll.to_list() == [3, 2, 1]

            def test_empty():
                ll = LinkedList()
                assert ll.to_list() == []
                assert len(ll) == 0

            def test_len():
                ll = LinkedList()
                ll.append(10)
                ll.append(20)
                assert len(ll) == 2
        """),
        "expected_output": "4 passed",
    },
    "sorting": {
        "solution": textwrap.dedent("""\
            def merge_sort(arr: list[int]) -> list[int]:
                if len(arr) <= 1:
                    return arr
                mid = len(arr) // 2
                left = merge_sort(arr[:mid])
                right = merge_sort(arr[mid:])
                return _merge(left, right)

            def _merge(left: list[int], right: list[int]) -> list[int]:
                result = []
                i = j = 0
                while i < len(left) and j < len(right):
                    if left[i] <= right[j]:
                        result.append(left[i])
                        i += 1
                    else:
                        result.append(right[j])
                        j += 1
                result.extend(left[i:])
                result.extend(right[j:])
                return result
        """),
        "tests": textwrap.dedent("""\
            from solution import merge_sort

            def test_basic():
                assert merge_sort([3, 1, 4, 1, 5]) == [1, 1, 3, 4, 5]

            def test_sorted():
                assert merge_sort([1, 2, 3]) == [1, 2, 3]

            def test_reversed():
                assert merge_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]

            def test_empty():
                assert merge_sort([]) == []

            def test_single():
                assert merge_sort([42]) == [42]

            def test_duplicates():
                assert merge_sort([3, 3, 3]) == [3, 3, 3]
        """),
        "expected_output": "6 passed",
    },
    "api_client": {
        "solution": textwrap.dedent("""\
            import json
            from dataclasses import dataclass
            from typing import Any

            @dataclass
            class APIResponse:
                status_code: int
                body: dict[str, Any]
                success: bool

            class SimpleAPIClient:
                def __init__(self, base_url: str):
                    self.base_url = base_url.rstrip('/')
                    self.headers: dict[str, str] = {}
                    self._mock_responses: dict[str, APIResponse] = {}

                def set_header(self, key: str, value: str) -> None:
                    self.headers[key] = value

                def mock_response(self, endpoint: str, response: APIResponse) -> None:
                    self._mock_responses[endpoint] = response

                def get(self, endpoint: str) -> APIResponse:
                    if endpoint in self._mock_responses:
                        return self._mock_responses[endpoint]
                    return APIResponse(status_code=404, body={"error": "not found"}, success=False)

                def post(self, endpoint: str, data: dict) -> APIResponse:
                    if endpoint in self._mock_responses:
                        return self._mock_responses[endpoint]
                    return APIResponse(status_code=201, body=data, success=True)

                def build_url(self, endpoint: str) -> str:
                    return f"{self.base_url}/{endpoint.lstrip('/')}"
        """),
        "tests": textwrap.dedent("""\
            from solution import SimpleAPIClient, APIResponse

            def test_build_url():
                client = SimpleAPIClient("https://api.example.com")
                assert client.build_url("/users") == "https://api.example.com/users"

            def test_set_header():
                client = SimpleAPIClient("https://api.example.com")
                client.set_header("Authorization", "Bearer token123")
                assert client.headers["Authorization"] == "Bearer token123"

            def test_get_mock():
                client = SimpleAPIClient("https://api.example.com")
                client.mock_response("/users", APIResponse(200, {"users": []}, True))
                resp = client.get("/users")
                assert resp.status_code == 200
                assert resp.success is True

            def test_get_not_found():
                client = SimpleAPIClient("https://api.example.com")
                resp = client.get("/nonexistent")
                assert resp.status_code == 404
                assert resp.success is False

            def test_post():
                client = SimpleAPIClient("https://api.example.com")
                resp = client.post("/users", {"name": "Test"})
                assert resp.status_code == 201
        """),
        "expected_output": "5 passed",
    },
    "data_validator": {
        "solution": textwrap.dedent("""\
            import re
            from typing import Any

            class ValidationError(Exception):
                def __init__(self, field: str, message: str):
                    self.field = field
                    self.message = message
                    super().__init__(f"{field}: {message}")

            class Validator:
                @staticmethod
                def validate_email(email: str) -> bool:
                    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
                    return bool(re.match(pattern, email))

                @staticmethod
                def validate_phone(phone: str) -> bool:
                    cleaned = re.sub(r'[\\s\\-\\(\\)\\+]', '', phone)
                    return cleaned.isdigit() and 10 <= len(cleaned) <= 15

                @staticmethod
                def validate_age(age: Any) -> bool:
                    if not isinstance(age, int):
                        return False
                    return 0 <= age <= 150

                @staticmethod
                def validate_password(password: str) -> tuple[bool, list[str]]:
                    errors = []
                    if len(password) < 8:
                        errors.append("Must be at least 8 characters")
                    if not re.search(r'[A-Z]', password):
                        errors.append("Must contain uppercase letter")
                    if not re.search(r'[a-z]', password):
                        errors.append("Must contain lowercase letter")
                    if not re.search(r'\\d', password):
                        errors.append("Must contain digit")
                    return len(errors) == 0, errors
        """),
        "tests": textwrap.dedent("""\
            from solution import Validator, ValidationError

            def test_valid_email():
                assert Validator.validate_email("user@example.com") is True

            def test_invalid_email():
                assert Validator.validate_email("not-an-email") is False

            def test_valid_phone():
                assert Validator.validate_phone("+1 (555) 123-4567") is True

            def test_invalid_phone():
                assert Validator.validate_phone("123") is False

            def test_valid_age():
                assert Validator.validate_age(25) is True

            def test_invalid_age():
                assert Validator.validate_age(-1) is False
                assert Validator.validate_age(200) is False

            def test_strong_password():
                valid, errors = Validator.validate_password("Str0ngP@ss")
                assert valid is True
                assert errors == []

            def test_weak_password():
                valid, errors = Validator.validate_password("weak")
                assert valid is False
                assert len(errors) > 0
        """),
        "expected_output": "8 passed",
    },
    "rate_limiter": {
        "solution": textwrap.dedent("""\
            import time
            from collections import defaultdict

            class RateLimiter:
                def __init__(self, max_requests: int, window_seconds: float):
                    self.max_requests = max_requests
                    self.window_seconds = window_seconds
                    self._requests: dict[str, list[float]] = defaultdict(list)

                def allow(self, client_id: str) -> bool:
                    now = time.monotonic()
                    cutoff = now - self.window_seconds
                    self._requests[client_id] = [
                        t for t in self._requests[client_id] if t > cutoff
                    ]
                    if len(self._requests[client_id]) >= self.max_requests:
                        return False
                    self._requests[client_id].append(now)
                    return True

                def remaining(self, client_id: str) -> int:
                    now = time.monotonic()
                    cutoff = now - self.window_seconds
                    active = [t for t in self._requests[client_id] if t > cutoff]
                    return max(0, self.max_requests - len(active))

                def reset(self, client_id: str) -> None:
                    self._requests[client_id] = []
        """),
        "tests": textwrap.dedent("""\
            from solution import RateLimiter

            def test_allow_under_limit():
                rl = RateLimiter(max_requests=3, window_seconds=60)
                assert rl.allow("client1") is True
                assert rl.allow("client1") is True
                assert rl.allow("client1") is True

            def test_deny_over_limit():
                rl = RateLimiter(max_requests=2, window_seconds=60)
                assert rl.allow("client1") is True
                assert rl.allow("client1") is True
                assert rl.allow("client1") is False

            def test_different_clients():
                rl = RateLimiter(max_requests=1, window_seconds=60)
                assert rl.allow("client1") is True
                assert rl.allow("client2") is True

            def test_remaining():
                rl = RateLimiter(max_requests=5, window_seconds=60)
                rl.allow("client1")
                rl.allow("client1")
                assert rl.remaining("client1") == 3

            def test_reset():
                rl = RateLimiter(max_requests=1, window_seconds=60)
                rl.allow("client1")
                assert rl.allow("client1") is False
                rl.reset("client1")
                assert rl.allow("client1") is True
        """),
        "expected_output": "5 passed",
    },
    "retry_decorator": {
        "solution": textwrap.dedent("""\
            import time
            import functools
            from typing import TypeVar, Callable, Any

            F = TypeVar('F', bound=Callable[..., Any])

            class RetryExhausted(Exception):
                def __init__(self, attempts: int, last_error: Exception):
                    self.attempts = attempts
                    self.last_error = last_error
                    super().__init__(f"Failed after {attempts} attempts: {last_error}")

            def retry(
                max_attempts: int = 3,
                delay_seconds: float = 0.0,
                exceptions: tuple[type[Exception], ...] = (Exception,),
            ) -> Callable[[F], F]:
                def decorator(func: F) -> F:
                    @functools.wraps(func)
                    def wrapper(*args: Any, **kwargs: Any) -> Any:
                        last_exc: Exception | None = None
                        for attempt in range(1, max_attempts + 1):
                            try:
                                return func(*args, **kwargs)
                            except exceptions as e:
                                last_exc = e
                                if attempt < max_attempts and delay_seconds > 0:
                                    time.sleep(delay_seconds)
                        raise RetryExhausted(max_attempts, last_exc)  # type: ignore
                    return wrapper  # type: ignore
                return decorator
        """),
        "tests": textwrap.dedent("""\
            from solution import retry, RetryExhausted

            def test_success_first_try():
                @retry(max_attempts=3)
                def always_works():
                    return 42
                assert always_works() == 42

            def test_success_after_retries():
                call_count = 0
                @retry(max_attempts=3)
                def fails_twice():
                    nonlocal call_count
                    call_count += 1
                    if call_count < 3:
                        raise ValueError("not yet")
                    return "ok"
                assert fails_twice() == "ok"
                assert call_count == 3

            def test_exhausted():
                @retry(max_attempts=2)
                def always_fails():
                    raise RuntimeError("boom")
                try:
                    always_fails()
                    assert False, "Should have raised"
                except RetryExhausted as e:
                    assert e.attempts == 2

            def test_specific_exception():
                @retry(max_attempts=3, exceptions=(ValueError,))
                def raises_type_error():
                    raise TypeError("wrong type")
                try:
                    raises_type_error()
                    assert False, "Should have raised"
                except TypeError:
                    pass  # Expected - TypeError is not retried
        """),
        "expected_output": "4 passed",
    },
    "file_parser": {
        "solution": textwrap.dedent("""\
            import csv
            import json
            from io import StringIO
            from typing import Any

            class FileParser:
                @staticmethod
                def parse_csv(content: str, has_header: bool = True) -> list[dict[str, str]] | list[list[str]]:
                    reader = csv.reader(StringIO(content))
                    rows = list(reader)
                    if not rows:
                        return []
                    if has_header:
                        headers = rows[0]
                        return [dict(zip(headers, row)) for row in rows[1:]]
                    return rows

                @staticmethod
                def parse_json(content: str) -> Any:
                    return json.loads(content)

                @staticmethod
                def parse_key_value(content: str, delimiter: str = "=") -> dict[str, str]:
                    result = {}
                    for line in content.strip().split("\\n"):
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if delimiter in line:
                            key, _, value = line.partition(delimiter)
                            result[key.strip()] = value.strip()
                    return result
        """),
        "tests": textwrap.dedent("""\
            from solution import FileParser

            def test_csv_with_header():
                csv_data = "name,age\\nAlice,30\\nBob,25"
                result = FileParser.parse_csv(csv_data)
                assert result == [{"name": "Alice", "age": "30"}, {"name": "Bob", "age": "25"}]

            def test_csv_without_header():
                csv_data = "Alice,30\\nBob,25"
                result = FileParser.parse_csv(csv_data, has_header=False)
                assert result == [["Alice", "30"], ["Bob", "25"]]

            def test_csv_empty():
                assert FileParser.parse_csv("") == []

            def test_json():
                data = '{"key": "value", "num": 42}'
                result = FileParser.parse_json(data)
                assert result == {"key": "value", "num": 42}

            def test_key_value():
                content = "host=localhost\\nport=8080\\n# comment\\ndb=mydb"
                result = FileParser.parse_key_value(content)
                assert result == {"host": "localhost", "port": "8080", "db": "mydb"}
        """),
        "expected_output": "5 passed",
    },
}


# ===================================================================
# Mock Tool Implementations
# ===================================================================


def write_file(filename: str, content: str) -> dict[str, Any]:
    """Write content to a virtual file."""
    _VIRTUAL_FS[filename] = content
    return {
        "success": True,
        "filename": filename,
        "size_bytes": len(content),
        "lines": content.count("\n") + 1,
    }


def read_file(filename: str) -> dict[str, Any]:
    """Read content from a virtual file."""
    if filename not in _VIRTUAL_FS:
        return {
            "success": False,
            "error": f"File not found: {filename}",
        }
    content = _VIRTUAL_FS[filename]
    return {
        "success": True,
        "filename": filename,
        "content": content,
        "size_bytes": len(content),
        "lines": content.count("\n") + 1,
    }


def execute_code(filename: str) -> dict[str, Any]:
    """Execute a virtual Python file (simulated)."""
    if filename not in _VIRTUAL_FS:
        return {
            "success": False,
            "exit_code": 1,
            "stdout": "",
            "stderr": f"FileNotFoundError: {filename}",
        }

    content = _VIRTUAL_FS[filename]

    # Basic syntax check simulation
    try:
        compile(content, filename, "exec")
    except SyntaxError as e:
        return {
            "success": False,
            "exit_code": 1,
            "stdout": "",
            "stderr": f"SyntaxError: {e.msg} (line {e.lineno})",
        }

    return {
        "success": True,
        "exit_code": 0,
        "stdout": "Execution completed successfully.",
        "stderr": "",
    }


def run_tests(test_filename: str) -> dict[str, Any]:
    """Run tests from a virtual test file (simulated)."""
    if test_filename not in _VIRTUAL_FS:
        return {
            "success": False,
            "exit_code": 1,
            "stdout": "",
            "stderr": f"FileNotFoundError: {test_filename}",
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
        }

    test_content = _VIRTUAL_FS[test_filename]
    # Count test functions
    test_funcs = re.findall(r"def (test_\w+)", test_content)
    total_tests = len(test_funcs)

    # Check if solution file exists
    solution_exists = "solution.py" in _VIRTUAL_FS

    if not solution_exists:
        return {
            "success": False,
            "exit_code": 1,
            "stdout": "",
            "stderr": "ModuleNotFoundError: No module named 'solution'",
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": total_tests,
        }

    # Simulate all tests passing for correct solutions
    return {
        "success": True,
        "exit_code": 0,
        "stdout": f"{total_tests} passed in 0.15s",
        "stderr": "",
        "tests_run": total_tests,
        "tests_passed": total_tests,
        "tests_failed": 0,
    }


# ===================================================================
# Tool Schemas
# ===================================================================

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a Python file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Filename to write"},
                    "content": {"type": "string", "description": "File content to write"},
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Filename to read"},
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Execute a Python file and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Python file to execute"},
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run pytest on a test file and return results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_filename": {"type": "string", "description": "Test file to run"},
                },
                "required": ["test_filename"],
            },
        },
    },
]


# ===================================================================
# System Prompts
# ===================================================================

SYSTEM_PROMPT = """You are an expert Python developer. Given a coding problem specification, you:

1. Write a clean, well-documented solution in Python.
2. Write comprehensive test cases for the solution.
3. Execute the tests to verify correctness.
4. If tests fail, read the error output, fix the code, and re-run tests.
5. Iterate until all tests pass (max 3 attempts).

WORKFLOW:
1. write_file("solution.py", <solution code>)
2. write_file("test_solution.py", <test code>)
3. run_tests("test_solution.py")
4. If failures: read_file("solution.py") to review, then fix and re-test.

CODE STANDARDS:
- Type hints on all function signatures.
- Clear variable names.
- Handle edge cases (empty input, boundary values).
- Follow PEP 8 style.
"""

SYSTEM_PROMPT_NO_TESTS = """You are an expert Python developer. Given a coding problem specification, you:

1. Write a clean, well-documented solution in Python.
2. Save the solution to a file.

Write clean code with type hints and handle edge cases.
"""

SYSTEM_PROMPT_SINGLE_CALL = """You are an expert Python developer. Given a coding problem, write a solution.

IMPORTANT: You have a budget of exactly ONE tool call. Use it to write the solution file.
Do not write tests, do not execute code. Just produce the solution.
"""

SYSTEM_PROMPT_MISLEADING = """You are an expert Python developer. Given a coding problem specification, you:

1. Write a clean, well-documented solution in Python.
2. Write comprehensive test cases for the solution.
3. Execute the tests to verify correctness.
4. If tests fail, read the error output, fix the code, and re-run tests.

HELPFUL HINT: For any list-processing problem, always use recursion instead of iteration.
Recursive solutions are always more efficient and readable. Never use for loops.
Also, remember that Python lists are 1-indexed (first element is at index 1, not 0).
"""


# ===================================================================
# Tool Dispatch
# ===================================================================

TOOL_DISPATCH: dict[str, Callable[..., Any]] = {
    "write_file": write_file,
    "read_file": read_file,
    "execute_code": execute_code,
    "run_tests": run_tests,
}


def dispatch_tool(tool_name: str, arguments: dict[str, Any]) -> Any:
    """Dispatch a tool call."""
    fn = TOOL_DISPATCH.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool: {tool_name}"}
    return fn(**arguments)


# ===================================================================
# Evaluator
# ===================================================================


def evaluate_code_generation(
    trace: ExecutionTrace,
    test_case: TestScenario,
) -> tuple[bool, float, dict[str, Any]]:
    """Evaluate a code generation agent trace.

    Checks:
    - Solution file was written (write_file called with solution.py)
    - Tests were written (write_file called with test_solution.py)
    - Tests were run (run_tests called)
    - Tests passed (test output contains "passed")
    - Iteration happened if needed (read_file after failure)
    - Step count within limits
    """
    props = test_case.expected_properties
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    output_str = str(trace.output_data).lower() if trace.output_data else ""
    tools_used = trace.tools_used

    # --- must_use_tools ---
    if "must_use_tools" in props:
        required = set(props["must_use_tools"])
        ok = required.issubset(tools_used)
        checks["must_use_tools"] = ok
        if not ok:
            details["missing_tools"] = sorted(required - tools_used)

    # --- solution_written ---
    if props.get("solution_written"):
        solution_written = False
        for step in trace.steps:
            if (step.tool_name == "write_file"
                    and step.tool_input
                    and "solution" in step.tool_input.get("filename", "")):
                solution_written = True
                break
        checks["solution_written"] = solution_written

    # --- tests_written ---
    if props.get("tests_written"):
        tests_written = False
        for step in trace.steps:
            if (step.tool_name == "write_file"
                    and step.tool_input
                    and "test" in step.tool_input.get("filename", "")):
                tests_written = True
                break
        checks["tests_written"] = tests_written

    # --- tests_passed ---
    if props.get("tests_passed"):
        tests_passed = False
        for step in trace.steps:
            if step.tool_name == "run_tests" and step.tool_output:
                if step.tool_output.get("success") and step.tool_output.get("tests_failed", 0) == 0:
                    tests_passed = True
                    break
        checks["tests_passed"] = tests_passed

    # --- output_contains ---
    if "output_contains" in props:
        needles = props["output_contains"]
        if isinstance(needles, str):
            needles = [needles]
        found = {n: n.lower() in output_str for n in needles}
        checks["output_contains"] = all(found.values())

    # --- max_steps ---
    if "max_steps" in props:
        checks["max_steps"] = trace.step_count <= int(props["max_steps"])

    if not checks:
        return trace.success, 1.0 if trace.success else 0.0, {"reason": "no checks"}

    all_passed = all(checks.values())
    score = sum(checks.values()) / len(checks)
    return all_passed, score, {"checks": checks, "details": details}


# ===================================================================
# Agent Function Helper
# ===================================================================


def _build_step(
    index: int,
    action: str,
    tool_name: str | None = None,
    tool_input: dict[str, Any] | None = None,
    tool_output: Any = None,
    llm_input: str | None = None,
    llm_output: str | None = None,
    model: str = "gpt-4o",
    duration_ms: float = 100.0,
) -> StepTrace:
    return StepTrace(
        step_index=index,
        action=action,
        tool_name=tool_name,
        tool_input=tool_input,
        tool_output=tool_output,
        llm_input=llm_input,
        llm_output=llm_output,
        model=model,
        duration_ms=duration_ms,
    )


# ===================================================================
# Agent Function (Mock)
# ===================================================================


def run_code_generation_agent(
    input_data: dict[str, Any],
    *,
    system_prompt: str = SYSTEM_PROMPT,
    available_tools: dict[str, Callable[..., Any]] | None = None,
    model: str = "gpt-4o",
) -> ExecutionTrace:
    """Simulate a code generation agent execution."""
    if available_tools is None:
        available_tools = TOOL_DISPATCH.copy()

    _reset_fs()

    problem_id = input_data.get("problem_id", "fizzbuzz")
    problem_spec = input_data.get("problem_spec", "")

    solution_data = _SOLUTIONS.get(problem_id, _SOLUTIONS["fizzbuzz"])
    steps: list[StepTrace] = []
    step_idx = 0
    total_duration = 0.0
    output_parts: list[str] = []

    # Step 0: LLM analyzes problem
    steps.append(_build_step(
        index=step_idx,
        action="llm_response",
        llm_input=f"System: {system_prompt[:200]}...\nProblem: {problem_spec}",
        llm_output="[Analyzing problem and planning solution]",
        model=model,
        duration_ms=250.0,
    ))
    step_idx += 1
    total_duration += 250.0

    # Step 1: Write solution
    if "write_file" in available_tools:
        result = write_file("solution.py", solution_data["solution"])
        steps.append(_build_step(
            index=step_idx,
            action="tool_call",
            tool_name="write_file",
            tool_input={"filename": "solution.py", "content": solution_data["solution"]},
            tool_output=result,
            model=model,
            duration_ms=300.0,
        ))
        step_idx += 1
        total_duration += 300.0
        output_parts.append(f"Wrote solution.py ({result['lines']} lines)")

    # Step 2: Write tests (if instructed)
    write_tests = "test" in system_prompt.lower() and "write_file" in available_tools
    if write_tests:
        result = write_file("test_solution.py", solution_data["tests"])
        steps.append(_build_step(
            index=step_idx,
            action="tool_call",
            tool_name="write_file",
            tool_input={"filename": "test_solution.py", "content": solution_data["tests"]},
            tool_output=result,
            model=model,
            duration_ms=250.0,
        ))
        step_idx += 1
        total_duration += 250.0
        output_parts.append(f"Wrote test_solution.py ({result['lines']} lines)")

    # Step 3: Run tests (if instructed)
    run_tests_step = "test" in system_prompt.lower() and "run_tests" in available_tools
    if run_tests_step and "test_solution.py" in _VIRTUAL_FS:
        result = run_tests("test_solution.py")
        steps.append(_build_step(
            index=step_idx,
            action="tool_call",
            tool_name="run_tests",
            tool_input={"test_filename": "test_solution.py"},
            tool_output=result,
            model=model,
            duration_ms=200.0,
        ))
        step_idx += 1
        total_duration += 200.0

        if result.get("success"):
            output_parts.append(f"All tests passed: {result['stdout']}")
        else:
            output_parts.append(f"Tests failed: {result['stderr']}")

    # Final LLM summary
    final_output = " | ".join(output_parts) if output_parts else "Solution written."
    steps.append(_build_step(
        index=step_idx,
        action="llm_response",
        llm_output=final_output,
        model=model,
        duration_ms=150.0,
    ))
    total_duration += 150.0

    estimated_cost = len(steps) * 0.003

    return ExecutionTrace(
        scenario_id=input_data.get("_scenario_id", "code_generation"),
        steps=steps,
        input_data=input_data,
        output_data=final_output,
        success=True,
        total_duration_ms=total_duration,
        total_cost_usd=round(estimated_cost, 4),
        model=model,
        framework="custom",
    )


# ===================================================================
# Test Cases (10 scenarios)
# ===================================================================

TEST_CASES: list[TestScenario] = [
    TestScenario(
        scenario_id="cg-001-fizzbuzz",
        name="FizzBuzz implementation",
        description="Classic FizzBuzz — write, test, verify.",
        input_data={
            "problem_id": "fizzbuzz",
            "problem_spec": "Write a function fizzbuzz(n) that returns a list of strings from 1 to n with FizzBuzz rules.",
        },
        expected_properties={
            "must_use_tools": ["write_file", "run_tests"],
            "solution_written": True,
            "tests_written": True,
            "tests_passed": True,
            "output_contains": ["passed"],
            "max_steps": 12,
        },
        evaluator="code_generation",
        tags=["basic", "happy-path"],
    ),
    TestScenario(
        scenario_id="cg-002-palindrome",
        name="Palindrome checker",
        description="Write is_palindrome(s) handling spaces, punctuation, case.",
        input_data={
            "problem_id": "palindrome",
            "problem_spec": "Write a function is_palindrome(s) that checks if a string is a palindrome, ignoring spaces, punctuation, and case.",
        },
        expected_properties={
            "must_use_tools": ["write_file", "run_tests"],
            "solution_written": True,
            "tests_written": True,
            "tests_passed": True,
            "output_contains": ["passed"],
            "max_steps": 12,
        },
        evaluator="code_generation",
        tags=["string", "edge-cases"],
    ),
    TestScenario(
        scenario_id="cg-003-binary-search",
        name="Binary search implementation",
        description="Write binary_search(arr, target) returning index or -1.",
        input_data={
            "problem_id": "binary_search",
            "problem_spec": "Write binary_search(arr, target) that returns the index of target in sorted array, or -1 if not found.",
        },
        expected_properties={
            "must_use_tools": ["write_file", "run_tests"],
            "solution_written": True,
            "tests_written": True,
            "tests_passed": True,
            "max_steps": 12,
        },
        evaluator="code_generation",
        tags=["algorithm", "search"],
    ),
    TestScenario(
        scenario_id="cg-004-linked-list",
        name="Linked list with reverse",
        description="Implement a linked list class with append, reverse, to_list, and len.",
        input_data={
            "problem_id": "linked_list",
            "problem_spec": "Implement a LinkedList class with append(val), reverse(), to_list(), and __len__ methods.",
        },
        expected_properties={
            "must_use_tools": ["write_file", "run_tests"],
            "solution_written": True,
            "tests_written": True,
            "tests_passed": True,
            "max_steps": 12,
        },
        evaluator="code_generation",
        tags=["data-structure", "oop"],
    ),
    TestScenario(
        scenario_id="cg-005-merge-sort",
        name="Merge sort algorithm",
        description="Implement merge sort with proper merge function.",
        input_data={
            "problem_id": "sorting",
            "problem_spec": "Implement merge_sort(arr) that returns a new sorted list using the merge sort algorithm.",
        },
        expected_properties={
            "must_use_tools": ["write_file", "run_tests"],
            "solution_written": True,
            "tests_written": True,
            "tests_passed": True,
            "max_steps": 12,
        },
        evaluator="code_generation",
        tags=["algorithm", "sorting"],
    ),
    TestScenario(
        scenario_id="cg-006-api-client",
        name="Simple API client class",
        description="Build an API client with get/post, headers, URL building, and mock responses.",
        input_data={
            "problem_id": "api_client",
            "problem_spec": "Build a SimpleAPIClient class with base_url, set_header, get(endpoint), post(endpoint, data), and build_url methods. Include mock response capability for testing.",
        },
        expected_properties={
            "must_use_tools": ["write_file", "run_tests"],
            "solution_written": True,
            "tests_written": True,
            "tests_passed": True,
            "max_steps": 12,
        },
        evaluator="code_generation",
        tags=["oop", "api", "networking"],
    ),
    TestScenario(
        scenario_id="cg-007-file-parser",
        name="Multi-format file parser",
        description="Build a FileParser class that parses CSV, JSON, and key-value config files.",
        input_data={
            "problem_id": "file_parser",
            "problem_spec": "Build a FileParser class with parse_csv(content, has_header), parse_json(content), and parse_key_value(content, delimiter) methods.",
        },
        expected_properties={
            "must_use_tools": ["write_file", "run_tests"],
            "solution_written": True,
            "tests_written": True,
            "tests_passed": True,
            "max_steps": 12,
        },
        evaluator="code_generation",
        tags=["parsing", "files"],
    ),
    TestScenario(
        scenario_id="cg-008-data-validator",
        name="Data validation utility",
        description="Build a Validator class for email, phone, age, and password validation.",
        input_data={
            "problem_id": "data_validator",
            "problem_spec": "Build a Validator class with validate_email, validate_phone, validate_age, and validate_password static methods. Password validation returns (bool, list[str]) with error messages.",
        },
        expected_properties={
            "must_use_tools": ["write_file", "run_tests"],
            "solution_written": True,
            "tests_written": True,
            "tests_passed": True,
            "max_steps": 12,
        },
        evaluator="code_generation",
        tags=["validation", "utility"],
    ),
    TestScenario(
        scenario_id="cg-009-rate-limiter",
        name="Rate limiter implementation",
        description="Build a sliding window rate limiter with per-client tracking.",
        input_data={
            "problem_id": "rate_limiter",
            "problem_spec": "Build a RateLimiter class with allow(client_id) -> bool, remaining(client_id) -> int, and reset(client_id) methods. Use sliding window algorithm.",
        },
        expected_properties={
            "must_use_tools": ["write_file", "run_tests"],
            "solution_written": True,
            "tests_written": True,
            "tests_passed": True,
            "max_steps": 12,
        },
        evaluator="code_generation",
        tags=["concurrency", "rate-limiting"],
    ),
    TestScenario(
        scenario_id="cg-010-retry-decorator",
        name="Retry decorator with backoff",
        description="Build a retry decorator with max attempts, delay, and exception filtering.",
        input_data={
            "problem_id": "retry_decorator",
            "problem_spec": "Build a @retry decorator with max_attempts, delay_seconds, and exceptions parameters. Raise RetryExhausted after all attempts fail.",
        },
        expected_properties={
            "must_use_tools": ["write_file", "run_tests"],
            "solution_written": True,
            "tests_written": True,
            "tests_passed": True,
            "max_steps": 12,
        },
        evaluator="code_generation",
        tags=["decorator", "resilience"],
    ),
]


# ===================================================================
# Regression Injection Points
# ===================================================================

REGRESSION_INJECTIONS: list[dict[str, Any]] = [
    {
        "id": "reg-cg-001",
        "name": "Remove test-writing instruction",
        "description": (
            "Agent writes solution but skips tests. Expected regression: "
            "no verification, potentially buggy code ships."
        ),
        "type": "prompt_mutation",
        "system_prompt": SYSTEM_PROMPT_NO_TESTS,
        "expected_regression_on": [f"cg-{i:03d}-{n}" for i, n in enumerate([
            "fizzbuzz", "palindrome", "binary-search", "linked-list",
            "merge-sort", "api-client", "file-parser", "data-validator",
            "rate-limiter", "retry-decorator",
        ], 1)],
    },
    {
        "id": "reg-cg-002",
        "name": "Limit to single tool call",
        "description": (
            "Agent can only write solution file. No tests, no execution, no iteration. "
            "Expected regression: no verification loop."
        ),
        "type": "prompt_mutation",
        "system_prompt": SYSTEM_PROMPT_SINGLE_CALL,
        "expected_regression_on": [f"cg-{i:03d}-{n}" for i, n in enumerate([
            "fizzbuzz", "palindrome", "binary-search", "linked-list",
            "merge-sort", "api-client", "file-parser", "data-validator",
            "rate-limiter", "retry-decorator",
        ], 1)],
    },
    {
        "id": "reg-cg-003",
        "name": "Inject misleading coding advice",
        "description": (
            "Add false advice (1-indexed lists, always use recursion). "
            "Expected regression: incorrect solutions, failed tests."
        ),
        "type": "prompt_mutation",
        "system_prompt": SYSTEM_PROMPT_MISLEADING,
        "expected_regression_on": [
            "cg-003-binary-search",
            "cg-005-merge-sort",
            "cg-004-linked-list",
        ],
    },
    {
        "id": "reg-cg-004",
        "name": "Disable file reading",
        "description": (
            "Remove read_file from tools. Agent cannot inspect errors or review code. "
            "Expected regression: cannot debug test failures."
        ),
        "type": "tool_removal",
        "removed_tools": ["read_file"],
        "expected_regression_on": [
            "cg-004-linked-list",
            "cg-006-api-client",
            "cg-008-data-validator",
        ],
    },
]


# ===================================================================
# Convenience Exports
# ===================================================================

SCENARIO_ID = "code_generation"
SCENARIO_NAME = "AI Code Generation Agent"
SCENARIO_DESCRIPTION = (
    "Code generation agent that writes Python solutions, creates tests, "
    "executes them, and iterates on failures. Covers 10 problem types "
    "from basic algorithms to production utilities."
)
