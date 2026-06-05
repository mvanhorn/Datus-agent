import argparse
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PROJECT_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_CONF_DIR = Path(__file__).parent / "conf"
RERUN_LOG_MAX_LINES = int(os.getenv("DATUS_RERUN_LOG_MAX_LINES", "80"))
RERUN_CAPTURE_LOG_MAX_LINES = int(os.getenv("DATUS_RERUN_CAPTURE_LOG_MAX_LINES", "40"))
_DATUS_RERUN_REPORTS: list[dict[str, object]] = []


@pytest.fixture
def mock_args():
    """Create a mock arguments object for testing."""
    args = argparse.Namespace(
        model="deepseek-v3",
        temperature=0.5,
        top_p=0.9,
        max_tokens=2500,
        task="Select all employees who earn more than $50,000",
        task_type="local",
        db_path="test_db.sqlite",
        schema_path="test_schema.sql",
        plan=True,
        max_steps=20,
        human_in_loop=False,
        output_dir="test_output",
    )
    return args


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.generate.return_value = "Generated text response"
    model.generate_with_json_output.return_value = {"result": "success"}
    model.generate_sql.return_value = "SELECT * FROM employees WHERE salary > 50000;"
    return model


# @pytest.fixture
# def sample_workflow():
#     """Create a sample workflow for testing."""
#     from datus.agent.workflow import Node, Workflow

#     workflow = Workflow("Test Workflow", "A workflow for testing")

#     # Add some tasks to the workflow
#     task1 = Node(
#         "task1",
#         "Parse the query",
#         "query_processing",
#         "Select all employees who earn more than $50,000",
#     )
#     task2 = Node("task2", "Generate SQL", "sql_generation", "Parsed query data")
#     task3 = Node(
#         "task3",
#         "Execute SQL",
#         "sql_execution",
#         "SELECT * FROM employees WHERE salary > 50000;",
#     )

#     workflow.add_task(task1)
#     workflow.add_task(task2)
#     workflow.add_task(task3)

#     return workflow


@pytest.fixture
def sample_database_schema():
    """Create a sample database schema for testing."""
    return """
    CREATE TABLE employees (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        department TEXT NOT NULL,
        salary REAL NOT NULL,
        hire_date TEXT NOT NULL
    );

    CREATE TABLE departments (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        budget REAL NOT NULL
    );
    """


@pytest.fixture
def sample_database_data():
    """Create sample database data for testing."""
    return [
        {
            "id": 1,
            "name": "John Doe",
            "department": "Engineering",
            "salary": 75000,
            "hire_date": "2020-01-15",
        },
        {
            "id": 2,
            "name": "Jane Smith",
            "department": "Marketing",
            "salary": 65000,
            "hire_date": "2019-05-20",
        },
        {
            "id": 3,
            "name": "Bob Johnson",
            "department": "Engineering",
            "salary": 85000,
            "hire_date": "2018-11-10",
        },
        {
            "id": 4,
            "name": "Alice Brown",
            "department": "HR",
            "salary": 45000,
            "hire_date": "2021-03-01",
        },
        {
            "id": 5,
            "name": "Charlie Wilson",
            "department": "Marketing",
            "salary": 55000,
            "hire_date": "2020-07-30",
        },
    ]


def load_acceptance_config(datasource: str = "snowflake", home: str = "") -> AgentConfig:
    return load_agent_config(
        config=str(TEST_CONF_DIR / "agent.yml"), datasource=datasource, home=home, reload=True, force=True, yes=True
    )


def _tail_lines(text: str, max_lines: int) -> list[str]:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return lines
    omitted = len(lines) - max_lines
    return [f"... omitted {omitted} earlier line(s) ...", *lines[-max_lines:]]


def _format_report_sections(sections: Iterable[tuple[str, str]]) -> list[str]:
    formatted: list[str] = []
    for name, content in sections:
        if not content:
            continue
        formatted.append(f"-- {name} --")
        formatted.extend(_tail_lines(content, RERUN_CAPTURE_LOG_MAX_LINES))
    return formatted


def _dump_async_tasks_to(out) -> None:
    """Write the stacks of all pending asyncio tasks to *out* (a text stream).

    Collects ``asyncio.Task`` objects across every event loop via ``gc`` (so the
    per-call loop in chat_executor and all background loops are covered, without
    needing a loop reference) and prints each pending task's suspended stack --
    the frame that reveals the exact ``await`` a hang is stuck on.
    """
    import asyncio
    import gc

    try:
        objects = gc.get_objects()
    except Exception as exc:  # pragma: no cover - diagnostics only
        out.write(f"<failed to enumerate objects: {exc}>\n")
        return

    # Guard every membership test individually: scanning the whole heap can touch
    # lazy proxy objects whose introspection raises (e.g. the Anthropic vertex
    # shim imports ``google-auth`` on attribute access). A single bad object must
    # not abort the entire dump.
    tasks = []
    for obj in objects:
        try:
            if isinstance(obj, asyncio.Task):
                tasks.append(obj)
        except Exception:
            continue
    pending = []
    for task in tasks:
        try:
            if not task.done():
                pending.append(task)
        except Exception:  # pragma: no cover - diagnostics only
            continue
    out.write(f"{len(pending)} pending / {len(tasks)} total asyncio task(s)\n")
    for index, task in enumerate(pending):
        try:
            out.write(f"\n--- asyncio task #{index}: {task!r} ---\n")
        except Exception:  # pragma: no cover - diagnostics only
            out.write(f"\n--- asyncio task #{index}: <repr failed> ---\n")
        try:
            task.print_stack(file=out)
        except Exception as exc:  # pragma: no cover - diagnostics only
            out.write(f"  <print_stack failed: {exc}>\n")


def _install_async_task_dumper() -> None:
    """Make nightly hangs reveal asyncio task stacks, not just thread stacks.

    Two entry points, both nightly-layer only and inert on normal runs:

    * a ``SIGQUIT`` handler so ``kill -QUIT <pid>`` on the runner dumps thread
      stacks plus every pending asyncio task's stack on demand; and
    * a monkeypatch of ``pytest_timeout.dump_stacks`` so the same task stacks are
      appended automatically whenever pytest-timeout fires -- both its ``signal``
      and ``thread`` methods call ``dump_stacks`` right before failing/killing.

    A thread-stack dump alone cannot tell a stalled network read apart from an
    async deadlock (an ``await`` on a future that never resolves): both park the
    loop in ``epoll_wait`` and the suspended coroutine is on no thread's stack.
    The per-task stacks are the decisive evidence.
    """
    if os.getenv("DATUS_TEST_LAYER") != "nightly":
        return

    import signal

    if hasattr(signal, "SIGQUIT"):  # POSIX only
        import faulthandler

        def _on_sigquit(signum, frame) -> None:  # noqa: ARG001 - handler signature
            out = sys.stderr
            try:
                out.write("\n==== SIGQUIT: thread stacks ====\n")
                faulthandler.dump_traceback(file=out, all_threads=True)
                out.write("\n==== SIGQUIT: asyncio tasks ====\n")
                _dump_async_tasks_to(out)
            except Exception as exc:  # pragma: no cover - diagnostics only
                out.write(f"<async task dump failed: {exc}>\n")
            finally:
                out.flush()

        signal.signal(signal.SIGQUIT, _on_sigquit)

    # Augment pytest-timeout's dump so the automatic timeout kill also prints
    # asyncio tasks. timeout_sigalrm()/timeout_timer() both resolve dump_stacks
    # as a module global at call time, so reassigning the attribute is enough.
    try:
        import io

        import pytest_timeout
    except Exception:  # pragma: no cover - plugin is always present in nightly
        return

    original_dump_stacks = getattr(pytest_timeout, "dump_stacks", None)
    if original_dump_stacks is None or getattr(original_dump_stacks, "_datus_wrapped", False):
        return

    def _dump_stacks_with_tasks(terminal) -> None:
        original_dump_stacks(terminal)
        try:
            buffer = io.StringIO()
            _dump_async_tasks_to(buffer)
            terminal.sep("~", title="asyncio tasks")
            terminal.write(buffer.getvalue())
        except Exception as exc:  # pragma: no cover - diagnostics only
            terminal.write(f"<async task dump failed: {exc}>\n")

    _dump_stacks_with_tasks._datus_wrapped = True
    pytest_timeout.dump_stacks = _dump_stacks_with_tasks


def pytest_configure(config) -> None:
    _DATUS_RERUN_REPORTS.clear()
    _install_async_task_dumper()


def pytest_runtest_logreport(report) -> None:
    if report.outcome != "rerun":
        return

    longrepr_text = getattr(report, "longreprtext", "") or str(report.longrepr or "")
    _DATUS_RERUN_REPORTS.append(
        {
            "nodeid": report.nodeid,
            "when": report.when,
            "duration": report.duration,
            "rerun": getattr(report, "rerun", "?"),
            "worker": getattr(report, "worker_id", os.getenv("PYTEST_XDIST_WORKER", "main")),
            "longrepr": _tail_lines(longrepr_text, RERUN_LOG_MAX_LINES),
            "sections": _format_report_sections(getattr(report, "sections", [])),
        }
    )


def pytest_terminal_summary(terminalreporter) -> None:
    if not _DATUS_RERUN_REPORTS:
        return

    terminalreporter.section("Datus rerun diagnostics", sep="=")
    for report in _DATUS_RERUN_REPORTS:
        terminalreporter.write_line(
            "RERUN "
            f"{report['nodeid']} "
            f"when={report['when']} "
            f"attempt={report['rerun']} "
            f"worker={report['worker']} "
            f"duration={report['duration']:.2f}s"
        )
        if report["longrepr"]:
            terminalreporter.write_line("First failure traceback summary:")
            for line in report["longrepr"]:
                terminalreporter.write_line(f"  {line}")
        if report["sections"]:
            terminalreporter.write_line("Captured output summary:")
            for line in report["sections"]:
                terminalreporter.write_line(f"  {line}")
