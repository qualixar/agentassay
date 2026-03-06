# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""AgentAssay Quickstart — Simplest Possible Example.

This demonstrates the minimal code required to test an AI agent with
AgentAssay. No external frameworks required — just a Python callable.

This example shows:
1. How to wrap any function with CustomAdapter
2. How to run stochastic trials with TrialRunner
3. How to get a statistical pass/fail verdict

Run this example:
    python examples/basic/quickstart.py
"""

from agentassay.core.models import AssayConfig, ExecutionTrace, StepTrace, TestScenario
from agentassay.core.trial_runner import TrialRunner
from agentassay.integrations.custom_adapter import CustomAdapter
from rich.console import Console

console = Console()


def my_simple_agent(input_data: dict) -> dict:
    """A toy agent that returns a response based on input.

    In a real system, this would call an LLM, invoke tools, etc.
    For this example, we simulate an agent that sometimes "fails"
    (returns None) to demonstrate stochastic testing.
    """
    query = input_data.get("query", "")

    # Simulate stochastic behavior: occasionally fail
    import random
    if random.random() < 0.1:  # 10% failure rate
        return {"output": None, "success": False, "error": "Random failure"}

    # Otherwise, return a simple response
    return {
        "output": f"Response to: {query}",
        "success": True,
        "steps": [
            {"action": "llm_response", "llm_output": f"Thinking about: {query}"},
            {"action": "tool_call", "tool_name": "search", "tool_output": "Results found"},
        ]
    }


def main():
    console.print("[bold cyan]AgentAssay Quickstart Example[/bold cyan]\n")

    # Step 1: Wrap your agent with CustomAdapter
    console.print("[yellow]Step 1:[/yellow] Create adapter for your agent")
    adapter = CustomAdapter(
        my_simple_agent,
        framework="custom",
        model="gpt-4o",
        agent_name="quickstart-agent"
    )
    console.print("✓ Adapter created\n")

    # Step 2: Define a test scenario
    console.print("[yellow]Step 2:[/yellow] Define test scenario")
    scenario = TestScenario(
        scenario_id="scenario-1",
        name="Basic Query Test",
        description="Test if agent responds to a query",
        input_data={"query": "What is the capital of France?"},
        expected_properties={"max_steps": 10}
    )
    console.print(f"✓ Scenario: {scenario.name}\n")

    # Step 3: Configure the assay (20 trials for quick demo)
    console.print("[yellow]Step 3:[/yellow] Configure statistical testing")
    assay_config = AssayConfig(
        num_trials=20,              # Run 20 stochastic trials
        significance_level=0.05,     # 95% confidence
        power=0.80,                  # 80% power to detect failures
        use_sprt=True,               # Enable early stopping (saves cost!)
    )
    console.print(f"✓ Config: {assay_config.num_trials} trials, α={assay_config.significance_level}\n")

    # Step 4: Run the trials
    console.print("[yellow]Step 4:[/yellow] Run stochastic trials")
    runner = TrialRunner(
        agent_callable=adapter.to_callable(),
        assay_config=assay_config,
        agent_config=adapter.get_config()
    )

    with console.status("[bold green]Running trials..."):
        verdict = runner.run_trials([scenario])

    console.print("✓ Trials complete\n")

    # Step 5: Check the verdict
    console.print("[yellow]Step 5:[/yellow] Statistical verdict")

    if scenario.scenario_id in verdict:
        result = verdict[scenario.scenario_id]

        console.print(f"[bold]Pass Rate:[/bold] {result.pass_rate:.1%}")
        console.print(f"[bold]Confidence Interval:[/bold] [{result.ci_lower:.1%}, {result.ci_upper:.1%}]")
        console.print(f"[bold]Trials Executed:[/bold] {result.num_trials}")

        if result.decision == "PASS":
            console.print("\n[bold green]✓ PASS[/bold green] — Agent meets quality threshold")
        elif result.decision == "FAIL":
            console.print("\n[bold red]✗ FAIL[/bold red] — Agent quality below threshold")
        else:
            console.print(f"\n[bold yellow]? {result.decision}[/bold yellow] — Inconclusive, need more trials")

    console.print("\n[dim]See examples/basic/token_efficient.py for cost-saving techniques.[/dim]")


if __name__ == "__main__":
    main()
