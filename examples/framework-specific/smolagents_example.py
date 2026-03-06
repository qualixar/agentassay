# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""HuggingFace smolagents Integration Example.

Prerequisites: pip install agentassay[smolagents]
Run: python examples/framework-specific/smolagents_example.py
"""

from agentassay.core.models import AssayConfig, TestScenario
from agentassay.core.trial_runner import TrialRunner
from agentassay.integrations.smolagents_adapter import SmolAgentsAdapter
from rich.console import Console

console = Console()


def create_mock_agent():
    """Mock smolagents agent. Replace with your actual agent:
    >>> from smolagents import CodeAgent, HfApiModel
    >>> model = HfApiModel(model_id="Qwen/Qwen2.5-72B")
    >>> agent = CodeAgent(tools=[], model=model)
    """
    try:
        from smolagents import CodeAgent
        return CodeAgent(tools=[], model=None)
    except ImportError:
        console.print("[yellow]smolagents not installed.[/yellow]")
        console.print("[dim]Install: pip install agentassay[smolagents][/dim]\n")

        class MockAgent:
            model = type('obj', (object,), {'model_id': 'Qwen/Qwen2.5-72B'})()
            logs = []
        return MockAgent()


def main():
    console.print("[bold cyan]smolagents + AgentAssay[/bold cyan]\n")

    agent = create_mock_agent()
    adapter = SmolAgentsAdapter(agent=agent, model="Qwen/Qwen2.5-72B")

    scenario = TestScenario(
        scenario_id="smol-test",
        name="Task Execution",
        input_data={"task": "Find the population of Tokyo"},
    )

    assay_config = AssayConfig(num_trials=15, use_sprt=True)
    runner = TrialRunner(adapter.to_callable(), assay_config, adapter.get_config())

    with console.status("[green]Running..."):
        verdict = runner.run_trials([scenario])

    result = verdict.get(scenario.scenario_id)
    if result:
        console.print(f"[bold]Pass Rate:[/bold] {result.pass_rate:.1%}")
        console.print(f"[bold]Decision:[/bold] {result.decision}")


if __name__ == "__main__":
    main()
