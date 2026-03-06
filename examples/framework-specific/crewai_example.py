# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""CrewAI Integration Example.

Shows how to test a CrewAI crew with AgentAssay. Each task in the crew
becomes a separate StepTrace.

Prerequisites:
    pip install agentassay[crewai]

Run this example:
    python examples/framework-specific/crewai_example.py
"""

from agentassay.core.models import AssayConfig, TestScenario
from agentassay.core.trial_runner import TrialRunner
from agentassay.integrations.crewai_adapter import CrewAIAdapter
from rich.console import Console

console = Console()


def create_mock_crew():
    """Create a mock CrewAI crew for demonstration.

    In a real application, replace with your actual crew:

    >>> from crewai import Agent, Task, Crew
    >>> researcher = Agent(role="Researcher", goal="Find information")
    >>> writer = Agent(role="Writer", goal="Write report")
    >>> research_task = Task(description="Research topic", agent=researcher)
    >>> write_task = Task(description="Write report", agent=writer)
    >>> crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task])
    """
    try:
        from crewai import Agent, Task, Crew

        # Create agents
        agent1 = Agent(
            role="Researcher",
            goal="Research the topic",
            backstory="Expert researcher",
            allow_delegation=False
        )

        # Create tasks
        task1 = Task(
            description="Research: {topic}",
            agent=agent1,
            expected_output="Research findings"
        )

        # Create crew
        crew = Crew(agents=[agent1], tasks=[task1])
        return crew

    except ImportError:
        console.print("[yellow]CrewAI not installed. Using mock crew.[/yellow]")
        console.print("[dim]Install with: pip install agentassay[crewai][/dim]\n")

        class MockCrew:
            def kickoff(self, inputs):
                topic = inputs.get("topic", "unknown")

                class MockOutput:
                    tasks_output = [
                        type('obj', (object,), {
                            'agent': 'researcher',
                            'raw': f"Research findings for {topic}",
                            'description': f"Research: {topic}"
                        })
                    ]
                    raw = f"Completed research on {topic}"

                return MockOutput()

        return MockCrew()


def main():
    console.print("[bold cyan]CrewAI + AgentAssay Integration[/bold cyan]\n")

    # Step 1: Create your crew
    console.print("[yellow]Step 1:[/yellow] Create CrewAI crew")
    crew = create_mock_crew()
    console.print("✓ Crew created\n")

    # Step 2: Wrap with CrewAIAdapter
    console.print("[yellow]Step 2:[/yellow] Wrap crew with adapter")
    adapter = CrewAIAdapter(
        crew=crew,
        model="gpt-4o",
        agent_name="my-crew"
    )
    console.print("✓ Adapter created\n")

    # Step 3: Define test scenario
    console.print("[yellow]Step 3:[/yellow] Define test scenario")
    scenario = TestScenario(
        scenario_id="crew-test",
        name="Research Task Test",
        input_data={"topic": "AI agent testing"},
        expected_properties={"max_steps": 3}
    )
    console.print(f"✓ Scenario: {scenario.name}\n")

    # Step 4: Run trials
    console.print("[yellow]Step 4:[/yellow] Run stochastic trials")
    assay_config = AssayConfig(num_trials=15, use_sprt=True)
    runner = TrialRunner(
        agent_callable=adapter.to_callable(),
        assay_config=assay_config,
        agent_config=adapter.get_config()
    )

    with console.status("[bold green]Running trials..."):
        verdict = runner.run_trials([scenario])

    console.print("✓ Trials complete\n")

    # Step 5: Check verdict
    console.print("[yellow]Step 5:[/yellow] Results")
    if scenario.scenario_id in verdict:
        result = verdict[scenario.scenario_id]
        console.print(f"[bold]Pass Rate:[/bold] {result.pass_rate:.1%}")
        console.print(f"[bold]Decision:[/bold] {result.decision}")

    console.print("\n[dim]Each CrewAI task becomes a StepTrace with agent attribution.[/dim]")


if __name__ == "__main__":
    main()
