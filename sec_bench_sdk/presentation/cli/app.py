"""Modern CLI for the SEC-Bench SDK multi-agent system."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import click

from sec_bench_sdk.application.dto.phase_data import ReproducerInput
from sec_bench_sdk.application.services.phase_executor import PhaseExecutor
from sec_bench_sdk.application.services.reproducer_orchestrator import ReproducerOrchestrator
from sec_bench_sdk.infrastructure.sdk.agent_builder import AgentBuilder
from sec_bench_sdk.infrastructure.sdk.conversation_runner import ConversationRunner
from sec_bench_sdk.infrastructure.sdk.llm_factory import LLMConfig


@click.group()
@click.version_option(version="2.0.0")
def cli():
    """SEC-Bench SDK - Multi-agent security vulnerability reproduction system."""
    pass


@cli.command()
@click.option(
    '--instance-id',
    required=True,
    help='Instance ID to process (e.g., gpac.cve-2022-3178)',
)
@click.option(
    '--model',
    default='openai/gpt-4o',
    help='LLM model to use (e.g., openai/gpt-5, anthropic/claude-sonnet-4-5)',
)
@click.option(
    '--workspace',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=Path.cwd(),
    help='Workspace directory for execution',
)
@click.option(
    '--output-dir',
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path.cwd() / 'output',
    help='Output directory for results',
)
@click.option(
    '--repository-url',
    help='Repository URL (if not in instance metadata)',
)
@click.option(
    '--base-commit',
    help='Base commit hash (if not in instance metadata)',
)
@click.option(
    '--vulnerability-desc',
    help='Vulnerability description (if not in instance metadata)',
)
@click.option(
    '--max-retries',
    default=1,
    type=int,
    help='Maximum retries per phase',
)
@click.option(
    '--temperature',
    default=0.0,
    type=float,
    help='LLM temperature',
)
@click.option(
    '--max-output-tokens',
    default=8192,
    type=int,
    help='Maximum output tokens to generate',
)
@click.option(
    '--prompt-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=Path(__file__).parent.parent.parent.parent / 'config' / 'prompts',
    help='Directory containing prompt templates',
)
def run(
    instance_id: str,
    model: str,
    workspace: Path,
    output_dir: Path,
    repository_url: Optional[str],
    base_commit: Optional[str],
    vulnerability_desc: Optional[str],
    max_retries: int,
    temperature: float,
    max_output_tokens: int,
    prompt_dir: Path,
):
    """Run the multi-agent vulnerability reproduction workflow."""
    click.echo(f"🚀 Starting SEC-Bench SDK for instance: {instance_id}")
    click.echo(f"📦 Model: {model}")
    click.echo(f"📁 Workspace: {workspace}")
    click.echo(f"📤 Output: {output_dir}")

    # Validate required information
    if not repository_url:
        click.echo("❌ Error: --repository-url is required", err=True)
        sys.exit(1)
    if not base_commit:
        click.echo("❌ Error: --base-commit is required", err=True)
        sys.exit(1)
    if not vulnerability_desc:
        click.echo("❌ Error: --vulnerability-desc is required", err=True)
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure LLM
    llm_config = LLMConfig(
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    # Create services
    agent_builder = AgentBuilder(prompt_dir=prompt_dir)
    conversation_runner = ConversationRunner()
    phase_executor = PhaseExecutor(
        agent_builder=agent_builder,
        conversation_runner=conversation_runner,
    )
    orchestrator = ReproducerOrchestrator(
        phase_executor=phase_executor,
        llm_config=llm_config,
        max_retries=max_retries,
    )

    # Create input
    reproducer_input = ReproducerInput(
        instance_id=instance_id,
        repository_url=repository_url,
        base_commit=base_commit,
        vulnerability_description=vulnerability_desc,
        workspace=workspace,
        output_dir=output_dir,
        metadata={
            'instance_id': instance_id,
            'repository_url': repository_url,
            'base_commit': base_commit,
            'vulnerability_description': vulnerability_desc,
        },
    )

    # Execute workflow
    try:
        click.echo("\n🔧 Phase 1: Builder")
        result = asyncio.run(orchestrator.execute(reproducer_input))

        if result.success:
            click.echo("\n✅ All phases completed successfully!")

            # Save output
            output_file = output_dir / f"{instance_id}_output.json"
            output_data = {
                'instance_id': instance_id,
                'success': result.success,
                'builder': _phase_output_to_dict(result.builder_output),
                'exploiter': _phase_output_to_dict(result.exploiter_output),
                'fixer': _phase_output_to_dict(result.fixer_output),
            }

            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)

            click.echo(f"📄 Output saved to: {output_file}")
        else:
            click.echo(f"\n❌ Workflow failed: {result.error_message}", err=True)
            sys.exit(1)

    except Exception as e:
        click.echo(f"\n❌ Error: {str(e)}", err=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)


def _phase_output_to_dict(output):
    """Convert phase output to dictionary."""
    if output is None:
        return None
    return {
        'success': output.success,
        'final_thought': output.final_thought,
        'outputs': output.outputs,
        'artifacts': output.artifacts,
        'error_message': output.error_message,
        'retry_count': output.retry_count,
    }


@cli.command()
@click.option(
    '--dataset',
    default='SEC-bench/Seed',
    help='HuggingFace dataset name',
)
@click.option(
    '--split',
    default='test',
    help='Dataset split',
)
@click.option(
    '--limit',
    type=int,
    help='Limit number of instances to process',
)
def batch(dataset: str, split: str, limit: Optional[int]):
    """Run batch processing on a dataset."""
    click.echo(f"📦 Loading dataset: {dataset} ({split})")
    click.echo("🚧 Batch processing not yet implemented")
    # TODO: Implement batch processing


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()
