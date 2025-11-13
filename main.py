#!/usr/bin/env python3
"""
SEC-Bench SDK - Multi-Agent Security Vulnerability Reproduction System

Usage:
    python main.py --instance-id wasm3.cve-2022-28966 --model anthropic/claude-sonnet-4-5-20250929
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import click
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set OpenHands environment variables BEFORE any imports that might use OpenHands
# This bypasses workspace root resolution when using OpenHands SDK outside its repo
os.environ.setdefault('OPENHANDS_DEFAULT_AGENT_IMAGE', 'arise-lab-security/secb.base:test')
# Set git info to bypass _default_sdk_project_root() call in build.py module initialization
os.environ.setdefault('SDK_SHA', 'unknown')
os.environ.setdefault('SDK_REF', 'main')

# Monkey-patch openhands.agent_server.docker.build to avoid SDK workspace root check
# We need to patch _default_sdk_project_root BEFORE the build module initializes
import sys
from pathlib import Path as _Path

# Create fake SDK workspace structure before any openhands imports
_fake_sdk_root = _Path('/tmp/openhands_sdk_fake_root')
_fake_sdk_root.mkdir(exist_ok=True)

# Create minimal pyproject.toml with workspace members
_pyproject = _fake_sdk_root / 'pyproject.toml'
_pyproject.write_text('''[tool.uv.workspace]
members = ["openhands/*"]
''')

# Create fake subproject directories with pyproject.toml files
for _subproject in ['openhands-sdk', 'openhands-tools', 'openhands-workspace', 'openhands-agent-server']:
    _subdir = _fake_sdk_root / _subproject
    _subdir.mkdir(exist_ok=True)
    (_subdir / 'pyproject.toml').write_text(f'[project]\nname = "{_subproject}"\n')

# Temporarily change directory to fake root so _default_sdk_project_root() finds it
_original_cwd = os.getcwd()
os.chdir(_fake_sdk_root)

from sec_bench_sdk.application.dto.phase_data import ReproducerInput
from sec_bench_sdk.application.services.phase_executor import PhaseExecutor
from sec_bench_sdk.application.services.reproducer_orchestrator import ReproducerOrchestrator
from sec_bench_sdk.infrastructure.sdk.agent_builder import AgentBuilder
from sec_bench_sdk.infrastructure.sdk.conversation_runner import ConversationRunner
from sec_bench_sdk.infrastructure.sdk.llm_factory import LLMConfig

# Restore original working directory after imports
os.chdir(_original_cwd)


@click.command()
@click.option('--instance-id', help='Instance ID (e.g., wasm3.cve-2022-28966). If not provided, processes all instances.')
@click.option('--model', required=True, help='Model (e.g., anthropic/claude-sonnet-4-5-20250929, openai/gpt-4o)')
@click.option('--dataset', default='SEC-bench/Seed', help='Dataset name')
@click.option('--split', default='cve', help='Dataset split')
@click.option('--workspace', type=click.Path(path_type=Path), default=Path('workspace'), help='Workspace directory')
@click.option('--output-dir', type=click.Path(path_type=Path), default=Path('output'), help='Output directory')
@click.option('--condenser', type=click.Choice(['recent', 'llm', 'none']), default='recent', help='Condenser type for context management')
@click.option('--limit', help='Limit instances to process (e.g., ":10" for first 10, "5:15" for instances 5-14, "20:" from 20 onwards)')
@click.option('--max-iterations', default=50, type=int, help='Maximum agent iterations per phase (Note: Not yet supported by SDK)')
def main(instance_id: str, model: str, dataset: str, split: str, workspace: Path, output_dir: Path, condenser: str, limit: str, max_iterations: int):
    """Run multi-agent vulnerability reproduction."""

    # Setup
    workspace.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir = Path('prompts')

    click.echo(f"🚀 SEC-Bench SDK")
    click.echo(f"📋 Instance: {instance_id or 'ALL'}")
    click.echo(f"🤖 Model: {model}")
    click.echo(f"🧠 Condenser: {condenser}")
    click.echo(f"🔄 Max iterations: {max_iterations}")

    # Load dataset
    click.echo(f"📦 Loading from {dataset}...")
    ds = load_dataset(dataset, split=split)

    # Convert to list for filtering
    instances = list(ds)
    click.echo(f"📊 Total instances in dataset: {len(instances)}")

    # Apply limit filter if provided
    if limit:
        instances = _apply_limit(instances, limit)
        click.echo(f"📊 Instances after limit filter: {len(instances)}")

    # Filter by instance_id if provided
    if instance_id:
        instances = [item for item in instances if item['instance_id'] == instance_id]
        if len(instances) == 0:
            click.echo(f"❌ Instance not found in dataset: {instance_id}", err=True)
            sys.exit(1)
        click.echo(f"📊 Processing single instance: {instance_id}")

    # Process instances
    for idx, instance_data in enumerate(instances, 1):
        click.echo(f"\n{'='*70}")
        click.echo(f"Processing instance {idx}/{len(instances)}: {instance_data['instance_id']}")
        click.echo(f"{'='*70}")
        _process_instance(instance_data, model, workspace, output_dir, condenser, max_iterations)


def _apply_limit(instances: list, limit: str) -> list:
    """Apply limit filter to instances list.

    Args:
        instances: List of instances
        limit: Limit string (e.g., ":10", "5:15", "20:")

    Returns:
        Filtered list of instances
    """
    try:
        if ':' not in limit:
            raise ValueError("Limit must contain ':' (e.g., ':10', '5:15', '20:')")

        parts = limit.split(':')
        if len(parts) != 2:
            raise ValueError("Limit must have exactly one ':' separator")

        start_str, end_str = parts
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else len(instances)

        return instances[start:end]
    except ValueError as e:
        click.echo(f"❌ Invalid limit format: {e}", err=True)
        sys.exit(1)


def _process_instance(
    instance_data: dict,
    model: str,
    workspace: Path,
    output_dir: Path,
    condenser: str,
    max_iterations: int,
):
    """Process a single instance.

    Args:
        instance_data: Instance data from dataset
        model: Model name
        workspace: Workspace directory
        output_dir: Output directory
        condenser: Condenser type
        max_iterations: Maximum iterations
    """
    instance_id = instance_data['instance_id']
    prompt_dir = Path('prompts')

    # Create services
    llm_config = LLMConfig(model=model)
    agent_builder = AgentBuilder(prompt_dir=prompt_dir, condenser_type=condenser)
    conversation_runner = ConversationRunner(max_iterations=max_iterations)
    phase_executor = PhaseExecutor(agent_builder, conversation_runner)
    orchestrator = ReproducerOrchestrator(phase_executor, llm_config, max_retries=1)

    # Create input
    reproducer_input = ReproducerInput(
        instance_id=instance_id,
        repository_url=instance_data['repo'],
        base_commit=instance_data['base_commit'],
        vulnerability_description=instance_data.get('problem_statement', ''),
        workspace=workspace,
        output_dir=output_dir,
        metadata=instance_data,
    )

    # Execute
    try:
        result = asyncio.run(orchestrator.execute(reproducer_input))

        if result.success:
            click.echo(f"\n✅ Success: {instance_id}")
            output_file = output_dir / f"{instance_id}.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'instance_id': instance_id,
                    'success': True,
                    'builder': _to_dict(result.builder_output),
                    'exploiter': _to_dict(result.exploiter_output),
                    'fixer': _to_dict(result.fixer_output),
                }, f, indent=2)
            click.echo(f"📄 Output: {output_file}")
        else:
            click.echo(f"\n❌ Failed: {instance_id} - {result.error_message}")
    except KeyboardInterrupt:
        click.echo(f"\n⚠️  Interrupted: {instance_id}")
        raise
    except Exception as e:
        click.echo(f"\n❌ Error: {instance_id} - {e}", err=True)
        import traceback
        traceback.print_exc()


def _to_dict(output):
    if not output:
        return None
    return {
        'success': output.success,
        'final_thought': output.final_thought,
        'outputs': output.outputs,
        'artifacts': output.artifacts,
        'error_message': output.error_message,
    }


if __name__ == '__main__':
    main()
