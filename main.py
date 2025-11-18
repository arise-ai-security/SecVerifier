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

load_dotenv()

# TODO: Replace with custom image - currently using whiwonlee's secb.base as foundation
# This bypasses workspace root resolution when using OpenHands SDK outside its repo
os.environ.setdefault('OPENHANDS_DEFAULT_AGENT_IMAGE', 'arise-lab-security/secb.base:test')
os.environ.setdefault('SDK_SHA', 'unknown')
os.environ.setdefault('SDK_REF', 'main')

# Create fake SDK workspace structure to bypass OpenHands monorepo checks
# OpenHands expects to run from its own repo with specific pyproject.toml structure
import sys
from pathlib import Path as _Path
import time
_fake_sdk_root = _Path(f'/tmp/openhands_sdk_fake_root_{int(time.time())}')
_fake_sdk_root.mkdir(exist_ok=True)

_pyproject = _fake_sdk_root / 'pyproject.toml'
_pyproject.write_text('''[tool.uv.workspace]
members = ["openhands/*"]
''')

for _subproject in ['openhands-sdk', 'openhands-tools', 'openhands-workspace', 'openhands-agent-server']:
    _subdir = _fake_sdk_root / _subproject
    _subdir.mkdir(exist_ok=True)
    (_subdir / 'pyproject.toml').write_text(f'[project]\nname = "{_subproject}"\n')

_original_cwd = os.getcwd()
os.chdir(_fake_sdk_root)

from sec_bench_sdk.application.dto.phase_data import ReproducerInput
from sec_bench_sdk.application.services.phase_executor import PhaseExecutor
from sec_bench_sdk.application.services.reproducer_orchestrator import ReproducerOrchestrator
from sec_bench_sdk.infrastructure.sdk.agent_builder import AgentBuilder
from sec_bench_sdk.infrastructure.sdk.conversation_runner import ConversationRunner
from sec_bench_sdk.infrastructure.sdk.llm_factory import LLMConfig

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

    workspace.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir = Path('prompts')

    click.echo("SEC-Bench SDK")
    click.echo(f"Instance: {instance_id or 'ALL'}")
    click.echo(f"Model: {model}")
    click.echo(f"Condenser: {condenser}")
    click.echo(f"Max iterations: {max_iterations}")

    click.echo(f"Loading dataset: {dataset}...")
    ds = load_dataset(dataset, split=split)

    instances = list(ds)
    click.echo(f"Total instances in dataset: {len(instances)}")

    if limit:
        instances = _apply_limit(instances, limit)
        click.echo(f"Instances after limit filter: {len(instances)}")

    if instance_id:
        instances = [item for item in instances if item['instance_id'] == instance_id]
        if len(instances) == 0:
            click.echo(f"Instance not found in dataset: {instance_id}", err=True)
            sys.exit(1)
        click.echo(f"Processing single instance: {instance_id}")

    for idx, instance_data in enumerate(instances, 1):
        click.echo(f"\n{'='*70}")
        click.echo(f"Processing instance {idx}/{len(instances)}: {instance_data['instance_id']}")
        click.echo(f"{'='*70}")
        _process_instance(instance_data, model, workspace, output_dir, condenser, max_iterations)


def _apply_limit(instances: list, limit: str) -> list:
    """Apply limit filter to instances list using Python slice notation."""
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
        click.echo(f"Invalid limit format: {e}", err=True)
        sys.exit(1)


def _process_instance(
    instance_data: dict,
    model: str,
    workspace: Path,
    output_dir: Path,
    condenser: str,
    max_iterations: int,
):
    """Process a single instance through the multi-agent workflow."""
    instance_id = instance_data['instance_id']
    prompt_dir = Path('prompts')

    llm_config = LLMConfig(model=model)
    agent_builder = AgentBuilder(prompt_dir=prompt_dir, condenser_type=condenser)
    conversation_runner = ConversationRunner(max_iterations=max_iterations)
    phase_executor = PhaseExecutor(agent_builder, conversation_runner)
    orchestrator = ReproducerOrchestrator(phase_executor, llm_config, max_retries=1)

    reproducer_input = ReproducerInput(
        instance_id=instance_id,
        repository_url=instance_data['repo'],
        base_commit=instance_data['base_commit'],
        vulnerability_description=instance_data.get('problem_statement', ''),
        workspace=workspace,
        output_dir=output_dir,
        metadata=instance_data,
    )

    try:
        result = asyncio.run(orchestrator.execute(reproducer_input))

        if result.success:
            click.echo(f"\nSuccess: {instance_id}")
            output_file = output_dir / f"{instance_id}.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'instance_id': instance_id,
                    'success': True,
                    'builder': _to_dict(result.builder_output),
                    'exploiter': _to_dict(result.exploiter_output),
                    'fixer': _to_dict(result.fixer_output),
                }, f, indent=2)
            click.echo(f"Output saved: {output_file}")
        else:
            click.echo(f"\nFailed: {instance_id} - {result.error_message}")
    except KeyboardInterrupt:
        click.echo(f"\nInterrupted: {instance_id}")
        raise
    except Exception as e:
        click.echo(f"\nError: {instance_id} - {e}", err=True)
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
