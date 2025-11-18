from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sec_bench_sdk.application.dto.phase_data import PhaseOutput, ReproducerOutput
from sec_bench_sdk.domain.value_objects import AgentType


def _safe_json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def _format_model_slug(model: str) -> str:
    slug = model.split('/', 1)[-1] if '/' in model else model
    slug = slug.replace(':', '-').replace('/', '-').replace(' ', '-').strip('-')
    parts = [part for part in slug.split('-') if part]
    if parts and parts[-1].isdigit() and len(parts[-1]) >= 8:
        parts = parts[:-1]
    return '-'.join(parts) if parts else 'model'


@dataclass
class InstanceRunContext:
    instance_id: str
    timestamp: str
    run_uuid: str
    run_dir: Path
    completions_dir: Path
    summary_path: Path
    metadata: Dict[str, Any]
    phase_logs: Dict[str, List[str]] = field(default_factory=dict)

    def record_phase_events(self, phase_type: AgentType, events: List[Dict[str, Any]]) -> Path:
        phase_name = phase_type.value
        filename = f"{phase_name}-{uuid.uuid4()}.json"
        target = self.run_dir / filename
        target.write_text(json.dumps(events, indent=2, default=_safe_json_default), encoding='utf-8')
        self.phase_logs.setdefault(phase_name, []).append(str(target))
        return target


class RunLogger:
    def __init__(
        self,
        base_output_dir: Path,
        dataset: str,
        split: str,
        model: str,
        condenser: str,
        max_iterations: int,
        agent_label: str = 'MultiAgent',
        command: Optional[str] = None,
    ) -> None:
        self.base_output_dir = base_output_dir
        self.dataset = dataset
        self.split = split
        self.model = model
        self.condenser = condenser
        self.max_iterations = max_iterations
        self.agent_label = agent_label
        self.command = command

        self.run_root = self._resolve_run_root()
        self.run_root.mkdir(parents=True, exist_ok=True)

        self.metadata_path = self.run_root / 'metadata.json'
        self.index_path = self.run_root / 'output.jsonl'

        self._write_metadata()

    def _resolve_run_root(self) -> Path:
        dataset_root = self.base_output_dir / Path(self.dataset)
        run_dir_name = f"{_format_model_slug(self.model)}_maxiter_{self.max_iterations}_N_condenser={self.condenser}"
        return dataset_root / self.agent_label / run_dir_name

    def _write_metadata(self, extra: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {
            'agent_class': self.agent_label,
            'model': self.model,
            'dataset': self.dataset,
            'split': self.split,
            'condenser': self.condenser,
            'max_iterations': self.max_iterations,
            'command': self.command,
            'log_root': str(self.run_root),
            'updated_at': datetime.utcnow().isoformat() + 'Z',
        }
        if extra:
            payload.update(extra)
        self.metadata_path.write_text(json.dumps(payload, indent=2, default=_safe_json_default), encoding='utf-8')

    def record_invocation(
        self,
        processed_instances: List[str],
        instance_option: Optional[str],
        limit: Optional[str],
    ) -> None:
        extra = {
            'processed_instances': processed_instances,
            'instance_option': instance_option,
            'limit_option': limit,
            'invocation_size': len(processed_instances),
        }
        self._write_metadata(extra)

    def start_instance_run(self, instance_data: Dict[str, Any]) -> InstanceRunContext:
        instance_id = instance_data.get('instance_id') or 'unknown-instance'
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        run_uuid = uuid.uuid4().hex

        instance_dir = self.run_root / instance_id
        run_dir = self._next_run_directory(instance_dir, timestamp)
        completions_dir = run_dir / 'completions'
        summary_path = run_dir / 'output.json'

        completions_dir.mkdir(parents=True, exist_ok=True)

        return InstanceRunContext(
            instance_id=instance_id,
            timestamp=run_dir.name,
            run_uuid=run_uuid,
            run_dir=run_dir,
            completions_dir=completions_dir,
            summary_path=summary_path,
            metadata=instance_data,
        )

    def finalize_instance_run(
        self,
        context: InstanceRunContext,
        result: ReproducerOutput,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        summary = {
            'instance_id': context.instance_id,
            'timestamp': context.timestamp,
            'run_uuid': context.run_uuid,
            'model': self.model,
            'condenser': self.condenser,
            'max_iterations': self.max_iterations,
            'dataset': self.dataset,
            'split': self.split,
            'command': self.command,
            'phase_logs': context.phase_logs,
            'completions_dir': str(context.completions_dir),
            'instance': context.metadata,
            'result': self._serialize_reproducer_output(result),
        }
        if extra:
            summary.update(extra)

        context.summary_path.write_text(json.dumps(summary, indent=2, default=_safe_json_default), encoding='utf-8')
        with self.index_path.open('a', encoding='utf-8') as fout:
            fout.write(json.dumps(summary, default=_safe_json_default) + '\n')

    def _next_run_directory(self, instance_dir: Path, timestamp: str) -> Path:
        instance_dir.mkdir(parents=True, exist_ok=True)
        candidate = instance_dir / timestamp
        suffix = 1
        while candidate.exists():
            candidate = instance_dir / f"{timestamp}_{suffix:02d}"
            suffix += 1
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    def _serialize_reproducer_output(self, result: ReproducerOutput) -> Dict[str, Any]:
        return {
            'success': result.success,
            'error_message': result.error_message,
            'builder_output': self._serialize_phase_output(result.builder_output),
            'exploiter_output': self._serialize_phase_output(result.exploiter_output),
            'fixer_output': self._serialize_phase_output(result.fixer_output),
        }

    @staticmethod
    def _serialize_phase_output(output: Optional[PhaseOutput]) -> Optional[Dict[str, Any]]:
        if output is None:
            return None
        return {
            'phase_type': output.phase_type.value,
            'success': output.success,
            'final_thought': output.final_thought,
            'outputs': output.outputs,
            'artifacts': output.artifacts,
            'error_message': output.error_message,
            'retry_count': output.retry_count,
        }