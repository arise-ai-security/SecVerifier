import os
from typing import Optional
from openhands.core.config import OpenHandsConfig
from openhands.runtime.plugins import PluginRequirement


def _compute_default_python_prefix() -> list[str]:
    """Determine how to launch the runtime server inside the container.

    If OPENHANDS_USE_MICROMAMBA is set to '0' or 'false' (case-insensitive),
    we will skip micromamba and poetry prefixes and run the module directly with python.
    Otherwise, default to using micromamba + poetry as before.
    """
    use_mamba = os.environ.get('OPENHANDS_USE_MICROMAMBA', '1').lower()
    if use_mamba in ('0', 'false', 'no'):  # direct python
        return []
    return [
        '/openhands/micromamba/bin/micromamba',
        'run',
        '-n',
        'openhands',
        'poetry',
        'run',
    ]


DEFAULT_PYTHON_PREFIX = _compute_default_python_prefix()
DEFAULT_MAIN_MODULE = 'openhands.runtime.action_execution_server'


def get_action_execution_server_startup_command(
    server_port: int,
    plugins: list[PluginRequirement],
    app_config: OpenHandsConfig,
    python_prefix: Optional[list[str]] = None,
    override_user_id: int | None = None,
    override_username: str | None = None,
    main_module: str = DEFAULT_MAIN_MODULE,
    python_executable: str = 'python',
) -> list[str]:
    sandbox_config = app_config.sandbox

    # Decide whether to use micromamba based on config/env if caller didn't override python_prefix
    if python_prefix is None:
        # Prefer explicit signal from runtime_startup_env_vars in app_config
        flag = None
        if sandbox_config.runtime_startup_env_vars:
            flag = sandbox_config.runtime_startup_env_vars.get('OPENHANDS_USE_MICROMAMBA')
        # Fall back to host env var
        if flag is None:
            flag = os.environ.get('OPENHANDS_USE_MICROMAMBA')

        if flag is not None and str(flag).lower() in ('0', 'false', 'no'):
            python_prefix = []  # run directly with python
        else:
            python_prefix = DEFAULT_PYTHON_PREFIX

    # Plugin args
    plugin_args = []
    if plugins is not None and len(plugins) > 0:
        plugin_args = ['--plugins'] + [plugin.name for plugin in plugins]

    # Browsergym stuffs
    browsergym_args = []
    if sandbox_config.browsergym_eval_env is not None:
        browsergym_args = [
            '--browsergym-eval-env'
        ] + sandbox_config.browsergym_eval_env.split(' ')

    username = override_username or (
        'openhands' if app_config.run_as_openhands else 'root'
    )
    user_id = override_user_id or (
        sandbox_config.user_id if app_config.run_as_openhands else 0
    )

    base_cmd = [
        *python_prefix,
        python_executable,
        '-u',
        '-m',
        main_module,
        str(server_port),
        '--working-dir',
        app_config.workspace_mount_path_in_sandbox,
        *plugin_args,
        '--username',
        username,
        '--user-id',
        str(user_id),
        '--git-user-name',
        app_config.git_user_name,
        '--git-user-email',
        app_config.git_user_email,
        *browsergym_args,
    ]

    if not app_config.enable_browser:
        base_cmd.append('--no-enable-browser')

    return base_cmd
