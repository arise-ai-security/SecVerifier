# Fix for "No such file or directory" Issue

## Problem
The agents (BuilderAgent, ExploiterAgent, FixerAgent) are failing because the repository is not set up in the Docker container at `/src/gpac`.

## Root Cause
The system uses a generic runtime image (`ghcr.io/all-hands-ai/runtime:0.52-nikolaik`) that doesn't have the project repository. The dataset provides a `dockerfile` that should set up the environment, but it's not being used.

## Solution Options

### Option 1: Initialize Repository at Runtime (Quick Fix)

Add a repository initialization function after runtime.connect() in multi-agent.py:

**Add this function** around line 1000 (before `run_controller`):

```python
async def initialize_repository(runtime: Runtime, instance: pd.Series) -> bool:
    """Initialize the repository in the runtime container."""
    logger.info('Initializing repository in container...')
    
    repo = instance.get('repo', '')
    base_commit = instance.get('base_commit', '')
    work_dir = instance.get('work_dir', '/src')
    build_sh = instance.get('build_sh', '')
    
    if not repo or not base_commit:
        logger.warning('Missing repo or base_commit, skipping repository initialization')
        return False
    
    # Create parent directory
    parent_dir = os.path.dirname(work_dir)
    if parent_dir:
        action = CmdRunAction(command=f'mkdir -p {parent_dir}')
        obs = await runtime.run_action(action)
        if obs.exit_code != 0:
            logger.error(f'Failed to create parent directory {parent_dir}')
            return False
    
    # Clone repository
    project_name = repo.split('/')[-1]
    action = CmdRunAction(command=f'cd {parent_dir} && git clone https://github.com/{repo} {project_name}', timeout=300)
    obs = await runtime.run_action(action)
    if obs.exit_code != 0:
        logger.error(f'Failed to clone repository: {repo}')
        return False
    
    # Checkout specific commit
    action = CmdRunAction(command=f'cd {work_dir} && git checkout {base_commit}')
    obs = await runtime.run_action(action)
    if obs.exit_code != 0:
        logger.error(f'Failed to checkout commit: {base_commit}')
        return False
    
    # Write build.sh script
    if build_sh:
        build_sh_path = f'{parent_dir}/build.sh'
        # Escape quotes and newlines for shell
        escaped_build_sh = build_sh.replace('\\', '\\\\').replace('"', '\\"').replace('$', '\\$')
        action = CmdRunAction(command=f'cat > {build_sh_path} << \'EOF\'\n{build_sh}\nEOF')
        obs = await runtime.run_action(action)
        if obs.exit_code == 0:
            # Make it executable
            action = CmdRunAction(command=f'chmod +x {build_sh_path}')
            await runtime.run_action(action)
        else:
            logger.warning(f'Failed to write build.sh to {build_sh_path}')
    
    logger.info(f'Repository initialized successfully at {work_dir}')
    return True
```

**Then modify the call site** (around line 915):

```python
call_async_from_sync(runtime.connect)

# ADD THIS LINE:
await initialize_repository(runtime, instance)

try:
    fake_user_resp_fn = (
        cast(FakeUserResponseFunc, auto_continue_response) if headless else None
    )
    ...
```

### Option 2: Build Custom Docker Images (Better Long-term)

1. **Create a script to build Docker images** from the dataset dockerfiles:

```bash
#!/bin/bash
# build_dataset_images.sh

DATASET_FILE="data/cve-phase-1.jsonl"

while IFS= read -r line; do
    instance_id=$(echo "$line" | jq -r '.instance_id')
    dockerfile=$(echo "$line" | jq -r '.dockerfile')
    
    if [ "$instance_id" != "null" ] && [ "$dockerfile" != "null" ]; then
        echo "Building image for $instance_id..."
        
        # Create temp directory
        temp_dir=$(mktemp -d)
        echo "$dockerfile" > "$temp_dir/Dockerfile"
        
        # Build image
        docker build -t "secverifier:$instance_id" "$temp_dir"
        
        # Cleanup
        rm -rf "$temp_dir"
    fi
done < "$DATASET_FILE"
```

2. **Modify multi-agent.py** to use the custom image:

```python
# Around line 796, replace:
runtime_container_image = os.environ.get(
    'SANDBOX_RUNTIME_CONTAINER_IMAGE',
    'ghcr.io/all-hands-ai/runtime:0.52-nikolaik',
)

# With:
instance_id = instance.get('instance_id', '')
runtime_container_image = os.environ.get(
    'SANDBOX_RUNTIME_CONTAINER_IMAGE',
    f'secverifier:{instance_id}' if instance_id else 'ghcr.io/all-hands-ai/runtime:0.52-nikolaik',
)
```

## Recommended Approach

**Start with Option 1** (runtime initialization) as it's quicker and doesn't require pre-building images. This will:
- Clone the repository into the container at startup
- Checkout the correct commit
- Set up the build.sh script
- Be ready for all agents to use

**Move to Option 2** later for better performance, as it:
- Pre-builds images with all dependencies
- Faster startup times
- More reliable environment setup

## Testing the Fix

After implementing Option 1, test with:

```bash
poetry run python multi-agent.py --llm-config llm.4o --iterations 10 --headless --condenser recent --dataset-name SEC-bench/Seed --label cve --num-workers 1 --instance-id gpac.cve-2023-5586
```

The error should be resolved, and you should see:
- Repository cloned successfully
- BuilderAgent able to access `/src/gpac`
- Build commands executing properly
