"""
Analyze and visualize success rates from SecVerifier output files.
Creates histograms showing success rates by instance type and agent.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not installed. Visualization will be skipped.")
    print("Install with: pip install matplotlib")


def parse_jsonl(file_path):
    """Parse a JSONL file and return a list of records."""
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def extract_instance_type(instance_id):
    """Extract the instance type (e.g., 'gpac', 'unicorn') from instance_id."""
    # Instance IDs are like "gpac.cve-2020-22674" or "unicorn.cve-2022-29694"
    if '.' in instance_id:
        return instance_id.split('.')[0]
    return instance_id


def analyze_results(records):
    """
    Analyze success rates by instance type and agent.
    
    Returns:
        dict: {
            'instance_type': {
                'builder': {'success': int, 'total': int},
                'exploiter': {'success': int, 'total': int},
                'fixer': {'success': int, 'total': int},
                'overall': {'success': int, 'total': int}
            }
        }
    """
    stats = defaultdict(lambda: {
        'builder': {'success': 0, 'total': 0},
        'exploiter': {'success': 0, 'total': 0},
        'fixer': {'success': 0, 'total': 0},
        'overall': {'success': 0, 'total': 0}
    })
    
    for record in records:
        instance_id = record.get('instance_id', '')
        instance_type = extract_instance_type(instance_id)
        test_result = record.get('test_result', {})
        execution = test_result.get('execution', {})
        
        # Builder stats
        builder = execution.get('builder', {})
        if builder:
            stats[instance_type]['builder']['total'] += 1
            if builder.get('success'):
                stats[instance_type]['builder']['success'] += 1
        
        # Exploiter stats (only count if builder succeeded)
        exploiter = execution.get('exploiter', {})
        if exploiter:
            stats[instance_type]['exploiter']['total'] += 1
            if exploiter.get('success'):
                stats[instance_type]['exploiter']['success'] += 1
        
        # Fixer stats (only count if exploiter succeeded)
        fixer = execution.get('fixer', {})
        if fixer:
            stats[instance_type]['fixer']['total'] += 1
            if fixer.get('success'):
                stats[instance_type]['fixer']['success'] += 1
        
        # Overall stats - consider it a success only if all agents succeeded
        if builder and exploiter and fixer:
            stats[instance_type]['overall']['total'] += 1
            if (builder.get('success') and 
                exploiter.get('success') and 
                fixer.get('success')):
                stats[instance_type]['overall']['success'] += 1
    
    return stats


def calculate_success_rates(stats):
    """Calculate success rates as percentages."""
    rates = {}
    for instance_type, agents in stats.items():
        rates[instance_type] = {}
        for agent_name, counts in agents.items():
            total = counts['total']
            success = counts['success']
            rate = (success / total * 100) if total > 0 else 0
            rates[instance_type][agent_name] = {
                'rate': rate,
                'success': success,
                'total': total
            }
    return rates


def plot_success_rates(rates_dict, title="Success Rates by Instance Type"):
    """
    Create a grouped bar chart showing success rates.
    
    Args:
        rates_dict: dict with format {model_name: {instance_type: {agent: {rate, success, total}}}}
        title: Plot title
    """
    # Prepare data
    instance_types = set()
    for model_rates in rates_dict.values():
        instance_types.update(model_rates.keys())
    instance_types = sorted(instance_types)
    
    agents = ['builder', 'exploiter', 'fixer', 'overall']
    models = list(rates_dict.keys())
    
    # Set up the plot
    fig, axes = plt.subplots(len(agents), 1, figsize=(14, 4 * len(agents)))
    if len(agents) == 1:
        axes = [axes]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    for idx, agent in enumerate(agents):
        ax = axes[idx]
        
        # Width of bars and positions
        bar_width = 0.8 / len(models)
        x = np.arange(len(instance_types))
        
        # Plot bars for each model
        for model_idx, model in enumerate(models):
            model_rates = rates_dict[model]
            rates = []
            labels = []
            
            for inst_type in instance_types:
                if inst_type in model_rates and agent in model_rates[inst_type]:
                    rate_info = model_rates[inst_type][agent]
                    rates.append(rate_info['rate'])
                    labels.append(f"{rate_info['success']}/{rate_info['total']}")
                else:
                    rates.append(0)
                    labels.append("0/0")
            
            offset = (model_idx - len(models)/2 + 0.5) * bar_width
            bars = ax.bar(x + offset, rates, bar_width, 
                         label=model, color=colors[model_idx], alpha=0.8)
            
            # Add value labels on bars
            for bar, label in zip(bars, labels):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{height:.1f}%\n({label})',
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Instance Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{agent.capitalize()} Success Rates', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(instance_types, rotation=45, ha='right')
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(loc='upper right')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig


def print_summary_table(rates_dict):
    """Print a summary table of success rates."""
    for model, model_rates in rates_dict.items():
        print(f"\n{'=' * 80}")
        print(f"Model: {model}")
        print(f"{'=' * 80}")
        
        # Calculate totals across all instance types
        totals = defaultdict(lambda: {'success': 0, 'total': 0})
        for inst_type, agents in model_rates.items():
            for agent, stats in agents.items():
                totals[agent]['success'] += stats['success']
                totals[agent]['total'] += stats['total']
        
        # Print per-instance breakdown
        print(f"\n{'Instance Type':<20} {'Agent':<12} {'Success Rate':<15} {'Count'}")
        print("-" * 70)
        
        for inst_type in sorted(model_rates.keys()):
            agents = model_rates[inst_type]
            for agent in ['builder', 'exploiter', 'fixer', 'overall']:
                if agent in agents:
                    stats = agents[agent]
                    rate = stats['rate']
                    success = stats['success']
                    total = stats['total']
                    print(f"{inst_type:<20} {agent:<12} {rate:>6.1f}%{'':<8} {success}/{total}")
                inst_type = ""  # Only print instance type once
        
        # Print totals
        print("-" * 70)
        for agent in ['builder', 'exploiter', 'fixer', 'overall']:
            if agent in totals:
                success = totals[agent]['success']
                total = totals[agent]['total']
                rate = (success / total * 100) if total > 0 else 0
                print(f"{'TOTAL':<20} {agent:<12} {rate:>6.1f}%{'':<8} {success}/{total}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_success_rates.py <output.jsonl> [output2.jsonl ...]")
        print("\nExample:")
        print("  python analyze_success_rates.py \\")
        print("    output/SEC-bench/Seed/MultiAgent/gpt-4o_maxiter_300_N_condenser=recent/output.jsonl \\")
        print("    output/SEC-bench/Seed/MultiAgent/gpt-5_maxiter_300_N_condenser=recent/output.jsonl")
        sys.exit(1)
    
    rates_dict = {}
    
    for jsonl_path in sys.argv[1:]:
        jsonl_file = Path(jsonl_path)
        if not jsonl_file.exists():
            print(f"Error: File not found: {jsonl_path}")
            continue
        
        # Extract model name from path
        # Path format: .../gpt-4o_maxiter_300_N_condenser=recent/output.jsonl
        model_name = jsonl_file.parent.name.split('_')[0]
        
        print(f"\nAnalyzing {jsonl_path}...")
        records = parse_jsonl(jsonl_file)
        print(f"  Loaded {len(records)} records")
        
        stats = analyze_results(records)
        rates = calculate_success_rates(stats)
        rates_dict[model_name] = rates
    
    # Print summary tables
    print_summary_table(rates_dict)
    
    # Create visualization (if matplotlib is available)
    if rates_dict and HAS_MATPLOTLIB:
        fig = plot_success_rates(rates_dict, 
                                 title="SecVerifier: Success Rates by Instance Type and Agent")
        
        # Save the plot
        output_file = Path("success_rates_histogram.png")
        fig.savefig(str(output_file), dpi=300, bbox_inches='tight')
        print(f"\n✓ Histogram saved to: {output_file}")
        
        # Also save as PDF
        output_pdf = Path("success_rates_histogram.pdf")
        fig.savefig(str(output_pdf), bbox_inches='tight')
        print(f"✓ PDF saved to: {output_pdf}")
        
        plt.show()
    elif rates_dict and not HAS_MATPLOTLIB:
        print("\n⚠ Skipping visualization (matplotlib not installed)")


if __name__ == "__main__":
    main()
