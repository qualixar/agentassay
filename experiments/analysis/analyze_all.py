#!/usr/bin/env python3
"""
AgentAssay Experiment Analysis — Comprehensive Data Extraction
Processes all E1-E7 results and generates paper-ready tables.
"""
import json
import os
import sys
import statistics
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("experiments/results/fresh-pull-mar2")
OUTPUT_DIR = Path("experiments/analysis/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS_SHORT = {
    "gpt-5.2-chat": "GPT-5.2",
    "claude-sonnet-4-6": "Sonnet 4.6",
    "Mistral-Large-3": "Mistral-L3",
    "Llama-4-Maverick-17B-128E-Instruct-FP8": "Llama-4-Mav",
    "Phi-4": "Phi-4"
}

SCENARIOS_SHORT = {
    "ecommerce": "E-com",
    "customer_support": "CustSup",
    "code_generation": "CodeGen"
}

def load_experiment(exp_dir):
    """Load all JSON files from an experiment directory."""
    results = {}
    exp_path = RESULTS_DIR / exp_dir
    if not exp_path.exists():
        return results
    for f in sorted(exp_path.glob("*.json")):
        if f.name == "summary.json":
            results["_summary"] = json.loads(f.read_text())
        else:
            results[f.stem] = json.loads(f.read_text())
    return results


# ============================================================
# E1-E6: AGGREGATE STATISTICS
# ============================================================
def analyze_e1_e6():
    """Extract aggregate statistics from E1-E6 experiments."""
    experiments = {
        "e1_verdict": "Verdict",
        "e2_coverage": "Coverage",
        "e3_mutation": "Mutation",
        "e4_sprt": "SPRT",
        "e5_contracts": "Contracts",
        "e5_metamorphic": "Metamorphic",
        "e6_cicd": "CI/CD"
    }
    
    print("\n" + "="*80)
    print("E1-E6: AGGREGATE STATISTICS")
    print("="*80)
    
    # Master table: per-model, per-scenario metrics
    all_data = defaultdict(lambda: defaultdict(list))
    
    for exp_id, exp_name in experiments.items():
        data = load_experiment(exp_id)
        if not data:
            print(f"\n  MISSING: {exp_id}")
            continue
            
        print(f"\n--- {exp_name} ({exp_id}) ---")
        
        for key, filedata in data.items():
            if key == "_summary":
                continue
            
            model = filedata.get("model", "?")
            scenario = filedata.get("scenario_id", "?")
            results = filedata.get("results", [])
            
            if not results:
                continue
            
            # Extract per-trial metrics
            tokens = [r.get("tokens", 0) for r in results if r.get("success")]
            costs = [r.get("cost_usd", 0) for r in results if r.get("success")]
            durations = [r.get("duration_ms", 0) for r in results if r.get("success")]
            steps = [r.get("step_count", 0) for r in results if r.get("success")]
            passed = sum(1 for r in results if r.get("passed"))
            total = len(results)
            success = sum(1 for r in results if r.get("success"))
            
            model_short = MODELS_SHORT.get(model, model)
            scen_short = SCENARIOS_SHORT.get(scenario, scenario)
            
            # Store for aggregate
            all_data[model_short]["tokens"].extend(tokens)
            all_data[model_short]["costs"].extend(costs)
            all_data[model_short]["durations"].extend(durations)
            all_data[model_short]["steps"].extend(steps)
            all_data[model_short]["total_trials"] += [total]
            all_data[model_short]["success_trials"] += [success]
            
            if tokens:
                print(f"  {model_short:12s} | {scen_short:8s} | "
                      f"trials={total:3d} pass={passed:3d} "
                      f"tokens={statistics.mean(tokens):7.1f}±{statistics.stdev(tokens) if len(tokens)>1 else 0:6.1f} "
                      f"cost=${sum(costs):.4f} "
                      f"dur={statistics.mean(durations):7.1f}ms "
                      f"steps={statistics.mean(steps):.1f}")
    
    # Print aggregate table
    print("\n\n" + "="*80)
    print("AGGREGATE TABLE: Per-Model Summary (across ALL E1-E6 experiments)")
    print("="*80)
    print(f"{'Model':15s} {'Trials':>8s} {'Tokens/Trial':>14s} {'Cost/Trial':>12s} {'Duration/Trial':>16s} {'Steps/Trial':>12s}")
    print("-"*80)
    
    latex_rows = []
    for model in ["GPT-5.2", "Sonnet 4.6", "Mistral-L3", "Llama-4-Mav", "Phi-4"]:
        d = all_data[model]
        if not d["tokens"]:
            continue
        n = len(d["tokens"])
        tok_mean = statistics.mean(d["tokens"])
        tok_std = statistics.stdev(d["tokens"]) if n > 1 else 0
        cost_mean = statistics.mean(d["costs"])
        dur_mean = statistics.mean(d["durations"])
        dur_std = statistics.stdev(d["durations"]) if n > 1 else 0
        step_mean = statistics.mean(d["steps"])
        total = sum(d["total_trials"])
        
        print(f"{model:15s} {total:8d} {tok_mean:8.1f}±{tok_std:5.1f} "
              f"${cost_mean:.6f} {dur_mean:8.1f}±{dur_std:5.1f}ms {step_mean:6.2f}")
        
        latex_rows.append(f"    {model} & {total} & {tok_mean:.0f} $\\pm$ {tok_std:.0f} & "
                         f"\\${cost_mean:.4f} & {dur_mean:.0f} $\\pm$ {dur_std:.0f} & {step_mean:.1f} \\\\")
    
    # Save LaTeX fragment
    latex = """\\begin{table}[t]
\\centering
\\caption{Agent behavior characterization across 5 models, 3 scenarios, and 7 experiment configurations (E1--E6). Each cell aggregates 50 trials per model-scenario-experiment combination.}
\\label{tab:agent-characterization}
\\small
\\begin{tabular}{lrrrr}
\\toprule
\\textbf{Model} & \\textbf{Trials} & \\textbf{Tokens/Trial} & \\textbf{Cost/Trial} & \\textbf{Duration (ms)} \\\\
\\midrule
""" + "\n".join(latex_rows) + """
\\bottomrule
\\end{tabular}
\\end{table}"""
    
    (OUTPUT_DIR / "table_agent_characterization.tex").write_text(latex)
    print(f"\n  → Saved to {OUTPUT_DIR}/table_agent_characterization.tex")
    
    return all_data


# ============================================================
# E7: TOKEN EFFICIENCY (THE CROWN JEWEL)
# ============================================================
def analyze_e7():
    """Analyze E7 efficiency experiment — the core innovation."""
    data = load_experiment("e7_efficiency")
    if not data:
        print("ERROR: E7 data not found!")
        return
    
    print("\n\n" + "="*80)
    print("E7: TOKEN-EFFICIENT TESTING EVALUATION")
    print("="*80)
    
    # Collect all results by approach and model
    approach_data = defaultdict(lambda: defaultdict(list))
    model_data = defaultdict(lambda: defaultdict(list))
    
    for key, filedata in data.items():
        if key == "_summary":
            continue
        
        model = filedata.get("model", "?")
        model_short = MODELS_SHORT.get(model, model)
        results = filedata.get("results", [])
        
        for r in results:
            approach = r.get("approach", "?")
            approach_data[approach]["trials_used"].append(r.get("trials_used", 0))
            approach_data[approach]["tokens_used"].append(r.get("tokens_used", 0))
            approach_data[approach]["cost_usd"].append(r.get("cost_usd", 0))
            approach_data[approach]["verdict"].append(r.get("verdict", "?"))
            approach_data[approach]["power"].append(r.get("power", 0))
            
            model_data[model_short][approach] = model_data[model_short].get(approach, [])
            model_data[model_short][approach].append(r)
    
    # Print approach comparison table
    print(f"\n{'Approach':25s} {'Trials':>10s} {'Tokens':>12s} {'Cost':>10s} {'Savings':>10s} {'Power':>8s} {'PASS%':>8s}")
    print("-"*85)
    
    baseline_cost = None
    baseline_trials = None
    baseline_tokens = None
    
    approaches_order = ["fixed_n", "sprt_only", "sprt_fingerprint", "sprt_fp_budget", "full_system"]
    approach_names = {
        "fixed_n": "Fixed-$n$ (baseline)",
        "sprt_only": "SPRT",
        "sprt_fingerprint": "SPRT + Fingerprint",
        "sprt_fp_budget": "SPRT + FP + Budget",
        "full_system": "Full system"
    }
    
    latex_e7_rows = []
    
    for approach in approaches_order:
        d = approach_data[approach]
        if not d["trials_used"]:
            continue
        
        trials_mean = statistics.mean(d["trials_used"])
        trials_std = statistics.stdev(d["trials_used"]) if len(d["trials_used"]) > 1 else 0
        tokens_mean = statistics.mean(d["tokens_used"])
        tokens_std = statistics.stdev(d["tokens_used"]) if len(d["tokens_used"]) > 1 else 0
        cost_mean = statistics.mean(d["cost_usd"])
        cost_std = statistics.stdev(d["cost_usd"]) if len(d["cost_usd"]) > 1 else 0
        power_mean = statistics.mean(d["power"])
        pass_rate = sum(1 for v in d["verdict"] if v == "PASS") / len(d["verdict"]) * 100
        
        if approach == "fixed_n":
            baseline_cost = cost_mean
            baseline_trials = trials_mean
            baseline_tokens = tokens_mean
            savings_str = "---"
            savings_pct = 0
        else:
            if baseline_cost and baseline_cost > 0:
                savings_pct = (1 - cost_mean / baseline_cost) * 100
                savings_str = f"{savings_pct:.1f}%"
            else:
                savings_str = "N/A"
                savings_pct = 0
        
        print(f"{approach:25s} {trials_mean:6.1f}±{trials_std:3.1f} "
              f"{tokens_mean:8.0f}±{tokens_std:5.0f} "
              f"${cost_mean:.4f}±{cost_std:.4f} "
              f"{savings_str:>10s} {power_mean:6.2f} {pass_rate:6.1f}%")
        
        name = approach_names.get(approach, approach)
        if approach == "fixed_n":
            latex_e7_rows.append(f"    {name} & {trials_mean:.0f} & \\${cost_mean:.3f} & --- & {power_mean:.2f} \\\\")
        else:
            trial_savings = (1 - trials_mean / baseline_trials) * 100 if baseline_trials else 0
            latex_e7_rows.append(f"    {name} & {trials_mean:.1f} & \\${cost_mean:.4f} & {savings_pct:.1f}\\% & {power_mean:.2f} \\\\")
    
    # Per-model breakdown
    print(f"\n\n--- Per-Model E7 Breakdown ---")
    print(f"{'Model':15s} {'Approach':25s} {'Trials':>8s} {'Cost':>10s} {'Savings':>10s}")
    print("-"*70)
    
    for model in ["GPT-5.2", "Sonnet 4.6", "Mistral-L3", "Llama-4-Mav", "Phi-4"]:
        if model not in model_data:
            continue
        for approach in approaches_order:
            results = model_data[model].get(approach, [])
            if not results:
                continue
            trials = [r.get("trials_used", 0) for r in results]
            costs = [r.get("cost_usd", 0) for r in results]
            t_mean = statistics.mean(trials)
            c_mean = statistics.mean(costs)
            
            if approach == "fixed_n":
                model_baseline = c_mean
                sav = "---"
            else:
                sav = f"{(1 - c_mean/model_baseline)*100:.1f}%" if model_baseline > 0 else "N/A"
            
            print(f"{model:15s} {approach:25s} {t_mean:6.1f} ${c_mean:.4f} {sav:>10s}")
        print()
    
    # Save LaTeX table
    latex_e7 = """\\begin{table}[t]
\\centering
\\caption{E7: Token-efficient testing. Mean cost per regression check at
  equivalent $(\\alpha = 0.05, \\beta = 0.10)$ guarantees across
  ecommerce scenario, 4 models, 25 repetitions.}
\\label{tab:e7-efficiency}
\\small
\\begin{tabular}{lrrl}
\\toprule
\\textbf{Approach} & \\textbf{Mean Trials} & \\textbf{Mean Cost} &
  \\textbf{Savings} \\\\
\\midrule
""" + "\n".join(latex_e7_rows) + """
\\bottomrule
\\end{tabular}
\\end{table}"""
    
    (OUTPUT_DIR / "table_e7_efficiency.tex").write_text(latex_e7)
    print(f"\n  → Saved to {OUTPUT_DIR}/table_e7_efficiency.tex")
    
    # Per-model E7 table
    latex_model_rows = []
    for model in ["GPT-5.2", "Sonnet 4.6", "Mistral-L3", "Llama-4-Mav"]:
        if model not in model_data:
            continue
        fixed_cost = statistics.mean([r.get("cost_usd", 0) for r in model_data[model].get("fixed_n", [])]) if model_data[model].get("fixed_n") else 0
        sprt_cost = statistics.mean([r.get("cost_usd", 0) for r in model_data[model].get("sprt_only", [])]) if model_data[model].get("sprt_only") else 0
        fp_cost = statistics.mean([r.get("cost_usd", 0) for r in model_data[model].get("sprt_fingerprint", [])]) if model_data[model].get("sprt_fingerprint") else 0
        budget_cost = statistics.mean([r.get("cost_usd", 0) for r in model_data[model].get("sprt_fp_budget", [])]) if model_data[model].get("sprt_fp_budget") else 0
        full_cost = statistics.mean([r.get("cost_usd", 0) for r in model_data[model].get("full_system", [])]) if model_data[model].get("full_system") else 0
        
        sprt_sav = f"{(1-sprt_cost/fixed_cost)*100:.0f}\\%" if fixed_cost > 0 else "---"
        full_sav = f"{(1-full_cost/fixed_cost)*100:.0f}\\%" if fixed_cost > 0 else "---"
        
        latex_model_rows.append(f"    {model} & \\${fixed_cost:.3f} & \\${sprt_cost:.4f} ({sprt_sav}) & \\${full_cost:.4f} ({full_sav}) \\\\")
    
    latex_e7_models = """\\begin{table}[t]
\\centering
\\caption{E7: Per-model cost comparison. Fixed-$n$ baseline vs.\\ SPRT vs.\\ full system.}
\\label{tab:e7-per-model}
\\small
\\begin{tabular}{lrrr}
\\toprule
\\textbf{Model} & \\textbf{Fixed-$n$} & \\textbf{SPRT (savings)} & \\textbf{Full (savings)} \\\\
\\midrule
""" + "\n".join(latex_model_rows) + """
\\bottomrule
\\end{tabular}
\\end{table}"""
    
    (OUTPUT_DIR / "table_e7_per_model.tex").write_text(latex_e7_models)
    print(f"  → Saved to {OUTPUT_DIR}/table_e7_per_model.tex")
    
    return approach_data, model_data


# ============================================================
# TOTAL COST AND TRIAL COUNTS
# ============================================================
def compute_totals():
    """Compute total experimental costs and trial counts."""
    print("\n\n" + "="*80)
    print("TOTAL EXPERIMENT STATISTICS")
    print("="*80)
    
    total_trials = 0
    total_cost = 0
    total_tokens = 0
    
    experiments = ["e1_verdict", "e2_coverage", "e3_mutation", "e4_sprt", 
                   "e5_contracts", "e5_metamorphic", "e6_cicd", "e7_efficiency"]
    
    for exp_id in experiments:
        data = load_experiment(exp_id)
        exp_trials = 0
        exp_cost = 0
        exp_tokens = 0
        
        for key, filedata in data.items():
            if key == "_summary":
                continue
            results = filedata.get("results", [])
            for r in results:
                exp_trials += 1
                exp_cost += r.get("cost_usd", 0)
                exp_tokens += r.get("tokens", r.get("tokens_used", 0))
        
        total_trials += exp_trials
        total_cost += exp_cost
        total_tokens += exp_tokens
        
        print(f"  {exp_id:20s}: {exp_trials:5d} trials, ${exp_cost:.4f}, {exp_tokens:,} tokens")
    
    print(f"\n  {'TOTAL':20s}: {total_trials:5d} trials, ${total_cost:.4f}, {total_tokens:,} tokens")
    print(f"  Models: 5 (GPT-5.2, Sonnet 4.6, Mistral-Large-3, Llama-4-Maverick, Phi-4)")
    print(f"  Scenarios: 3 (ecommerce, customer_support, code_generation)")
    
    return total_trials, total_cost, total_tokens


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("AgentAssay Experiment Analysis")
    print(f"Data directory: {RESULTS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Run all analyses
    agg = analyze_e1_e6()
    e7_approach, e7_model = analyze_e7()
    totals = compute_totals()
    
    print("\n\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
