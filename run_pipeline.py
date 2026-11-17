"""
Master Pipeline Runner for E-Commerce Cart Abandonment Analysis & Prediction
Executes Step 1 through Step 7 sequentially within a single Spark session.
"""
import os
import sys

def run_step(step_num, file_name):
    print(f"\n{'='*70}")
    print(f"RUNNING STEP {step_num}: {file_name}")
    print(f"{'='*70}\n")
    with open(file_name, "r", encoding="utf-8") as f:
        code = compile(f.read(), file_name, "exec")
        exec(code, globals())

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    
    steps = [
        (1, "step1_load_data.py"),
        (2, "step2_clean_data.py"),
        (3, "step3_features.py"),
        (4, "step4_decision_tree.py"),
        (5, "step5_evaluate.py"),
        (6, "step6_visualize.py"),
        (7, "step7_more_visuals.py"),
    ]
    
    for step_num, script in steps:
        run_step(step_num, script)
        
    print("\n" + "="*70)
    print("PIPELINE COMPLETE: All steps executed and visuals generated.")
    print("="*70)
