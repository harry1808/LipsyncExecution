"""
Verify that all evaluation files were created successfully
"""

import os
from pathlib import Path

def verify_files():
    print("\n" + "="*70)
    print("EVALUATION SYSTEM - INSTALLATION VERIFICATION".center(70))
    print("="*70 + "\n")
    
    files_to_check = [
        ('webapp/evaluation_metrics.py', 'Core metrics calculation module'),
        ('webapp/evaluate_dubbing.py', 'Evaluation pipeline'),
        ('webapp/evaluation_visualizer.py', 'Report generation'),
        ('example_evaluation.py', 'Usage examples'),
        ('run_evaluation.py', 'Command-line runner'),
        ('test_data_template.json', 'Test data template'),
        ('EVALUATION_README.md', 'Complete documentation'),
        ('WHAT_TO_SHOW.md', 'Presentation guide'),
        ('QUICK_START_EVALUATION.md', '5-minute quick start'),
        ('EVALUATION_COMPLETE_SUMMARY.md', 'Summary document'),
    ]
    
    print("Checking installed files:\n")
    
    all_exist = True
    total_size = 0
    
    for filepath, description in files_to_check:
        exists = os.path.exists(filepath)
        size = os.path.getsize(filepath) if exists else 0
        total_size += size
        
        status = "[OK]" if exists else "[MISSING]"
        size_str = f"{size:,} bytes" if exists else "NOT FOUND"
        
        print(f"  {status} {filepath}")
        print(f"       {description} - {size_str}")
        
        if not exists:
            all_exist = False
    
    print("\n" + "-"*70)
    print(f"\nTotal files: {len(files_to_check)}")
    print(f"Total size: {total_size:,} bytes ({total_size/1024:.1f} KB)")
    
    if all_exist:
        print("\n[SUCCESS] All evaluation files installed correctly!")
        print("\n" + "="*70)
        print("NEXT STEPS".center(70))
        print("="*70)
        print("""
1. Read the Quick Start:
   See QUICK_START_EVALUATION.md

2. Try a quick test:
   python run_evaluation.py --quick ^
     --video "your_video.mp4" ^
     --source-lang en ^
     --dest-lang hi ^
     --transcript "Hello" ^
     --translation "Translation" ^
     --html

3. View example code:
   python example_evaluation.py

4. Read full documentation:
   See EVALUATION_README.md
        """)
        print("="*70 + "\n")
    else:
        print("\n[WARNING] Some files are missing!")
        print("   Please check the file creation process.\n")
    
    return all_exist


if __name__ == "__main__":
    verify_files()

