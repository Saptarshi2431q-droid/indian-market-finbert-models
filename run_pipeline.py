import os
import subprocess
import sys
import time

# Ensure UTF-8 execution environment for Windows
sys.stdout.reconfigure(encoding='utf-8')


def run_step(step_name, script_name):
  print(f'\n==================================================')
  print(f'🚀 RUNNING STEP: {step_name} ({script_name})')
  print(f'==================================================\n')

  start_time = time.time()
  result = subprocess.run([sys.executable, script_name], check=False)
  elapsed = round(time.time() - start_time, 2)

  if result.returncode == 0:
    print(f'\n SUCCESS: {step_name} completed in {elapsed}s.')
  else:
    print(f'\n❌ ERROR: {step_name} failed. Halting pipeline.')
    sys.exit(1)


if __name__ == '__main__':
  print('\n=== AI4INVEST AUTOMATED RAG PIPELINE EXECUTION ===')

  # Step 1: Load CSV Data
  run_step('CSV Ingestion & Dumping', '01_load_csv_data.py')

  # Step 2: FinBERT Sentence Chunking
  run_step('FinBERT Sentence Filtering', '02_finbert_sentence_chunker.py')

  # Step 3: Deduplication
  run_step('Set Deduplication & Assembly', '03_deduplicate_and_assemble.py')

  # Step 4: Final Synthesis
  run_step('Groq 70B Market Call Generation', '04_generate_real_nifty_call.py')

  print('\n==================================================')
  print('🎉 PIPELINE COMPLETE! Report saved to REAL_NIFTY_MARKET_CALL.md')
  print('==================================================\n')