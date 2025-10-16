import os
import sys
from gitingest import ingest
from pathlib import Path
import subprocess

# Global configuration: Directory path to print, and the output filename
DIR_PATH = os.getenv("DIR_PATH", Path(__file__).parent.as_posix()) # Please modify to the target directory
OUTPUT_FILE = 'output.txt'

def load_gitignore_patterns(root_dir):
    """
    Load exclusion patterns from .gitignore, ignoring comments, blanks, and negations.
    """
    gitignore_path = os.path.join(root_dir, '.gitignore')
    patterns = []
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('!'):
                    continue
                patterns.append(line)
    else:
        print(f"No .gitignore found in {root_dir}")
    return patterns

print(load_gitignore_patterns(DIR_PATH))
exclude_patterns = load_gitignore_patterns(DIR_PATH)

if len(sys.argv) < 2:
    print("Usage: python script.py <rg command and arguments>")
    sys.exit(1)

rg_command = sys.argv[1:]
include_patterns = []

try:
    output = subprocess.check_output(rg_command, cwd=DIR_PATH)
    include_patterns = set([os.path.join(DIR_PATH, p.split(':', 1)[0]) for p in output.decode('utf-8').splitlines() if p])
    print(include_patterns)
except Exception as e:
    print(f"Error running rg: {e}")
    include_patterns = []

# Ingest with filtering
summary, tree, content = ingest(
    DIR_PATH,
    include_patterns=include_patterns,
    exclude_patterns=exclude_patterns
)

print(f"\n=== Summary ===\n{summary}")
print(f"\n=== Tree ===\n{tree}")
print(f"\n=== Content Length: {len(content)} characters ===")

# Optional: Save to file for review
with open(OUTPUT_FILE, 'w') as f:
    f.write(f"## Summary\n{summary}\n\n")
    f.write(f"## Tree Structure\n```\n{tree}\n```\n\n")
    f.write(f"## Content\n{content}\n")

print(f"Context saved to: {OUTPUT_FILE}")