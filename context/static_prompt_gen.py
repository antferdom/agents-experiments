import os
import fnmatch
from pathlib import Path

# Global configuration: Directory path to print, and the output filename
DIR_PATH =  os.getenv("DIR_PATH", Path(__file__).parent.as_posix()) # Please modify to the target directory
OUTPUT_FILE = 'output.txt'

# Exclusion list: Folders/files to exclude (wildcards supported)
EXCLUDE_LIST = [
    f"{DIR_PATH}/logs",                 # Exclude the entire logs directory
    f"{DIR_PATH}/__pycache__",          # Exclude Python cache
    f"{DIR_PATH}/*/__pycache__",        # Exclude cache from all subdirectories
    "*.pyc",                            # Exclude Python compiled files
    "*.pyo",                            # Exclude Python optimized files
    "*.log",                            # Exclude log files
    "*.pt",                             # Exclude log files
    "*.pth",                            # Exclude log files
    ".git",                        
    "DS_Store",                    
    "*.tmp",                       
    ".gitignore",
]


def should_exclude(path):
    """
    Check if a path should be excluded.
    """
    # Normalize the path (using forward slashes)
    normalized_path = path.replace('\\', '/')
    
    for exclude_pattern in EXCLUDE_LIST:
        # Exact match
        if normalized_path == exclude_pattern:
            return True
        
        # Wildcard match
        if fnmatch.fnmatch(normalized_path, exclude_pattern):
            return True
        
        # Check if the path contains a directory name to be excluded
        path_parts = normalized_path.split('/')
        for part in path_parts:
            if fnmatch.fnmatch(part, exclude_pattern):
                return True
    
    return False


def get_tree(root, prefix=''):
    """
    Returns each line of the directory tree and a list of all file paths.
    Automatically skips files and directories specified in EXCLUDE_LIST.
    """
    entries = sorted(os.listdir(root))
    lines = []
    files = []
    
    # Filter out excluded files and directories
    filtered_entries = []
    for name in entries:
        path = os.path.join(root, name)
        if not should_exclude(path):
            filtered_entries.append(name)
        else:
            print(f"Skipping excluded path: {path}")
    
    for idx, name in enumerate(filtered_entries):
        path = os.path.join(root, name)
        is_last = (idx == len(filtered_entries) - 1)
        branch = '└── ' if is_last else '├── '
        lines.append(prefix + branch + name)
        
        if os.path.isdir(path):
            extension = '    ' if is_last else '│   '
            child_lines, child_files = get_tree(path, prefix + extension)
            lines.extend(child_lines)
            files.extend(child_files)
        else:
            # Double-check if the file should be excluded (as a safeguard)
            if not should_exclude(path):
                files.append(path)
    
    return lines, files


def main():
    print(f"Starting to scan directory: {DIR_PATH}")
    print("Exclusion rules:")
    for rule in EXCLUDE_LIST:
        print(f"  - {rule}")
    print("-" * 50)
    
    tree_lines, file_paths = get_tree(DIR_PATH)

    print(f"\nScan complete! Found {len(file_paths)} files")
    print(f"Writing to: {OUTPUT_FILE}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        # Write header information
        out.write(f"Directory structure: {DIR_PATH}\n")
        if os.name != 'nt':
            date_cmd = os.popen('date').read().strip()
        else:
            date_cmd = 'N/A'
        out.write(f"Generated at: {date_cmd}\n")
        out.write(f"Number of files included: {len(file_paths)}\n")
        out.write("Exclusion rules:\n")
        for rule in EXCLUDE_LIST:
            out.write(f"  - {rule}\n")
        out.write("\n" + "=" * 80 + "\n")
        
        # Write the directory tree
        for line in tree_lines:
            out.write(line + '\n')

        # Write the file contents
        out.write("\nStarting to print all file contents...\n")
        for idx, path in enumerate(file_paths, 1):
            out.write('\n' + '=' * 80 + '\n')
            out.write(f"File {idx}/{len(file_paths)}: {path}\n")
            out.write('-' * 80 + '\n')
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    out.write(content)
                    if not content.endswith('\n'):
                        out.write('\n')
            except Exception as e:
                out.write(f"[Could not read this file: {e}]\n")

    print(f"Done! Output saved to: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
