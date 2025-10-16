import sys
import subprocess

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <rg command and arguments>")
        sys.exit(1)
    
    # Get all arguments except the script name (sys.argv[0])
    rg_command = sys.argv[1:]
    
    print(rg_command)
    print(f"Full ripgrep command: {' '.join(rg_command)}")


if __name__ == "__main__":
    main()