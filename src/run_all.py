import subprocess
import sys


def run_step(command):
    print(f"\nRunning: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def main():
    python = sys.executable

    run_step([python, "-m", "src.pipeline"])
    run_step([python, "-m", "src.eda"])
    run_step([python, "-m", "src.train"])
    run_step([python, "-m", "pytest", "-v"])

    print("\nChurnGuard completed successfully.")


if __name__ == "__main__":
    main()