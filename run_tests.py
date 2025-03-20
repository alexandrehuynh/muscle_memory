import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run tests for Muscle Memory')
    parser.add_argument('--type', choices=['unit', 'integration', 'e2e', 'all'], 
                        default='all', help='Type of tests to run')
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    args = parser.parse_args()
    
    # Generate test data if needed
    if not os.path.exists("test_data/videos/squat_test.mp4"):
        print("Generating test data...")
        subprocess.run(["python", "scripts/create_test_data.py"])
    
    # Build command
    cmd = ["pytest"]
    
    if args.type != 'all':
        cmd.extend(["-m", args.type])
    
    if args.coverage:
        cmd.extend(["--cov=muscle_memory", "--cov-report=html", "--cov-report=term"])
    
    # Run tests
    subprocess.run(cmd)

if __name__ == "__main__":
    main()