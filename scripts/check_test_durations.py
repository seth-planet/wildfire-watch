#!/usr/bin/env python3
"""
Check test durations and identify tests that should be marked as slow.
Run with: python scripts/check_test_durations.py
"""
import subprocess
import sys
import re
from pathlib import Path

def run_tests_with_durations():
    """Run pytest with duration reporting"""
    print("Running tests to collect duration data...")
    result = subprocess.run([
        sys.executable, '-m', 'pytest',
        '--durations=0',  # Show all test durations
        '--tb=no',        # No traceback for failures
        '--no-header',    # Cleaner output
        '-q',             # Quiet mode
        'tests/'
    ], capture_output=True, text=True)
    
    return result.stdout

def parse_durations(output):
    """Parse pytest duration output"""
    durations = []
    
    # Look for duration lines (format: "0.50s call     tests/test_foo.py::test_bar")
    duration_pattern = r'(\d+\.\d+)s\s+call\s+(.+?)::(.+?)(?:\[|$)'
    
    for line in output.split('\n'):
        match = re.match(duration_pattern, line)
        if match:
            duration = float(match.group(1))
            test_file = match.group(2)
            test_name = match.group(3)
            durations.append({
                'duration': duration,
                'file': test_file,
                'test': test_name,
                'full_name': f"{test_file}::{test_name}"
            })
    
    return sorted(durations, key=lambda x: x['duration'], reverse=True)

def check_markers(test_file, test_name):
    """Check if a test has slow/very_slow markers"""
    try:
        with open(test_file, 'r') as f:
            content = f.read()
        
        # Find the test function
        test_pattern = rf'def {re.escape(test_name)}\('
        test_match = re.search(test_pattern, content)
        
        if not test_match:
            return {'slow': False, 'very_slow': False, 'timeout': False}
        
        # Look backwards from test definition for markers
        before_test = content[:test_match.start()]
        lines = before_test.split('\n')
        
        # Check last few lines before test for markers
        recent_lines = '\n'.join(lines[-10:])
        
        return {
            'slow': '@pytest.mark.slow' in recent_lines,
            'very_slow': '@pytest.mark.very_slow' in recent_lines,
            'timeout': '@pytest.mark.timeout' in recent_lines
        }
    except Exception as e:
        print(f"Error checking markers for {test_file}: {e}")
        return {'slow': False, 'very_slow': False, 'timeout': False}

def main():
    """Main function"""
    output = run_tests_with_durations()
    durations = parse_durations(output)
    
    if not durations:
        print("No duration data collected. Make sure tests are running.")
        return 1
    
    print(f"\nAnalyzed {len(durations)} tests\n")
    
    # Tests that should be marked slow (>60s)
    should_be_slow = []
    # Tests that should be marked very_slow (>300s)
    should_be_very_slow = []
    # Tests that are correctly marked
    correctly_marked = []
    # Tests that are over-marked (marked slow but fast)
    over_marked = []
    
    for test_info in durations:
        duration = test_info['duration']
        markers = check_markers(test_info['file'], test_info['test'])
        
        if duration >= 300:  # 5 minutes
            if not markers['very_slow'] and not markers['timeout']:
                should_be_very_slow.append(test_info)
            else:
                correctly_marked.append(test_info)
        elif duration >= 60:  # 1 minute
            if not markers['slow'] and not markers['very_slow'] and not markers['timeout']:
                should_be_slow.append(test_info)
            else:
                correctly_marked.append(test_info)
        else:
            # Fast test
            if markers['slow'] or markers['very_slow']:
                over_marked.append(test_info)
    
    # Report findings
    if should_be_very_slow:
        print("❌ Tests that should be marked @pytest.mark.very_slow (>5 min):")
        for test in should_be_very_slow:
            print(f"   {test['full_name']} - {test['duration']:.1f}s")
        print()
    
    if should_be_slow:
        print("⚠️  Tests that should be marked @pytest.mark.slow (>1 min):")
        for test in should_be_slow:
            print(f"   {test['full_name']} - {test['duration']:.1f}s")
        print()
    
    if over_marked:
        print("🔍 Tests that might be over-marked (marked slow but run fast):")
        for test in over_marked:
            print(f"   {test['full_name']} - {test['duration']:.1f}s")
        print()
    
    # Summary
    print("\n📊 Summary:")
    print(f"   Total tests: {len(durations)}")
    print(f"   Need 'very_slow' marker: {len(should_be_very_slow)}")
    print(f"   Need 'slow' marker: {len(should_be_slow)}")
    print(f"   Correctly marked: {len(correctly_marked)}")
    print(f"   Over-marked: {len(over_marked)}")
    
    # Show top 10 slowest tests
    print("\n🐌 Top 10 slowest tests:")
    for i, test in enumerate(durations[:10], 1):
        print(f"   {i}. {test['full_name']} - {test['duration']:.1f}s")
    
    # Return non-zero if any tests need markers
    return 1 if (should_be_slow or should_be_very_slow) else 0

if __name__ == "__main__":
    sys.exit(main())