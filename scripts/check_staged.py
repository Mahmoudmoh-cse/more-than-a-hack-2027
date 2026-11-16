"""Fail closed on unexpected staged paths; never print file contents.

This complements review, not a guarantee about arbitrary files. Existing organizer
baseline/public-GT assets are untouched and do not enter this staged additions gate.
"""

from pathlib import PurePosixPath
import subprocess
import sys


def main():
    raw = subprocess.check_output(['git','diff','--cached','--name-only','--diff-filter=ACMR','-z'])
    paths = raw.decode('utf-8').split('\0')
    allowed_top = {'.gitignore','.gitattributes','run_goalgorithm.py','requirements-goalgorithm.txt',
                   'requirements-report.txt','Goalgorithm.pdf'}
    allowed_dirs = {'goalgorithm','tests','configs','scripts','.github'}
    allowed_suffixes = {'.py','.md','.json','.yml','.yaml'}
    rejected = []
    checked = 0
    for name in filter(None,paths):
        path = PurePosixPath(name)
        okay = name in allowed_top or (len(path.parts) == 1 and name.startswith('GOALGORITHM_') and path.suffix == '.md')
        okay |= path.parts[0] in allowed_dirs and path.suffix in allowed_suffixes
        if not okay:
            rejected.append(name)
        else:
            checked += 1
    if rejected:
        print('FAIL: unexpected staged paths; inspect locally before committing.')
        return 1
    print(f'PASS: {checked} staged paths are permitted code/configuration/documentation artifacts.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
