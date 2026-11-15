"""Archive or restore existing portal activity without deleting events or evidence.

python -m snatchguard.central.cleanup --data runtime/central
python -m snatchguard.central.cleanup --data runtime/central --restore
"""
import argparse
import json
from pathlib import Path

from snatchguard.central.db import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='runtime/central')
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    store = Store(Path(args.data) / 'central.db')
    try:
        if args.restore:
            store.restore_activity()
            print('Restored archived activity to the portal.')
        else:
            print(json.dumps(store.archive_activity()))
    finally:
        store.db.close()


if __name__ == '__main__':
    main()
