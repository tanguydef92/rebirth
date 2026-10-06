#!/usr/bin/env python3
"""Command-line entry point for individual-drone PPO training."""
import sys
from training.ppo import main

if __name__ == '__main__':
    try: sys.exit(main())
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
