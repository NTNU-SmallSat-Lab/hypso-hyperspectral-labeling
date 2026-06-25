from pathlib import Path
import argparse
from .gui_app import run_gui

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cal", choices=["1a", "1b", "1c", "1d", "2a"], default="1a")
    parser.add_argument("--mission", choices=["H1", "H2"], default="H2")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    run_gui(args.nc, args.out, mission=args.mission, calibration=args.cal, verbose=args.verbose)

if __name__ == "__main__":
    main()