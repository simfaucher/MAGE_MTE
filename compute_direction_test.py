import argparse

from MTE import MTE

def convert_to_float(frac_str):
    """Convert a fraction written as string to a float"""

    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac

ap = argparse.ArgumentParser()
ap.add_argument("-x", "--x", required=True, default=0, type=int, help="x")
ap.add_argument("-y", "--y", required=True, default=0, type=int, help="y")
ap.add_argument("-w", "--width", required=False, default=1000, type=int, help="Width")
ap.add_argument("-r", "--ratio", required=False, default="16/9", help="Format resolution")
args = vars(ap.parse_args())

mte = MTE()
mte.format_resolution = convert_to_float(args["ratio"])
direction = mte.compute_direction((args["x"], args["y"]), (1, 1), args["width"])

print(direction)