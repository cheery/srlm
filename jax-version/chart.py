"""
chart.py — Draw a line chart in the terminal using unicode block elements.

Usage:
    echo "1 39.65\n2 32.76\n3 31.12" | python chart.py
    python chart.py data.txt
    python chart.py --title "epoch loss" --width 64 --height 32 data.txt

Input format: two columns, x and y, whitespace separated. One point per line.
Lines starting with # are ignored.

Unicode block elements used:
    ▁▂▃▄▅▆▇█  lower eighth blocks (for bar/fill)
    ─ │ ┼ ┤ ┐ └  box drawing for axes
    • ·          plot points
"""

import sys
import argparse
import math


EIGHTHS = " ▁▂▃▄▅▆▇█"   # index 0=empty, 8=full block


def read_data(source):
    points = []
    for line in source:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                x, y = float(parts[0]), float(parts[1])
                points.append((x, y))
            except ValueError:
                continue
    return points


def render(points, width=64, height=32, title=None):
    if not points:
        print("no data")
        return

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Add a little padding
    y_pad = (y_max - y_min) * 0.08 or 1.0
    y_lo = y_min - y_pad
    y_hi = y_max + y_pad

    # Reserve left margin for y-axis labels (7 chars) and right border
    label_w = 7
    plot_w  = width - label_w - 1   # -1 for right border char
    plot_h  = height - 2             # -2 for x-axis row + title row

    # Each cell is 1 column wide, 1 row high.
    # We use sub-cell vertical resolution via eighths: 8 sub-rows per cell.
    # Total vertical resolution: plot_h * 8

    total_sub = plot_h * 8

    def y_to_sub(y):
        """Map y value to sub-row index (0 = bottom)."""
        frac = (y - y_lo) / (y_hi - y_lo)
        return frac * total_sub

    def x_to_col(x):
        """Map x value to column index (0-indexed)."""
        if x_max == x_min:
            return plot_w // 2
        frac = (x - x_min) / (x_max - x_min)
        return int(frac * (plot_w - 1))

    # Build a grid: grid[row][col] = sub-height of the fill (0..8)
    # We draw a filled area chart: from y=0 (or y_lo) up to the data point.
    # For a line chart, just mark the exact cell with the appropriate eighth block.

    # grid[row][col] = character to draw
    grid = [[" "] * plot_w for _ in range(plot_h)]

    # For line chart: compute sub-row for each column, interpolating between points
    # First map all data points to columns
    col_values = {}   # col -> list of y values landing there
    for x, y in points:
        c = x_to_col(x)
        col_values.setdefault(c, []).append(y)

    # Interpolate to fill all columns
    # Sort points by x
    sorted_pts = sorted(points, key=lambda p: p[0])

    col_y = {}  # col -> interpolated y
    for i in range(plot_w):
        # Find x value corresponding to this column
        x = x_min + (x_max - x_min) * i / max(plot_w - 1, 1)
        # Linear interpolation
        # Find surrounding points
        lo_pt = sorted_pts[0]
        hi_pt = sorted_pts[-1]
        for j in range(len(sorted_pts) - 1):
            if sorted_pts[j][0] <= x <= sorted_pts[j+1][0]:
                lo_pt = sorted_pts[j]
                hi_pt = sorted_pts[j+1]
                break
        if hi_pt[0] == lo_pt[0]:
            y_interp = lo_pt[1]
        else:
            t = (x - lo_pt[0]) / (hi_pt[0] - lo_pt[0])
            y_interp = lo_pt[1] + t * (hi_pt[1] - lo_pt[1])
        col_y[i] = y_interp

    # Draw filled area: for each column, fill from bottom up to the data value
    for col, y in col_y.items():
        sub = y_to_sub(y)
        sub = max(0, min(total_sub, sub))
        full_rows = int(sub) // 8
        partial   = int(sub) % 8

        # Fill full rows from bottom
        for r in range(full_rows):
            row = plot_h - 1 - r
            if 0 <= row < plot_h:
                grid[row][col] = "█"

        # Partial row
        if partial > 0:
            row = plot_h - 1 - full_rows
            if 0 <= row < plot_h:
                grid[row][col] = EIGHTHS[partial]

    # Build y-axis tick labels (4 ticks)
    n_ticks = 4
    tick_rows = [int(plot_h * i / (n_ticks - 1)) for i in range(n_ticks)]
    tick_vals = {r: y_hi - (y_hi - y_lo) * r / (plot_h - 1) for r in tick_rows}

    # Assemble output lines
    lines = []

    # Title
    if title:
        pad = max(0, (width - len(title)) // 2)
        lines.append(" " * pad + title)
    else:
        lines.append("")

    # Plot rows
    for row in range(plot_h):
        # Y-axis label
        if row in tick_vals:
            label = f"{tick_vals[row]:6.1f} "
        else:
            label = " " * label_w

        # Axis line
        axis = "│"

        # Plot content with color
        content = ""
        for col in range(plot_w):
            ch = grid[row][col]
            if ch != " ":
                content += f"\033[34m{ch}\033[0m"
            else:
                content += ch

        # Right border at top only
        right = "┐" if row == 0 else "│" if row == plot_h - 1 else " "

        lines.append(label + axis + content + right)

    # X-axis
    x_axis_line = " " * label_w + "└" + "─" * plot_w + "┘"
    lines.append(x_axis_line)

    # X-axis labels: show first, middle, last
    x_label_line = " " * (label_w + 1)
    first_label  = f"{x_min:.4g}"
    mid_label    = f"{(x_min+x_max)/2:.4g}"
    last_label   = f"{x_max:.4g}"
    mid_pos      = (plot_w - len(mid_label)) // 2
    end_pos      = plot_w - len(last_label)

    buf = list(" " * plot_w)
    for i, ch in enumerate(first_label):
        if i < plot_w:
            buf[i] = ch
    for i, ch in enumerate(mid_label):
        pos = mid_pos + i
        if 0 <= pos < plot_w:
            buf[pos] = ch
    for i, ch in enumerate(last_label):
        pos = end_pos + i
        if 0 <= pos < plot_w:
            buf[pos] = ch
    x_label_line += "".join(buf)
    lines.append(x_label_line)

    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Terminal chart using unicode block elements.")
    parser.add_argument("file", nargs="?", help="data file (default: stdin)")
    parser.add_argument("--width",  type=int, default=64, help="total width in chars (default: 64)")
    parser.add_argument("--height", type=int, default=32, help="total height in rows (default: 32)")
    parser.add_argument("--title",  type=str, default=None)
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            points = read_data(f)
    else:
        points = read_data(sys.stdin)

    render(points, width=args.width, height=args.height, title=args.title)


if __name__ == "__main__":
    main()
