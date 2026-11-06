The Python API's `load_raw` takes `window=(start, end)`: a read that keeps
only the points of the plot's axis from `start` to `end`, in each step, by the
rule the exports' windows use, alone or with `signals`. The bounds are in the
axis's unit, as numbers or SPICE text such as `"1m"`, either one open. The
512 MiB limit on a read's numbers applies to the points kept, so a run with
too many points for one read can be read a stretch at a time. A plot with no
axis, an axis that decreases within a step, and a window that meets no point
of a step are refused with the reason. A file with too many points for one
read is now refused with the numbers it holds and the ways around it, the
window among them, where it said only "points limit exceeded".
