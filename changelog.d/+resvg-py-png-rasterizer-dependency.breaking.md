`resvg-py`, the PNG rasterizer, is a dependency: every install draws a
schematic or an attached plot as a PNG, which is the only image an
assistant can see, without the `raster` extra. The extra stays, empty, so a
launch line that names it (`uvx --from 'ltspice-mcp[raster]' ltspice-mcp`)
still works. `resvg-py` ships prebuilt wheels for Windows, macOS and Linux
and loads no native library. Where it cannot be imported, the reply's
`missing` (in `inspect(kind="capabilities")`'s `render`, and in a render's
`png_unavailable`) now reads `"dependency"` instead of `"extra"`, and the
remedy is to reinstall the server (`uv sync` for a server run from a
source checkout).
