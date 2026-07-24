"""Turn rendered SVG into a returnable image, rasterizing when possible.

The schematic and result renderers are SVG-only by design. A model consuming a
render over MCP generally wants a bitmap, so this module is the one place that
converts SVG markup to PNG and describes what the caller actually got.

Rasterization needs ``cairosvg``, which is an **optional** dependency (extra:
``raster``). Without it this module still returns the SVG unchanged rather than
failing — a render that arrives as markup is degraded, not broken, and the
caller is told so through :attr:`RenderedImage.note` rather than through an
exception.

Interface note: ``image_format`` is a named choice rather than a boolean so a
third format can be added without changing the shape of every call site, and
``scale`` is reported back as applied (``None`` for vector) so a caller never
has to infer what it received from what it asked for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ltspice_mcp.errors import LTSpiceMCPError

ImageFormat = Literal["png", "svg"]

# Scale is the only real cost lever. Image cost tracks pixel area (roughly
# width*height/750), so it grows with the SQUARE of scale, while format does not
# move it at all: lossless WebP is ~57% fewer bytes at the same token count, and
# JPEG is both larger and lossy on thin lines and small text. The area rule is
# general; absolute token figures are per-drawing, so read `estimated_tokens`
# off a given render rather than assuming a number.
#
# The default is 1.5x on the strength of a three-arm blinded study
# (benchmarks/render_gate/scale_study: 3.0x / 2.0x / 1.5x, nine fresh raters,
# frozen rubric, sealed arm key). Recall did not degrade at any tested scale:
# every rater found every seeded defect in every fixture in every arm. So the
# 4x cost saving from 3.0x costs nothing measurable here.
#
# Read that for exactly what it is. Recall saturated at 100%, so the gate had no
# discriminating power in this run — it measured a ceiling, not a floor. It did
# not show that 1.5x is enough in general; it showed that these fixtures cannot
# tell the three scales apart. The legibility floor is unlocated and sits at or
# below 1.5x. Finding it needs harder fixtures or lower scales, not a rerun of
# the same one.
#
# The residual to watch is stroke width, not density. Crowding is a property of
# the layout and does not change with scale, and feature size turned out to be
# fixture-independent (font-size 11 or 14 px, stroke 1 or 1.5 px on all seven),
# so a busier sheet does not need a bigger number. What does thin out is the
# stroke: at 1.5x a 1-1.5 px line lands on 1.5-2.25 device pixels, where
# aliasing between two nearly-coincident lines becomes plausible. A drawing that
# turns on telling such lines apart is the case to raise `scale` for.
DEFAULT_SCALE = 1.5

PNG = "png"
SVG = "svg"
IMAGE_FORMATS = (PNG, SVG)

_MIME = {PNG: "image/png", SVG: "image/svg+xml"}


@dataclass(frozen=True)
class RenderedImage:
    """An image ready to hand back, plus what it actually turned out to be.

    ``image_format`` and ``scale`` describe the result, not the request: when
    rasterization is unavailable a PNG request yields ``image_format="svg"``,
    ``scale=None``, and a ``note`` saying why.
    """

    data: bytes
    image_format: ImageFormat
    mime_type: str
    scale: float | None = None
    note: str | None = None
    width: int | None = None
    height: int | None = None

    @property
    def is_raster(self) -> bool:
        return self.image_format == PNG

    @property
    def estimated_tokens(self) -> int | None:
        """Roughly what this image costs a model to look at, or ``None``.

        Image cost is driven by pixel area (see :data:`DEFAULT_SCALE`), so a
        caller can decide whether to send a render, or re-render smaller, without
        having to know the rule. ``None`` for vector output, which has no pixel
        extent and is charged as text.
        """
        if self.width is None or self.height is None:
            return None
        return round(self.width * self.height / 750)

    def to_dict(self) -> dict[str, object]:
        """Metadata a caller can act on without decoding the image itself."""
        return {
            "image_format": self.image_format,
            "mime_type": self.mime_type,
            "scale": self.scale,
            "width": self.width,
            "height": self.height,
            "bytes": len(self.data),
            "estimated_tokens": self.estimated_tokens,
            "note": self.note,
        }


class RasterUnavailableError(LTSpiceMCPError):
    """Rasterization was requested but the optional dependency is not installed.

    Part of the project error hierarchy so a consumer calling
    :func:`rasterize_svg` directly raises something the server already handles,
    rather than an exception that escapes to the protocol layer.
    """


def _check_scale(scale: float) -> None:
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")


def raster_available() -> bool:
    """Whether SVG can be rasterized in this environment."""
    return _load_cairosvg() is not None


def _load_cairosvg():
    """Import ``cairosvg`` if installed; ``None`` when the extra is absent.

    Imported lazily rather than at module load: the dependency is optional, and
    importing it costs a non-trivial shared-library load that a server never
    rendering an image should not pay.
    """
    try:
        import cairosvg
    except ImportError:
        return None
    return cairosvg


def rasterize_svg(svg: str, *, scale: float = DEFAULT_SCALE, background: str = "white") -> bytes:
    """Rasterize ``svg`` markup to PNG bytes at ``scale``.

    Raises :class:`RasterUnavailableError` when the optional dependency is
    missing — callers wanting graceful degradation should use
    :func:`render_image` instead.
    """
    # Validate before loading: doing it the other way round makes rejecting a
    # bad scale depend on whether the optional package happens to be installed,
    # so the same call raises on one machine and silently degrades on another.
    _check_scale(scale)
    cairosvg = _load_cairosvg()
    if cairosvg is None:
        raise RasterUnavailableError(
            "PNG rasterization needs the optional 'cairosvg' dependency "
            "(install the 'raster' extra: pip install 'ltspice-mcp[raster]')"
        )
    png = cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        # cairosvg's type stub declares scale as int, but the library takes a
        # float and a fractional scale is the entire point of the parameter —
        # test_scale_is_the_cost_lever pins that 3.0 really does yield a larger
        # image than 1.0 at runtime.
        scale=scale,  # pyright: ignore[reportArgumentType]
        background_color=background,
    )
    if not isinstance(png, bytes):
        # Its stub admits None; that only happens when writing to a file, which
        # this call never does. Fail loudly rather than pass a non-image on.
        raise TypeError(f"cairosvg returned {type(png).__name__}, expected PNG bytes")
    return png


def render_image(
    svg: str,
    *,
    image_format: ImageFormat = PNG,
    scale: float = DEFAULT_SCALE,
) -> RenderedImage:
    """Package rendered ``svg`` as the requested image format.

    Always returns an image for a well-formed request. Asking for PNG without
    ``cairosvg`` installed yields the SVG with an explanatory ``note`` instead of
    raising, so a missing optional dependency degrades the render rather than
    failing the call.

    A malformed request is a different thing from a degraded environment, and is
    rejected either way: arguments are validated before anything is loaded, so
    the same call behaves identically with or without the optional package.
    """
    if image_format not in IMAGE_FORMATS:
        raise ValueError(
            f"unknown image_format {image_format!r}; expected one of {', '.join(IMAGE_FORMATS)}"
        )
    _check_scale(scale)

    if image_format == SVG:
        return RenderedImage(
            data=svg.encode("utf-8"),
            image_format=SVG,
            mime_type=_MIME[SVG],
        )

    try:
        png = rasterize_svg(svg, scale=scale)
    except RasterUnavailableError as exc:
        return RenderedImage(
            data=svg.encode("utf-8"),
            image_format=SVG,
            mime_type=_MIME[SVG],
            note=f"returned SVG instead of PNG: {exc}",
        )
    width, height = _png_size(png)
    return RenderedImage(
        data=png,
        image_format=PNG,
        mime_type=_MIME[PNG],
        scale=scale,
        width=width,
        height=height,
    )


def _png_size(png: bytes) -> tuple[int | None, int | None]:
    """Pixel dimensions from a PNG's IHDR header, or ``(None, None)``.

    Read from the bytes rather than derived from the source SVG and the scale:
    the rasterizer rounds, so the two disagree, and the number a caller reasons
    about should be the one it is actually sending.
    """
    # 8-byte signature, then a 4-byte length + "IHDR", then width and height as
    # big-endian 32-bit integers.
    if len(png) < 24 or png[12:16] != b"IHDR":
        return (None, None)
    return (
        int.from_bytes(png[16:20], "big"),
        int.from_bytes(png[20:24], "big"),
    )
