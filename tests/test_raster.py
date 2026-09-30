"""Behavior tests for image return: rasterization and the MCP image response.

Rasterization depends on an optional package, so the tests that need a real PNG
skip when it is absent — but the *degradation* path is tested unconditionally by
failing the import the way a deployment without it does (the
``raster_extra_missing`` and ``raster_native_missing`` fixtures in conftest).
That is the branch such a deployment actually runs, so it must not be the
branch that only gets exercised when someone happens not to have cairosvg
installed.
"""

from __future__ import annotations

import base64
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import pytest
from mcp import types

from ltspice_mcp.lib import raster
from ltspice_mcp.lib.raster import (
    DEFAULT_SCALE,
    PNG,
    SVG,
    RasterSupport,
    RasterUnavailableError,
    RenderedImage,
    raster_available,
    raster_support,
    render_image,
)
from ltspice_mcp.tools._base import image_response
from tests.conftest import FailingCairosvgImport, needs_raster

# Smallest thing cairosvg will accept that still has measurable extent.
TINY_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 10" width="20" height="10">'
    '<rect x="0" y="0" width="20" height="10" fill="black"/>'
    "</svg>"
)

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------


@needs_raster
def test_png_request_returns_real_png_bytes() -> None:
    image = render_image(TINY_SVG, image_format=PNG, scale=1.0)
    assert image.image_format == PNG
    assert image.mime_type == "image/png"
    assert image.scale == 1.0
    assert image.note is None
    assert image.data.startswith(_PNG_MAGIC)
    assert image.png_unavailable is None


@needs_raster
def test_scale_is_the_cost_lever() -> None:
    # Asserted in pixels, not bytes: the source viewBox is 20x10, so these exact
    # dimensions prove the float scale reached the rasterizer and was applied,
    # past a type stub that declares the parameter as an int.
    small = render_image(TINY_SVG, image_format=PNG, scale=1.0)
    large = render_image(TINY_SVG, image_format=PNG, scale=3.0)
    assert (small.width, small.height) == (20, 10)
    assert (large.width, large.height) == (60, 30)
    assert large.scale == 3.0


@needs_raster
def test_text_rendering_survives_caller_thread_exit() -> None:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100">'
        '<text x="10" y="30" font-family="Georgia" font-size="14">R1 10k</text>'
        "</svg>"
    )
    # Per-request executor threads may exit between calls. Native font caches
    # must not retain resources owned by those now-dead threads.
    for scale in (3.0, 0.1):
        with ThreadPoolExecutor(max_workers=1) as caller:
            image = caller.submit(render_image, svg, image_format=PNG, scale=scale).result()
        assert (image.width, image.height) == (int(200 * scale), int(100 * scale))
        assert image.data.startswith(_PNG_MAGIC)


def test_scale_is_validated_without_the_extra(raster_extra_missing: None) -> None:
    # Whether an invalid argument is rejected must not depend on which optional
    # packages are installed, or the same call raises on one machine and quietly
    # degrades on another.
    with pytest.raises(ValueError, match="scale must be positive"):
        render_image(TINY_SVG, image_format=PNG, scale=-1)
    with pytest.raises(ValueError, match="scale must be positive"):
        raster.rasterize_svg(TINY_SVG, scale=-1)


def test_scale_is_validated_for_vector_requests_too(raster_extra_missing: None) -> None:
    with pytest.raises(ValueError, match="scale must be positive"):
        render_image(TINY_SVG, image_format=SVG, scale=0)


def test_svg_request_never_rasterizes(raster_extra_missing: None) -> None:
    # Vector output must not depend on the optional package at all.
    image = render_image(TINY_SVG, image_format=SVG)
    assert image.image_format == SVG
    assert image.mime_type == "image/svg+xml"
    assert image.scale is None
    assert image.note is None
    # Asking for SVG is not a degraded PNG, even where no rasterizer exists.
    assert image.png_unavailable is None
    assert image.data.decode("utf-8") == TINY_SVG


def test_png_request_degrades_to_svg_without_the_extra(raster_extra_missing: None) -> None:
    # The point of the mechanism: a missing optional dependency downgrades the
    # render, it does not fail the call.
    image = render_image(TINY_SVG, image_format=PNG)
    assert image.image_format == SVG
    assert image.data.decode("utf-8") == TINY_SVG
    assert image.note is not None
    assert "cairosvg" in image.note
    assert "raster" in image.note  # names the extra that fixes it
    # The same fact as data, for a caller that acts on it rather than reads it.
    assert image.png_unavailable is not None
    assert image.png_unavailable.missing == "extra"
    assert image.png_unavailable.remedy is not None
    assert "ltspice-mcp[raster]" in image.png_unavailable.remedy


def test_missing_extra_is_reported_as_the_extra(raster_extra_missing: None) -> None:
    support = raster_support()
    assert support.png is False
    assert support.missing == "extra"
    assert support.reason is not None and "'raster' extra" in support.reason
    assert support.remedy is not None and "ltspice-mcp[raster]" in support.remedy
    assert raster_available() is False


def test_missing_native_library_is_reported_as_the_library(
    raster_native_missing: None,
) -> None:
    """The extra is installed, so telling the caller to install it sends them
    round a loop: the reason names the native library, and the remedy is the
    one for this platform."""
    support = raster_support()
    assert support.png is False
    assert support.missing == "native_library"
    assert support.reason is not None
    assert "Cairo" in support.reason
    assert "not installed" not in support.reason
    # cairocffi's own words stay in the reason, first line only: it says which
    # names were tried, and a multi-line message does not fit in one sentence.
    assert 'no library called "cairo-2" was found' in support.reason
    assert "\n" not in support.reason
    assert support.remedy == raster.native_library_remedy(sys.platform)


def test_a_failed_probe_is_not_repeated(raster_native_missing: FailingCairosvgImport) -> None:
    # A failed import is not cached by Python, and each retry reruns
    # cairocffi's library search. The remedy ends in a restart, so the loader
    # answers from its first attempt.
    first = raster_support()
    render_image(TINY_SVG, image_format=PNG)
    assert raster_support() == first
    assert raster_native_missing.attempts == 1


def test_native_library_remedy_is_per_platform() -> None:
    # On Windows the dependent DLLs are found through PATH or cairocffi's own
    # variable, not the loader's default search, so the remedy has to name it.
    windows = raster.native_library_remedy("win32")
    assert "CAIROCFFI_DLL_DIRECTORIES" in windows
    assert "libcairo-2.dll" in windows
    assert "brew install cairo" in raster.native_library_remedy("darwin")
    assert "libcairo2" in raster.native_library_remedy("linux")
    for platform in ("win32", "darwin", "linux"):
        assert "restart" in raster.native_library_remedy(platform)


@needs_raster
def test_a_working_rasterizer_reports_png_support() -> None:
    support = raster_support()
    assert asdict(support) == {"png": True, "missing": None, "reason": None, "remedy": None}


def test_direct_rasterize_raises_when_unavailable(raster_extra_missing: None) -> None:
    # render_image degrades; the lower-level call is explicit about failing, so
    # a caller that genuinely requires a bitmap can detect it.
    with pytest.raises(RasterUnavailableError, match="cairosvg") as caught:
        raster.rasterize_svg(TINY_SVG)
    assert caught.value.support.missing == "extra"


def test_the_unavailable_error_survives_pickling(raster_native_missing: None) -> None:
    # A Python API caller may carry it across a process boundary; what is
    # missing and how to fix it must arrive with the message.
    with pytest.raises(RasterUnavailableError) as caught:
        raster.rasterize_svg(TINY_SVG)
    copy = pickle.loads(pickle.dumps(caught.value))
    assert copy.support == caught.value.support
    assert str(copy) == str(caught.value)


def test_unknown_format_is_rejected() -> None:
    # The Literal stops this at type-check time, which is the point of it, but a
    # format arriving from JSON is never type-checked — so the runtime guard has
    # to hold too. Deliberately passing what the annotation forbids.
    with pytest.raises(ValueError, match="unknown image_format"):
        render_image(TINY_SVG, image_format="jpeg")  # pyright: ignore[reportArgumentType]


def test_non_positive_scale_is_rejected() -> None:
    # Deliberately NOT gated on the extra — see the two tests above; gating this
    # is what let the environment-dependent contract hide.
    with pytest.raises(ValueError, match="scale must be positive"):
        raster.rasterize_svg(TINY_SVG, scale=0)


@needs_raster
def test_raster_reports_its_pixel_size_and_cost() -> None:
    # Dimensions come from the PNG itself, and are what a consumer needs to
    # decide whether to send this render or re-render it smaller.
    image = render_image(TINY_SVG, image_format=PNG, scale=2.0)
    assert image.width == 40  # 20px viewBox at 2.0
    assert image.height == 20
    assert image.estimated_tokens == round(40 * 20 / 750)


def test_vector_has_no_pixel_size_or_token_estimate(raster_extra_missing: None) -> None:
    image = render_image(TINY_SVG, image_format=SVG)
    assert image.width is None
    assert image.height is None
    assert image.estimated_tokens is None


def test_png_size_tolerates_a_non_png() -> None:
    assert raster._png_size(b"not a png at all") == (None, None)


def test_default_scale_is_the_measured_one() -> None:
    # 1.5 is the lowest scale that has been checked for legibility, and
    # detection did not degrade there. Pinned because that floor is unlocated and
    # lies somewhere at or below this: going lower is not a tuning change, it
    # steps off measured ground, and an under-scaled render fails silently.
    assert DEFAULT_SCALE == 1.5


# ---------------------------------------------------------------------------
# MCP response shape
# ---------------------------------------------------------------------------


def _blocks(result: types.CallToolResult, kind: type) -> list:
    return [c for c in result.content if isinstance(c, kind)]


def test_raster_is_returned_as_an_image_block() -> None:
    image = RenderedImage(data=b"\x89PNG-ish", image_format=PNG, mime_type="image/png", scale=2.0)
    result = image_response(image, "rendered rc.asc")

    images = _blocks(result, types.ImageContent)
    assert len(images) == 1
    assert images[0].mime_type == "image/png"
    assert base64.b64decode(images[0].data) == b"\x89PNG-ish"


def test_vector_is_returned_as_text_not_an_image_block() -> None:
    # image/svg+xml is not reliably rendered by clients; sending markup as an
    # image block would show the model a blank space instead of a schematic.
    image = RenderedImage(data=TINY_SVG.encode(), image_format=SVG, mime_type="image/svg+xml")
    result = image_response(image, "rendered rc.asc")

    assert not _blocks(result, types.ImageContent)
    texts = [c.text for c in _blocks(result, types.TextContent)]
    assert TINY_SVG in texts


def test_metadata_survives_for_structured_only_clients() -> None:
    # A client that renders only structuredContent must still learn that it
    # asked for a PNG and got SVG, without decoding the payload.
    image = RenderedImage(
        data=b"<svg/>",
        image_format=SVG,
        mime_type="image/svg+xml",
        png_unavailable=RasterSupport(
            png=False, missing="extra", reason="cairosvg missing", remedy="install it"
        ),
    )
    result = image_response(image, "rendered", {"path": "/tmp/rc.asc"})

    assert result.structured_content is not None
    payload = result.structured_content
    assert payload["path"] == "/tmp/rc.asc"
    assert payload["image"]["image_format"] == SVG
    assert payload["image"]["scale"] is None
    assert payload["image"]["bytes"] == len(b"<svg/>")
    assert "cairosvg" in payload["image"]["note"]


def test_raster_response_carries_caller_text_and_full_metadata() -> None:
    # The caller's message must survive alongside the image, and the raster's
    # own scale/size/bytes must reach a structured-only client.
    image = RenderedImage(
        data=b"\x89PNG-ish",
        image_format=PNG,
        mime_type="image/png",
        scale=3.0,
        width=60,
        height=30,
    )
    result = image_response(image, "rendered rc.asc", {"path": "/tmp/rc.asc"})

    # Image first, then the caller's text — a client that shows only the first
    # block should show the picture.
    assert [type(c).__name__ for c in result.content] == ["ImageContent", "TextContent"]
    assert _blocks(result, types.TextContent)[0].text == "rendered rc.asc"

    assert result.structured_content is not None
    meta = result.structured_content["image"]
    assert meta["scale"] == 3.0
    assert (meta["width"], meta["height"]) == (60, 30)
    assert meta["bytes"] == len(b"\x89PNG-ish")
    assert meta["note"] is None


def test_caller_cannot_clobber_the_image_metadata_key() -> None:
    image = RenderedImage(data=b"x", image_format=SVG, mime_type="image/svg+xml")
    with pytest.raises(ValueError, match="owns the 'image' key"):
        image_response(image, "rendered", {"image": "something else"})


@needs_raster
def test_end_to_end_svg_to_image_block() -> None:
    result = image_response(render_image(TINY_SVG, scale=1.0), "rendered")
    images = _blocks(result, types.ImageContent)
    assert len(images) == 1
    assert base64.b64decode(images[0].data).startswith(_PNG_MAGIC)
    assert result.structured_content is not None
    assert result.structured_content["image"]["image_format"] == PNG


def test_a_missing_native_cairo_library_counts_as_no_rasterizer(
    raster_native_missing: None,
) -> None:
    """cairocffi raises OSError, not ImportError, when cairosvg is installed
    but libcairo is not — the state of a CI runner or a bare host that pip
    installed the extra on. That must read as "no rasterizer here", not escape
    from a render call."""
    assert raster_available() is False
    image = render_image(TINY_SVG, image_format=PNG)
    assert image.image_format == SVG
    assert image.data.decode("utf-8") == TINY_SVG
    assert image.png_unavailable is not None
    assert image.png_unavailable.missing == "native_library"
    # The extra is installed; the note blames the library, not the extra.
    assert image.note is not None and "Cairo" in image.note
    assert "not installed" not in image.note
