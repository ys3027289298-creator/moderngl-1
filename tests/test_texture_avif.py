"""Tests for AVIF texture loading (moderngl.avif / Context.texture_avif).

Fixtures are generated on the fly:
* 8-bit RGB/RGBA files are encoded with pillow-avif-plugin (skipped if missing).
* 10-bit and 12-bit files are encoded with ffmpeg/libaom (skipped if missing).
"""
import shutil
import subprocess

import numpy as np
import pytest

avif = pytest.importorskip("moderngl.avif")
moderngl = pytest.importorskip("moderngl")

if not avif.libavif_available():
    pytest.skip("libavif is not available", allow_module_level=True)


def _gradient_rgba(width, height):
    x = np.linspace(0, 255, width, dtype=np.float32)
    y = np.linspace(0, 255, height, dtype=np.float32)
    r = np.broadcast_to(x[None, :], (height, width))
    g = np.broadcast_to(y[:, None], (height, width))
    b = np.broadcast_to((x[None, :] + y[:, None]) / 2, (height, width))
    a = np.broadcast_to(255 - y[:, None], (height, width))
    return np.stack([r, g, b, a], axis=-1).round().astype(np.uint8)


def _encode_pillow(path, pixels):
    Image = pytest.importorskip("PIL.Image")
    pytest.importorskip("pillow_avif")
    mode = "RGBA" if pixels.shape[2] == 4 else "RGB"
    Image.fromarray(pixels, mode).save(path, quality=100)


def _encode_ffmpeg(path, source, pix_fmt):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not available")
    height, width = source.shape[:2]
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{width}x{height}",
        "-i", "-",
        "-frames:v", "1", "-pix_fmt", pix_fmt, "-strict", "experimental",
        str(path),
    ]
    proc = subprocess.run(cmd, input=source.tobytes(), capture_output=True)
    if proc.returncode != 0:
        pytest.skip("ffmpeg cannot encode %s: %s" % (pix_fmt, proc.stderr.decode()))


def _encode_avifenc_10bit_alpha(path, source):
    """Encode a 10-bit RGBA AVIF with avifenc (ffmpeg cannot keep 10-bit alpha)."""
    avifenc = shutil.which("avifenc")
    if avifenc is None or shutil.which("ffmpeg") is None:
        pytest.skip("avifenc/ffmpeg are not available")
    height, width = source.shape[:2]
    rgba16 = (source.astype(np.uint32) * 257).astype(np.uint16)
    png_path = path.with_suffix(".png")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgba64le", "-s", f"{width}x{height}",
        "-i", "-", str(png_path),
    ]
    subprocess.run(cmd, input=rgba16.tobytes(), capture_output=True, check=True)
    cmd = [avifenc, "-d", "10", "-y", "444", "-s", "10", str(png_path), str(path)]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        pytest.skip("avifenc failed: %s" % proc.stderr.decode())


@pytest.fixture(scope="module")
def gradient():
    return _gradient_rgba(64, 48)


@pytest.fixture(scope="module")
def avif_rgb8(tmp_path_factory, gradient):
    path = tmp_path_factory.mktemp("avif") / "rgb8.avif"
    _encode_pillow(path, gradient[..., :3])
    return path, gradient[..., :3]


@pytest.fixture(scope="module")
def avif_rgba8(tmp_path_factory, gradient):
    path = tmp_path_factory.mktemp("avif") / "rgba8.avif"
    _encode_pillow(path, gradient)
    return path, gradient


@pytest.fixture(scope="module")
def avif_rgb10(tmp_path_factory, gradient):
    path = tmp_path_factory.mktemp("avif") / "rgb10.avif"
    _encode_ffmpeg(path, gradient, "yuv420p10le")
    return path, gradient[..., :3]


@pytest.fixture(scope="module")
def avif_rgba10(tmp_path_factory, gradient):
    path = tmp_path_factory.mktemp("avif") / "rgba10.avif"
    _encode_avifenc_10bit_alpha(path, gradient)
    return path, gradient


@pytest.fixture(scope="module")
def avif_rgb12(tmp_path_factory, gradient):
    path = tmp_path_factory.mktemp("avif") / "rgb12.avif"
    _encode_ffmpeg(path, gradient, "yuv420p12le")
    return path


@pytest.fixture(scope="module")
def avif_corrupt(tmp_path_factory, avif_rgb8):
    path = tmp_path_factory.mktemp("avif") / "corrupt.avif"
    data = avif_rgb8[0].read_bytes()
    path.write_bytes(data[: len(data) // 2])
    return path


def test_decode_rgb8(avif_rgb8):
    path, _ = avif_rgb8
    with avif.decode_avif(path) as img:
        assert img.size == (64, 48)
        assert img.depth == 8
        assert img.has_alpha is False
        assert img.components == 3
        assert img.dtype == "f1"
        assert len(img.pixels) == 64 * 48 * 3


def test_decode_from_bytes(avif_rgb8):
    path, _ = avif_rgb8
    with avif.decode_avif(path.read_bytes()) as img:
        assert img.size == (64, 48)
        assert img.components == 3


def test_texture_rgb8(ctx, avif_rgb8):
    path, expected = avif_rgb8
    texture = ctx.texture_avif(path)
    assert texture.size == (64, 48)
    assert texture.components == 3
    assert texture.dtype == "f1"
    actual = np.frombuffer(texture.read(), dtype=np.uint8).reshape(48, 64, 3)
    # lossy YUV420: smooth gradients must survive within a few levels
    assert np.abs(actual.astype(int) - expected.astype(int)).max() <= 8


def test_texture_rgba8(ctx, avif_rgba8):
    path, expected = avif_rgba8
    texture = ctx.texture_avif(path)
    assert texture.components == 4
    actual = np.frombuffer(texture.read(), dtype=np.uint8).reshape(48, 64, 4)
    assert np.abs(actual.astype(int) - expected.astype(int)).max() <= 8


def test_texture_rgb10(ctx, avif_rgb10):
    path, expected = avif_rgb10
    texture = ctx.texture_avif(path)
    assert texture.size == (64, 48)
    assert texture.components == 3
    assert texture.dtype == "f2"
    actual = np.frombuffer(texture.read(), dtype=np.float16).reshape(48, 64, 3)
    assert actual.min() >= 0.0 and actual.max() <= 1.0
    reference = expected.astype(np.float32) / 255.0
    assert np.abs(actual - reference).max() <= 0.03


def test_texture_rgba10(ctx, avif_rgba10):
    path, expected = avif_rgba10
    texture = ctx.texture_avif(path)
    assert texture.components == 4
    assert texture.dtype == "f2"
    actual = np.frombuffer(texture.read(), dtype=np.float16).reshape(48, 64, 4)
    reference = expected.astype(np.float32) / 255.0
    assert np.abs(actual - reference).max() <= 0.03


def test_unsupported_bit_depth(ctx, avif_rgb12):
    with pytest.raises(avif.AvifUnsupportedError, match="12"):
        ctx.texture_avif(avif_rgb12)


def test_corrupt_file(ctx, avif_corrupt):
    with pytest.raises(avif.AvifError):
        ctx.texture_avif(avif_corrupt)


def test_not_an_avif(ctx, tmp_path):
    path = tmp_path / "fake.avif"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    with pytest.raises(avif.AvifError):
        ctx.texture_avif(path)


def test_missing_file(ctx, tmp_path):
    with pytest.raises(avif.AvifError):
        ctx.texture_avif(tmp_path / "does-not-exist.avif")


def test_mip_chain_uploaded(ctx, avif_rgb8):
    path, _ = avif_rgb8
    texture = ctx.texture_avif(path)
    level0 = np.frombuffer(texture.read(level=0), dtype=np.uint8).reshape(48, 64, 3)
    level1 = np.frombuffer(texture.read(level=1), dtype=np.uint8).reshape(24, 32, 3)
    level5 = np.frombuffer(texture.read(level=5), dtype=np.uint8)
    expected1 = avif._next_mip(level0)
    np.testing.assert_array_equal(level1, expected1)
    # 64x48 -> 7 levels down to 1x1
    assert len(level5) == 2 * 1 * 3  # level 5 is 2x1
    with pytest.raises(Exception):
        texture.read(level=7)


def test_no_mipmap(ctx, avif_rgb8):
    path, _ = avif_rgb8
    texture = ctx.texture_avif(path, mipmap=False)
    assert texture.read(level=0) is not None
    with pytest.raises(Exception):
        texture.read(level=1)


def test_upload_failure_releases_texture(ctx, avif_rgb8, monkeypatch):
    path, _ = avif_rgb8
    created = []
    original_texture = moderngl.Context.texture
    original_write = moderngl.Texture.write

    def spy_texture(self, *args, **kwargs):
        texture = original_texture(self, *args, **kwargs)
        created.append(texture)
        return texture

    def failing_write(self, data, viewport=None, level=0, alignment=1):
        if level == 1:
            raise RuntimeError("simulated upload failure")
        return original_write(self, data, viewport=viewport, level=level, alignment=alignment)

    monkeypatch.setattr(moderngl.Context, "texture", spy_texture)
    monkeypatch.setattr(moderngl.Texture, "write", failing_write)

    with pytest.raises(RuntimeError, match="simulated upload failure"):
        ctx.texture_avif(path)

    assert len(created) == 1
    assert isinstance(created[0].mglo, moderngl.InvalidObject)


def test_decode_failure_creates_no_texture(ctx, avif_corrupt, monkeypatch):
    created = []
    original_texture = moderngl.Context.texture

    def spy_texture(self, *args, **kwargs):
        texture = original_texture(self, *args, **kwargs)
        created.append(texture)
        return texture

    monkeypatch.setattr(moderngl.Context, "texture", spy_texture)
    with pytest.raises(avif.AvifError):
        ctx.texture_avif(avif_corrupt)
    assert created == []
