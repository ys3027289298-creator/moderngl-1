"""AVIF texture loading for ModernGL.

Decodes AVIF images through libavif (loaded with ctypes, no Python package
dependency) and uploads them as 2D textures with a full mip chain.

* 8-bit images become ``dtype="f1"`` textures.
* 10-bit images become ``dtype="f2"`` (half float) textures.
* Alpha is detected from the bitstream (3 or 4 components).
* Anything else (12-bit, exotic profiles) raises :class:`AvifUnsupportedError`.

The encoded file is never fully copied into memory: files are read through
``avifIOCreateFileReader`` which pulls extents from disk on demand, and the
decoded pixel buffer is uploaded in place (zero copy) level by level.
"""

import ctypes
import ctypes.util
import os

__all__ = [
    "AvifError",
    "AvifUnsupportedError",
    "AvifImage",
    "decode_avif",
    "texture_from_avif",
    "libavif_available",
]


class AvifError(Exception):
    """An AVIF file could not be decoded or libavif is unavailable."""


class AvifUnsupportedError(AvifError):
    """The AVIF file uses a profile that is not supported (bit depth, ...)."""


class _AvifRWData(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("size", ctypes.c_size_t)]


class _AvifImage(ctypes.Structure):
    # Mirrors avifImage from libavif 1.0.x (avif/avif.h).
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("depth", ctypes.c_uint32),
        ("yuvFormat", ctypes.c_int),
        ("yuvRange", ctypes.c_int),
        ("yuvChromaSamplePosition", ctypes.c_int),
        ("yuvPlanes", ctypes.c_void_p * 3),
        ("yuvRowBytes", ctypes.c_uint32 * 3),
        ("imageOwnsYUVPlanes", ctypes.c_int),
        ("alphaPlane", ctypes.c_void_p),
        ("alphaRowBytes", ctypes.c_uint32),
        ("imageOwnsAlphaPlane", ctypes.c_int),
        ("alphaPremultiplied", ctypes.c_int),
        ("icc", _AvifRWData),
        ("colorPrimaries", ctypes.c_uint16),
        ("transferCharacteristics", ctypes.c_uint16),
        ("matrixCoefficients", ctypes.c_uint16),
        ("clli", ctypes.c_uint16 * 2),
        ("transformFlags", ctypes.c_uint32),
        ("pasp", ctypes.c_uint32 * 4),
        ("clap", ctypes.c_uint32 * 4),
        ("irot", ctypes.c_uint8),
        ("imir", ctypes.c_uint8),
        ("exif", _AvifRWData),
        ("xmp", _AvifRWData),
    ]


class _AvifRGBImage(ctypes.Structure):
    # Mirrors avifRGBImage from libavif 1.0.x (avif/avif.h).
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("depth", ctypes.c_uint32),
        ("format", ctypes.c_int),
        ("chromaUpsampling", ctypes.c_int),
        ("chromaDownsampling", ctypes.c_int),
        ("avoidLibYUV", ctypes.c_int),
        ("ignoreAlpha", ctypes.c_int),
        ("alphaPremultiplied", ctypes.c_int),
        ("isFloat", ctypes.c_int),
        ("maxThreads", ctypes.c_int),
        ("pixels", ctypes.c_void_p),
        ("rowBytes", ctypes.c_uint32),
    ]


class _AvifDecoderHeader(ctypes.Structure):
    # Only the leading input fields of avifDecoder; the rest is opaque to us.
    _fields_ = [("codecChoice", ctypes.c_int), ("maxThreads", ctypes.c_int)]


_AVIF_RGB_FORMAT_RGB = 0
_AVIF_RGB_FORMAT_RGBA = 1
_AVIF_PIXEL_FORMATS = {0: "yuv444", 1: "yuv422", 2: "yuv420", 3: "yuv400"}

_lib = None
_lib_error = None


def _load_libavif():
    global _lib, _lib_error
    if _lib is not None or _lib_error is not None:
        return _lib
    candidates = [
        os.environ.get("MODERNGL_AVIF_LIBRARY"),
        ctypes.util.find_library("avif"),
        "libavif.so.16",
        "libavif.so",
        "libavif.16.dylib",
        "libavif.dylib",
        "avif.dll",
        "libavif.dll",
    ]
    errors = []
    for name in candidates:
        if not name:
            continue
        try:
            lib = ctypes.CDLL(name)
        except OSError as exc:
            errors.append("%s: %s" % (name, exc))
            continue
        lib.avifDecoderCreate.restype = ctypes.c_void_p
        lib.avifDecoderDestroy.argtypes = [ctypes.c_void_p]
        lib.avifDecoderSetIO.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.avifDecoderSetIOMemory.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        lib.avifDecoderRead.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.avifIOCreateFileReader.restype = ctypes.c_void_p
        lib.avifIOCreateFileReader.argtypes = [ctypes.c_char_p]
        lib.avifImageCreateEmpty.restype = ctypes.c_void_p
        lib.avifImageDestroy.argtypes = [ctypes.c_void_p]
        lib.avifRGBImageSetDefaults.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.avifRGBImageAllocatePixels.argtypes = [ctypes.c_void_p]
        lib.avifRGBImageFreePixels.argtypes = [ctypes.c_void_p]
        lib.avifImageYUVToRGB.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.avifResultToString.restype = ctypes.c_char_p
        _lib = lib
        return _lib
    _lib_error = AvifError(
        "libavif could not be loaded (tried: %s). "
        "Install libavif (e.g. apt install libavif16) or set MODERNGL_AVIF_LIBRARY."
        % "; ".join(errors or ["no candidates"])
    )
    return None


def libavif_available():
    """Return True if libavif can be loaded on this system."""
    return _load_libavif() is not None


def _require_lib():
    lib = _load_libavif()
    if lib is None:
        raise _lib_error
    return lib


def _result_string(lib, result):
    return lib.avifResultToString(result).decode("ascii", "replace")


class AvifImage:
    """A decoded AVIF image. Owns the libavif objects until :meth:`close`."""

    def __init__(self):
        self.width = 0
        self.height = 0
        self.depth = 0
        self.has_alpha = False
        self.components = 0
        self.dtype = None
        self.pixels = None  # memoryview over the libavif pixel buffer
        self._decoder = None
        self._image = None
        self._rgb = None
        self._lib = None

    @property
    def size(self):
        return self.width, self.height

    def close(self):
        lib = self._lib
        if lib is None:
            return
        if self._rgb is not None:
            lib.avifRGBImageFreePixels(ctypes.byref(self._rgb))
            self._rgb = None
        if self._image is not None:
            lib.avifImageDestroy(self._image)
            self._image = None
        if self._decoder is not None:
            lib.avifDecoderDestroy(self._decoder)
            self._decoder = None
        self.pixels = None
        self._lib = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def decode_avif(source, max_threads=None):
    """Decode an AVIF image from a file path or a bytes-like object.

    Returns an :class:`AvifImage`. The caller must call ``close()`` on it
    (or use it as a context manager) to release the libavif objects.
    """
    lib = _require_lib()
    if isinstance(source, (str, os.PathLike)):
        raw = None
        path = os.fspath(source)
    else:
        raw = source if isinstance(source, bytes) else bytes(source)
        path = None

    decoded = AvifImage()
    decoded._lib = lib
    try:
        decoder = lib.avifDecoderCreate()
        if not decoder:
            raise AvifError("avifDecoderCreate failed")
        decoded._decoder = decoder
        header = ctypes.cast(decoder, ctypes.POINTER(_AvifDecoderHeader))
        header.contents.maxThreads = max_threads or (os.cpu_count() or 1)

        if path is not None:
            io = lib.avifIOCreateFileReader(os.fsencode(path))
            if not io:
                raise AvifError("cannot open AVIF file: %s" % path)
            lib.avifDecoderSetIO(decoder, io)  # the decoder takes ownership
        else:
            result = lib.avifDecoderSetIOMemory(decoder, raw, len(raw))
            if result != 0:
                raise AvifError("avifDecoderSetIOMemory failed: %s" % _result_string(lib, result))

        image = lib.avifImageCreateEmpty()
        if not image:
            raise AvifError("avifImageCreateEmpty failed")
        decoded._image = image

        result = lib.avifDecoderRead(decoder, image)
        if result != 0:
            raise AvifError(
                "failed to decode AVIF %s: %s"
                % (path or "from memory", _result_string(lib, result))
            )

        info = ctypes.cast(image, ctypes.POINTER(_AvifImage)).contents
        if info.depth not in (8, 10):
            raise AvifUnsupportedError(
                "unsupported AVIF bit depth: %d (only 8-bit and 10-bit are supported)"
                % info.depth
            )
        if info.yuvFormat not in _AVIF_PIXEL_FORMATS:
            raise AvifUnsupportedError(
                "unsupported AVIF pixel format: %d" % info.yuvFormat
            )

        has_alpha = bool(info.alphaPlane)
        rgb = _AvifRGBImage()
        lib.avifRGBImageSetDefaults(ctypes.byref(rgb), image)
        rgb.format = _AVIF_RGB_FORMAT_RGBA if has_alpha else _AVIF_RGB_FORMAT_RGB
        rgb.maxThreads = header.contents.maxThreads
        if info.depth > 8:
            rgb.depth = 16
            rgb.isFloat = 1  # let libavif emit half float directly
        result = lib.avifRGBImageAllocatePixels(ctypes.byref(rgb))
        if result != 0:
            raise AvifError("avifRGBImageAllocatePixels failed: %s" % _result_string(lib, result))
        decoded._rgb = rgb  # owns rgb.pixels from here on

        result = lib.avifImageYUVToRGB(image, ctypes.byref(rgb))
        if result != 0:
            raise AvifError("avifImageYUVToRGB failed: %s" % _result_string(lib, result))

        components = 4 if has_alpha else 3
        nbytes = rgb.rowBytes * info.height
        decoded.width = info.width
        decoded.height = info.height
        decoded.depth = info.depth
        decoded.has_alpha = has_alpha
        decoded.components = components
        decoded.dtype = "f1" if info.depth == 8 else "f2"
        decoded.pixels = memoryview(
            (ctypes.c_char * nbytes).from_address(rgb.pixels)
        )
        return decoded
    except Exception:
        decoded.close()
        raise


def _alignment(row_bytes):
    if row_bytes % 4 == 0:
        return 4
    if row_bytes % 2 == 0:
        return 2
    return 1


def _mip_levels(width, height):
    levels = 1
    size = max(width, height)
    while size > 1:
        size //= 2
        levels += 1
    return levels


def _next_mip(array):
    """Box-filter 2x downsample of a (height, width, components) array."""
    import numpy as np

    height, width, components = array.shape
    out_height = max(1, height // 2)
    out_width = max(1, width // 2)
    factor_h = height // out_height
    factor_w = width // out_width
    cropped = array[: out_height * factor_h, : out_width * factor_w]
    blocks = cropped.reshape(out_height, factor_h, out_width, factor_w, components)
    if array.dtype == np.uint8:
        summed = blocks.sum(axis=(1, 3), dtype=np.uint16)
        np.add(summed, (factor_h * factor_w) // 2, out=summed)
        np.floor_divide(summed, factor_h * factor_w, out=summed)
        return summed.astype(np.uint8)
    summed = blocks.sum(axis=(1, 3), dtype=np.float32)
    summed *= 1.0 / (factor_h * factor_w)
    return summed.astype(array.dtype)


def texture_from_avif(ctx, source, mipmap=True, max_threads=None):
    """Decode an AVIF image and upload it to a new 2D texture.

    The mip chain is generated on the CPU and uploaded level by level.
    The decoded pixel buffer is released as soon as the base level and the
    first mip are uploaded, so the peak stays close to a single full-size
    pixel buffer plus the GL mip chain.
    On any decode or upload failure every allocated object is released,
    so no half-built texture is left behind.
    """
    import numpy as np

    decoded = decode_avif(source, max_threads=max_threads)
    try:
        dtype_size = 1 if decoded.dtype == "f1" else 2
        row_bytes = decoded.width * decoded.components * dtype_size
        texture = None
        try:
            texture = ctx.texture(
                decoded.size,
                decoded.components,
                decoded.pixels,
                alignment=_alignment(row_bytes),
                dtype=decoded.dtype,
            )
            levels = _mip_levels(decoded.width, decoded.height) if mipmap else 1
            if levels > 1:
                np_dtype = np.uint8 if decoded.dtype == "f1" else np.float16
                plane = np.frombuffer(decoded.pixels, dtype=np_dtype).reshape(
                    decoded.height, decoded.width, decoded.components
                )
                next_plane = _next_mip(plane)
                del plane  # last use of the decoded pixel buffer
                width, height, components = decoded.width, decoded.height, decoded.components
                decoded.close()
                decoded = None
                texture.build_mipmaps(0, levels - 1)
                texture.write(
                    next_plane,
                    level=1,
                    alignment=_alignment(next_plane.shape[1] * components * dtype_size),
                )
                plane = next_plane
                for level in range(2, levels):
                    plane = _next_mip(plane)
                    texture.write(
                        plane,
                        level=level,
                        alignment=_alignment(plane.shape[1] * components * dtype_size),
                    )
            return texture
        except Exception:
            if texture is not None:
                texture.release()
            raise
    finally:
        if decoded is not None:
            decoded.close()
