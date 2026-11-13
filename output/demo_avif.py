#!/usr/bin/env python3
"""Titled demo of AVIF texture loading (recorded to output/avif_demo.cast)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")


def title(text):
    line = "=" * 64
    print("\n" + line)
    print("  " + text)
    print(line, flush=True)


def main():
    import moderngl
    from moderngl import avif
    from glcontext import egl

    title("[1/5] 解码识别: 8/10 位与 alpha 检测")
    for name in ("4K_8bit_rgba_q90.avif", "4K_10bit_rgba_q90.avif"):
        with avif.decode_avif(os.path.join(FIXTURES, name)) as img:
            print(
                f"  {name}: {img.width}x{img.height} depth={img.depth}bit "
                f"alpha={img.has_alpha} -> components={img.components} dtype={img.dtype}"
            )

    title("[2/5] 纹理创建与 mip 链逐级上传")
    ctx = moderngl.create_context(
        standalone=True,
        context=egl.create_context(glversion=330, mode="standalone"),
    )
    texture = ctx.texture_avif(os.path.join(FIXTURES, "4K_8bit_rgba_q90.avif"))
    print(f"  texture: size={texture.size} components={texture.components} dtype={texture.dtype}")
    width, height = texture.size
    level = 0
    while True:
        data = texture.read(level=level)
        print(f"  mip level {level:2d}: {width}x{height} uploaded, read back {len(data)} bytes")
        if width == 1 and height == 1:
            break
        width, height = max(1, width // 2), max(1, height // 2)
        level += 1
    texture.release()

    title("[3/5] 不支持的 profile 明确报错")
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".avif", delete=False) as tmp:
        path12 = tmp.name
    subprocess.run(
        "ffmpeg -hide_banner -loglevel error -y -f lavfi -i testsrc2=size=64x64 "
        "-frames:v 1 -pix_fmt yuv420p12le -strict experimental " + path12,
        shell=True, check=True,
    )
    try:
        ctx.texture_avif(path12)
    except avif.AvifUnsupportedError as exc:
        print(f"  12-bit AVIF -> AvifUnsupportedError: {exc}")
    os.unlink(path12)
    try:
        ctx.texture_avif(os.path.join(FIXTURES, "corrupt.avif"))
    except avif.AvifError as exc:
        print(f"  corrupt file -> AvifError: {exc}")

    title("[4/5] 上传失败回收: 不留半成品纹理")
    original_write = moderngl.Texture.write
    created = []
    original_texture = moderngl.Context.texture

    def spy_texture(self, *args, **kwargs):
        tex = original_texture(self, *args, **kwargs)
        created.append(tex)
        return tex

    def failing_write(self, data, viewport=None, level=0, alignment=1):
        if level == 1:
            raise RuntimeError("simulated upload failure")
        return original_write(self, data, viewport=viewport, level=level, alignment=alignment)

    moderngl.Context.texture = spy_texture
    moderngl.Texture.write = failing_write
    try:
        ctx.texture_avif(os.path.join(FIXTURES, "4K_8bit_rgba_q90.avif"))
    except RuntimeError as exc:
        print(f"  upload failed as simulated: {exc}")
    finally:
        moderngl.Context.texture = original_texture
        moderngl.Texture.write = original_write
    print(f"  textures created before failure: {len(created)}")
    print(f"  released (InvalidObject): {all(isinstance(t.mglo, moderngl.InvalidObject) for t in created)}")

    title("[5/5] 4K/8K 内存峰值 (详见 output/memory_peak.md)")
    with open(os.path.join(HERE, "memory_peak.md")) as fh:
        for line in fh:
            if line.startswith("|"):
                print("  " + line.rstrip())
    print("\ndemo done.")


if __name__ == "__main__":
    sys.exit(main())
