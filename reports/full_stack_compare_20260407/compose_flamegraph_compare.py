#!/usr/bin/env python3

import copy
import sys
import xml.etree.ElementTree as ET

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
FG_NS = "http://github.com/jonhoo/inferno"

ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)
ET.register_namespace("fg", FG_NS)


def q(tag: str) -> str:
    return f"{{{SVG_NS}}}{tag}"


def load_svg(path: str):
    root = ET.parse(path).getroot()
    width = int(float(root.attrib["width"]))
    height = int(float(root.attrib["height"]))
    return root, width, height


def add_text(parent, x, y, text, size=18, weight="bold", fill="rgb(0,0,0)"):
    node = ET.SubElement(
        parent,
        q("text"),
        {
            "x": str(x),
            "y": str(y),
            "fill": fill,
            "style": f"font-family:monospace;font-size:{size}px;font-weight:{weight}",
        },
    )
    node.text = text
    return node


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: compose_flamegraph_compare.py VLLM.svg RVLLM.svg OUT.svg", file=sys.stderr)
        return 2

    v_path, r_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    v_root, width, v_h = load_svg(v_path)
    r_root, r_w, r_h = load_svg(r_path)
    if r_w != width:
        raise SystemExit(f"source widths differ: {width} vs {r_w}")

    top_pad = 72
    gap = 44
    bottom_pad = 24
    total_h = top_pad + v_h + gap + r_h + bottom_pad

    out = ET.Element(
        q("svg"),
        {
            "version": "1.1",
            "width": str(width),
            "height": str(total_h),
            "viewBox": f"0 0 {width} {total_h}",
        },
    )

    ET.SubElement(
        out,
        q("rect"),
        {
            "x": "0",
            "y": "0",
            "width": str(width),
            "height": str(total_h),
            "fill": "rgb(255,255,255)",
        },
    )

    add_text(out, 20, 28, "N=64 CPU Flamegraph Comparison", size=24)
    add_text(out, 20, 50, "Same native flamegraph format, stacked for direct comparison", size=12, weight="normal", fill="rgb(80,80,80)")

    v_wrap = ET.SubElement(out, q("g"), {"transform": f"translate(0,{top_pad})"})
    add_text(v_wrap, 20, 22, "vLLM", size=18)
    add_text(v_wrap, 120, 22, "Python wall-time flamegraph", size=12, weight="normal", fill="rgb(80,80,80)")
    v_svg = copy.deepcopy(v_root)
    v_svg.attrib["x"] = "0"
    v_svg.attrib["y"] = "32"
    v_svg.attrib.pop("width", None)
    v_svg.attrib.pop("height", None)
    v_wrap.append(v_svg)

    r_offset = top_pad + v_h + gap
    r_wrap = ET.SubElement(out, q("g"), {"transform": f"translate(0,{r_offset})"})
    add_text(r_wrap, 20, 22, "rvLLM", size=18)
    add_text(r_wrap, 120, 22, "Rust CPU flamegraph for the exact H100 N=64 benchmark run", size=12, weight="normal", fill="rgb(80,80,80)")
    r_svg = copy.deepcopy(r_root)
    r_svg.attrib["x"] = "0"
    r_svg.attrib["y"] = "32"
    r_svg.attrib.pop("width", None)
    r_svg.attrib.pop("height", None)
    r_wrap.append(r_svg)

    ET.ElementTree(out).write(out_path, encoding="utf-8", xml_declaration=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
