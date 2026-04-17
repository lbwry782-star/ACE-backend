from pathlib import Path


def main() -> None:
    p = Path("engine/video_planning.py")
    lines = p.read_text(encoding="utf-8").splitlines(keepends=True)

    def find_line(sub: str, start: int = 0) -> int | None:
        for i in range(start, len(lines)):
            if sub in lines[i]:
                return i
        return None

    i0 = find_line("def _headline_derived_from_interaction")
    i1 = find_line("# SIDE_BY_SIDE_SHAPE_ENFORCEMENT: tall vertical-axis")
    if i0 is None or i1 is None:
        raise SystemExit(f"block markers missing i0={i0} i1={i1}")
    del lines[i0:i1]

    i2 = find_line("_SIDE_BY_SIDE_VERTICAL_OPENING_ENFORCEMENT = (")
    i3 = find_line("def _runway_vertical_axis_hard_constraints_english()")
    if i2 is None or i3 is None:
        raise SystemExit(f"vertical const markers missing i2={i2} i3={i3}")
    del lines[i2:i3]

    i4 = find_line("def video_plan_required_fields_for_runway")
    i5 = find_line("def _object_pair_digest", i4 or 0)
    if i4 is None or i5 is None:
        raise SystemExit(f"gate markers missing i4={i4} i5={i5}")
    del lines[i4:i5]

    i6 = find_line("def _fuzzy_replacement_direction")
    i7 = find_line("# snake_case / alternate keys", i6 or 0)
    if i6 is None or i7 is None:
        raise SystemExit(f"fuzzy markers missing i6={i6} i7={i7}")
    del lines[i6:i7]

    p.write_text("".join(lines), encoding="utf-8")
    print("pruned", p, "lines", len(lines))


if __name__ == "__main__":
    main()
