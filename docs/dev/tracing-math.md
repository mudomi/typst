# Tracing: the math behind drawing a pattern along a path

Typst's `tracing` stroke paint takes a small curve (the *pattern*) and lays
copies of it along another curve (the *skeleton*), bending the pattern so it
follows every turn of the path:

```typst
#let sine = curve(
  curve.cubic(
    (1pt * calc.pi, -calc.sqrt(3) * 2pt),
    (1pt * calc.pi, calc.sqrt(3) * 2pt),
    (2pt * calc.pi, 0pt)
  )
)

#curve(
  curve.line((80pt, 0pt)),
  stroke: tracing(sine, repeat: "stretch")
)
```

This is the same idea as Inkscape's *Pattern Along Path* live path effect.
The entire effect boils down to one deceptively short formula, described by
J.F. Barraud in the comment at the top of Inkscape's
[`lpe-patternalongpath.cpp`]:

> The basic deformation associated to B is then given by:
>
> ```
> (x, y) --> U(x) + y * N(x)
> ```
>
> (i.e. we go for distance x along the path, and then for distance y along
> the normal)

In words: treat every point of the pattern as an instruction. Its
x‑coordinate says *how far to walk along the skeleton*, and its y‑coordinate
says *how far to step sideways* once you get there. Everything else —
arc lengths, derivatives, normals, scaling — exists to make those two words,
"walk" and "sideways", mathematically precise.

This post walks through each step the way the implementation in
`crates/typst-library/src/visualize/tracing.rs` actually performs it. For the
underlying Bézier theory, [Pomax's *A Primer on Bézier Curves*][pomax] is the
canonical reference; the relevant sections are linked along the way.

[`lpe-patternalongpath.cpp`]: https://gitlab.com/inkscape/inkscape/-/blob/master/src/live_effects/lpe-patternalongpath.cpp
[pomax]: https://pomax.github.io/bezierinfo/


## Step 1 — The skeleton is a parametric curve

A Typst `curve` is a sequence of line segments and cubic Bézier segments. A
cubic Bézier with control points P₀, P₁, P₂, P₃ is the parametric function

```
B(t) = (1−t)³·P₀ + 3(1−t)²t·P₁ + 3(1−t)t²·P₂ + t³·P₃,   t ∈ [0, 1]
```

(see [the Bézier curve chapter][pomax-curves]). The crucial — and
inconvenient — property of this parametrization is that **t is not
distance**. Equal steps in t do not produce equal steps along the curve: the
point speeds up and slows down depending on where the control points sit.
If we placed pattern copies at equal steps of t, they would bunch up in
tight bends and stretch out on straightaways.

[pomax-curves]: https://pomax.github.io/bezierinfo/#whatis


## Step 2 — Re-parametrize by arc length

What we actually want is to address the skeleton by *distance from its
start*: a function U(s) that returns the point exactly s units of length
along the path. This is the *arc-length parametrization* (Pomax covers it
under [arc length][pomax-arclength] and [tracing a curve at fixed
distances][pomax-tracing]).

The arc length of a parametric curve from 0 to t is the integral of its
speed:

```
s(t) = ∫₀ᵗ |B′(u)| du
```

This integral has no closed form for cubic Béziers, so it must be computed
numerically. The implementation leans on the `kurbo` geometry library:

- `calculate_curve_length` walks the skeleton segment by segment. Line
  segments contribute their Euclidean length; each cubic segment contributes
  `CubicBez::arclen(tolerance)`, which evaluates the integral above by
  adaptive Gauss–Legendre quadrature to within a tolerance of `0.01`.
- The total L = s(1) is the length budget into which pattern copies must
  fit.

To go the *other* way — "give me the point at distance s" —
`evaluate_curve_at_length` walks the segments, accumulating their lengths
until it finds the segment containing the target distance. Within a line
segment the answer is simple linear interpolation. Within a cubic segment it
uses `CubicBez::inv_arclen(s, tolerance)`, the *inverse* arc-length query:
numerically solve s(t) = target for t, then evaluate B(t). This is exactly
the technique Pomax describes for tracing a curve at fixed distance
intervals.

Together these give us U(s): the skeleton, addressable by distance.

[pomax-arclength]: https://pomax.github.io/bezierinfo/#arclength
[pomax-tracing]: https://pomax.github.io/bezierinfo/#tracing


## Step 3 — Tangents and normals

Walking along the path covers the x of `U(x) + y·N(x)`. For the sideways
step we need N(s), a unit vector perpendicular to the path at distance s.

The derivative of a cubic Bézier is itself a (quadratic) Bézier
([Pomax: derivatives][pomax-derivatives]):

```
B′(t) = 3(1−t)²·(P₁−P₀) + 6(1−t)t·(P₂−P₁) + 3t²·(P₃−P₂)
```

B′(t) points along the direction of travel — it is the *tangent*. Normalize
it to unit length, then rotate it by 90° to get the *normal*
([Pomax: pointing normals][pomax-normals]). Rotating a 2D vector (x, y) by
90° is just a coordinate swap with a sign flip, no trigonometry needed:

```
T = B′(t) / |B′(t)|          (unit tangent)
N = (−T.y, T.x)              (unit normal)
```

That is what `point_and_normal_at_length` returns: the pair (U(s), N(s)).
For line segments the tangent is constant (the segment's direction vector);
for cubics it is the derivative evaluated at the t found by the inverse
arc-length query. One Typst-specific note: the y-axis points *down* in page
coordinates, so this particular rotation direction makes positive pattern-y
offsets land on the side users expect.

[pomax-derivatives]: https://pomax.github.io/bezierinfo/#derivatives
[pomax-normals]: https://pomax.github.io/bezierinfo/#pointvectors


## Step 4 — Put the pattern in "path coordinates"

The deformation formula reads the pattern's coordinates as (distance-along,
distance-across). For that to behave sensibly the pattern is first
*centered*: `center_pattern` computes the pattern's bounding box and
translates the pattern so the box's center sits on the origin.

After centering, a pattern of width w spans x ∈ [−w/2, +w/2] and its
y-values straddle zero. So y = 0 means "on the spine of the path", positive
and negative y mean the two sides of it — and a copy "placed at distance d"
means its *center* sits at distance d along the skeleton.


## Step 5 — How many copies, and how much to squeeze them

Before any deformation happens, `calculate_pattern_layout` decides how many
copies of the pattern fit and whether they need scaling. With skeleton
length L, pattern width w, and user spacing g, the number of copies is

```
n = max(1, ⌊(L + g) / (w + g)⌋)
```

(the `+ g` in the numerator works because n copies need n·w + (n−1)·g of
room — one less gap than copies). What happens to the leftover space depends
on the `repeat` mode:

- **`scale`** and **`stretch`** distribute the slack by scaling each copy so
  the n copies exactly fill the path:

  ```
  s_x = (L − (n−1)·g) / (n·w)
  ```

  `scale` applies s_x to both axes (the pattern keeps its aspect ratio);
  `stretch` applies it to x only (the pattern keeps its height). The factor
  is clamped below at 0.1 so degenerate inputs can't collapse the pattern.

- **`preserve`** keeps copies at their natural size; leftover length simply
  stays empty at the end.

- **`rotate`** and **`none`** don't deform at all — see step 7.

Copy i is then placed with its center at distance

```
d_i = start + s_x·w/2 + i·(s_x·w + g)
```

and a copy is only emitted if it fits entirely before the end of its
allotted range (with a small relative epsilon to absorb floating-point
noise).


## Step 6 — The deformation itself

Now the actual bending, in `deform_pattern_along_skeleton`. The pattern is
not transformed analytically; it is *sampled and re-plotted*:

1. **Sample the pattern uniformly by its own arc length.** The sample count
   adapts to pattern complexity (10 + 5 per segment, capped at 200). Using
   arc length rather than raw t keeps samples evenly spread over the
   pattern, for the same reason as in step 2.

2. **Map every sample through the formula.** For a sample (x, y), with the
   copy centered at distance d and scale factors s_x, s_y:

   ```
   s = d + x·s_x                      (where along the skeleton)
   P = U(s) + (y·s_y) · N(s)          (step sideways along the normal)
   ```

3. **Connect the mapped points into a polyline**, and re-close it if the
   original pattern was closed.

This is a piecewise-linear approximation of the exact deformed curve. The
exact image of a Bézier under this map is not a Bézier (Inkscape gets
exactness by composing piecewise polynomial "S-basis" functions from
lib2geom), but with enough samples the polyline is visually
indistinguishable and much simpler to implement.


## Step 7 — The rigid modes: `rotate` and `none`

For stamp-like placement, `transform_pattern_affine` skips the per-point
deformation and applies one affine transform to the whole copy — the
pattern's own cubic segments survive intact, control points and all:

- **`none`**: translate the copy to U(d). Done.
- **`rotate`**: additionally rotate it so its x-axis lines up with the
  tangent at d. The rotation angle is `atan2(T.y, T.x)` of the tangent —
  this is the classic "align a marker to a path" construction.

The difference from step 6 is *local vs. global*: deformation evaluates
U and N at a different s for every pattern point (so the pattern bends),
while the affine modes evaluate them once, at the copy's center (so the
pattern stays rigid).


## Step 8 — Start, middle, end

A tracing pattern can also be a dictionary with `start`, `repeat`, and `end`
entries, like SVG's marker-start/mid/end. The assembly in
`apply_tracing` (`crates/typst-layout/src/shapes.rs`) is bookkeeping on the
arc-length axis [0, L]:

1. The `start` pattern is placed once, centered at `start_width / 2` —
   flush with the beginning of the path.
2. The `end` pattern is placed once, centered at `L − end_width / 2` —
   flush with the end.
3. The `repeat` pattern fills whatever interval remains between them
   (shrunk by the spacing on each side), using steps 5–7.

Because everything is expressed in arc length, the three sections cannot
overlap regardless of how curly the skeleton is — lengths along the path
add up like distances on a straight line. That is the quiet payoff of the
re-parametrization in step 2: once the skeleton is addressed by distance,
laying out patterns on a wild Bézier path is exactly the same 1-D packing
problem as laying them out on a ruler.


## Summary

| Step | Question | Tool |
|------|----------|------|
| 1 | What is the path? | Piecewise lines + cubic Béziers B(t) |
| 2 | Where is "distance s" on it? | Arc length s(t) = ∫|B′| (Gauss–Legendre), inverted numerically |
| 3 | Which way is "sideways"? | Unit tangent B′/|B′| rotated 90° |
| 4 | What do pattern coords mean? | Center pattern: x = along, y = across |
| 5 | How many copies fit? | n = ⌊(L+g)/(w+g)⌋, scale s_x = (L−(n−1)g)/(n·w) |
| 6 | How does it bend? | Sample by arc length, map (x,y) ↦ U(d+x) + y·N(d+x) |
| 7 | How do rigid copies work? | One translate(+rotate) affine per copy |
| 8 | Start/middle/end? | 1-D interval packing on [0, L] |

One formula, `U(x) + y·N(x)`, and seven steps of scaffolding to earn it.
