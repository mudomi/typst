use kurbo::{
    Affine, ParamCurve, ParamCurveArclen, ParamCurveDeriv, PathEl, PathSeg, Vec2,
};

use crate::diag::HintedStrResult;
use crate::foundations::{Cast, Content, Dict, Smart, cast, dict, elem};
use crate::layout::{Abs, Angle, Length, Point};
use crate::visualize::{Curve, CurveItem, Paint, Stroke};

/// Deformation samples grow with pattern complexity, within fixed bounds.
const BASE_SAMPLE_COUNT: usize = 10;
const SAMPLES_PER_SEGMENT: usize = 5;
const MAX_SAMPLE_COUNT: usize = 200;

/// Accuracy of kurbo's arc length computations, in raw units.
const BEZIER_TOLERANCE: f64 = 0.01;
/// Keeps degenerate layouts from collapsing pattern copies entirely.
const MIN_SCALE_FACTOR: f64 = 0.1;
/// Relative tolerance for boundary comparisons on the arc length axis.
const RELATIVE_EPSILON: f64 = 1e-6;

/// Repeats a pattern curve along a skeleton curve.
///
/// The pattern is bent to follow the skeleton: a pattern point's x-coordinate
/// determines how far to travel along the skeleton and its y-coordinate how
/// far to step sideways along the normal at that position.
///
/// # Example
/// ```example
/// #let sine = curve(
///   curve.cubic(
///     (1pt * calc.pi, -calc.sqrt(3) * 2pt),
///     (1pt * calc.pi, calc.sqrt(3) * 2pt),
///     (2pt * calc.pi, 0pt),
///   )
/// )
///
/// #tracing(
///   curve(curve.line((80pt, 0pt))),
///   sine,
///   repeat: "stretch",
/// )
/// ```
#[elem]
pub struct TracingElem {
    /// The curve to lay the pattern along.
    #[required]
    pub skeleton: Content,

    /// The pattern to trace along the skeleton: either a curve element that
    /// is repeated along the full path, or a dictionary with optional
    /// `start`, `repeat`, and `end` keys, each containing a curve element.
    #[required]
    pub pattern: PatternSpec,

    /// How to repeat the pattern.
    pub repeat: RepeatType,

    /// Space between pattern copies.
    pub spacing: Length,

    /// How to fill the traced pattern.
    pub fill: Option<Paint>,

    /// How to @stroke[stroke] the traced pattern.
    ///
    /// Can be set to `{none}` to disable the stroke or to `{auto}` for a
    /// stroke of `{1pt}` black if and only if no fill is given.
    #[fold]
    pub stroke: Smart<Option<Stroke>>,
}

/// How copies of the pattern are fitted onto the skeleton.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash, Cast)]
pub enum RepeatType {
    /// Scale uniformly to fill the path.
    #[default]
    Scale,
    /// Stretch horizontally along the path.
    Stretch,
    /// Keep original size.
    Preserve,
    /// Rotate to follow the path.
    Rotate,
    /// No transformation.
    None,
}

/// Pattern specification with optional start, repeat, and end curves.
#[derive(Debug, Clone, PartialEq, Hash)]
pub struct PatternSpec {
    pub start: Option<Content>,
    pub repeat: Option<Content>,
    pub end: Option<Content>,
}

impl Eq for PatternSpec {}

cast! {
    PatternSpec,
    self => match self {
        Self { start: None, repeat: Some(repeat), end: None } => repeat.into_value(),
        Self { start, repeat, end } => {
            dict! { "start" => start, "repeat" => repeat, "end" => end }.into_value()
        }
    },
    repeat: Content => Self { start: None, repeat: Some(repeat), end: None },
    mut dict: Dict => {
        let mut take = |key| -> HintedStrResult<Option<Content>> {
            dict.take(key).ok().map(|v| v.cast()).transpose()
        };
        let (start, repeat, end) = (take("start")?, take("repeat")?, take("end")?);
        dict.finish(&["start", "repeat", "end"])?;
        Self { start, repeat, end }
    },
}

/// Lay out repeated copies of the pattern between the arc lengths
/// `start_offset` and `end_offset` along the skeleton.
pub fn apply_tracing_in_range(
    repeat_type: RepeatType,
    pattern: &Curve,
    skeleton: &Curve,
    spacing: f64,
    skeleton_length: f64,
    start_offset: f64,
    end_offset: f64,
) -> Curve {
    let mut result = Curve::new();

    let centered = center_pattern(pattern);
    let pattern_width = centered.bbox(None).size().x.to_raw();
    let available_length = end_offset - start_offset;
    if pattern_width <= 0.0 || skeleton_length <= 0.0 || available_length <= 0.0 {
        return result;
    }

    let segments = to_segments(skeleton);
    let (num_copies, x_scale) =
        calculate_pattern_layout(repeat_type, available_length, pattern_width, spacing);

    let scaled_width = pattern_width * x_scale;
    let boundary_epsilon = skeleton_length * RELATIVE_EPSILON;
    let mut center = start_offset + scaled_width / 2.0;

    for _ in 0..num_copies {
        if center + scaled_width / 2.0 <= end_offset + boundary_epsilon {
            let copy = place_copy(
                repeat_type,
                &centered,
                &segments,
                center,
                x_scale,
                skeleton_length,
            );
            result.0.extend(copy.0);
        }
        center += scaled_width + spacing;
    }

    result
}

/// Place a single, unscaled copy of the pattern centered at the arc length
/// `offset` along the skeleton.
pub fn apply_pattern_once(
    pattern: &Curve,
    skeleton: &Curve,
    skeleton_length: f64,
    offset: f64,
) -> Curve {
    let centered = center_pattern(pattern);
    if centered.bbox(None).size().x.to_raw() <= 0.0 || skeleton_length <= 0.0 {
        return Curve::new();
    }

    let segments = to_segments(skeleton);
    place_copy(RepeatType::Preserve, &centered, &segments, offset, 1.0, skeleton_length)
}

/// Place one copy of the (centered) pattern at the arc length `center`.
fn place_copy(
    repeat: RepeatType,
    pattern: &Curve,
    segments: &[PathSeg],
    center: f64,
    x_scale: f64,
    skeleton_length: f64,
) -> Curve {
    match repeat {
        RepeatType::None | RepeatType::Rotate => transform_pattern_affine(
            pattern,
            segments,
            center,
            skeleton_length,
            repeat == RepeatType::Rotate,
        ),
        RepeatType::Scale => deform_pattern_along_skeleton(
            pattern,
            segments,
            center,
            x_scale,
            x_scale,
            skeleton_length,
        ),
        RepeatType::Stretch => deform_pattern_along_skeleton(
            pattern,
            segments,
            center,
            x_scale,
            1.0,
            skeleton_length,
        ),
        RepeatType::Preserve => deform_pattern_along_skeleton(
            pattern,
            segments,
            center,
            1.0,
            1.0,
            skeleton_length,
        ),
    }
}

/// Map pattern points along the skeleton: a point's x picks the position
/// along the path, its y the offset along the normal at that position.
fn deform_pattern_along_skeleton(
    pattern: &Curve,
    segments: &[PathSeg],
    center: f64,
    x_scale: f64,
    y_scale: f64,
    skeleton_length: f64,
) -> Curve {
    let mut result = Curve::new();
    let samples = sample_pattern_by_arc_length(pattern, adaptive_sample_count(pattern));
    let is_closed = pattern.0.iter().any(|item| matches!(item, CurveItem::Close));

    let mut is_first_point = true;
    for (x, y) in samples {
        let arc_length = center + x * x_scale;
        if !(0.0..=skeleton_length).contains(&arc_length) {
            continue;
        }
        let Some((pos, normal)) = point_and_normal_at_length(segments, arc_length) else {
            continue;
        };

        let offset = y * y_scale;
        let point = Point::new(
            pos.x + Abs::raw(normal.x * offset),
            pos.y + Abs::raw(normal.y * offset),
        );
        if is_first_point {
            result.move_(point);
            is_first_point = false;
        } else {
            result.line(point);
        }
    }

    if is_closed && !is_first_point {
        result.close();
    }

    result
}

/// Place the pattern rigidly: translate it to the skeleton point at `center`
/// and optionally rotate it to align with the tangent there. The pattern's
/// cubic segments survive intact.
fn transform_pattern_affine(
    pattern: &Curve,
    segments: &[PathSeg],
    center: f64,
    skeleton_length: f64,
    apply_rotation: bool,
) -> Curve {
    let mut result = Curve::new();

    if !(0.0..=skeleton_length).contains(&center) {
        return result;
    }
    let Some((pos, tangent)) = evaluate_at_length(segments, center) else {
        return result;
    };

    let mut transform = Affine::translate(pos.to_vec2());
    if apply_rotation {
        transform *= Affine::rotate(Angle::atan2(tangent.y, tangent.x).to_rad());
    }

    let apply = |p: Point| {
        let q = transform * kurbo::Point::new(p.x.to_raw(), p.y.to_raw());
        Point::new(Abs::raw(q.x), Abs::raw(q.y))
    };

    for item in &pattern.0 {
        match item {
            CurveItem::Move(p) => result.move_(apply(*p)),
            CurveItem::Line(p) => result.line(apply(*p)),
            CurveItem::Cubic(c1, c2, p) => {
                result.cubic(apply(*c1), apply(*c2), apply(*p))
            }
            CurveItem::Close => result.close(),
        }
    }

    result
}

/// How many copies fit into `available_length`, and the x-scale applied to
/// each. The `+ spacing` in the numerator accounts for n copies needing only
/// n - 1 gaps between them.
fn calculate_pattern_layout(
    repeat_type: RepeatType,
    available_length: f64,
    pattern_width: f64,
    spacing: f64,
) -> (usize, f64) {
    let effective_width = pattern_width + spacing;
    if effective_width <= 0.0 {
        return (1, 1.0);
    }

    let num_copies =
        ((available_length + spacing) / effective_width).floor().max(1.0) as usize;

    let x_scale = match repeat_type {
        RepeatType::Scale | RepeatType::Stretch => {
            let total_gap = (num_copies - 1) as f64 * spacing;
            let scale =
                (available_length - total_gap) / (num_copies as f64 * pattern_width);
            scale.max(MIN_SCALE_FACTOR)
        }
        RepeatType::Preserve | RepeatType::Rotate | RepeatType::None => 1.0,
    };

    (num_copies, x_scale)
}

/// The total arc length of a curve.
pub fn calculate_curve_length(curve: &Curve) -> f64 {
    to_segments(curve)
        .iter()
        .map(|seg| seg.arclen(BEZIER_TOLERANCE))
        .sum()
}

/// The curve as kurbo segments. Curves may begin without a move, in which
/// case they implicitly start at the origin.
fn to_segments(curve: &Curve) -> Vec<PathSeg> {
    let mut elements: Vec<PathEl> = curve.to_kurbo().collect();
    if !matches!(elements.first(), None | Some(PathEl::MoveTo(_))) {
        elements.insert(0, PathEl::MoveTo(kurbo::Point::ZERO));
    }
    kurbo::segments(elements).collect()
}

/// Position and tangent at `target` arc length along the segments. Targets
/// past the end clamp to the final endpoint.
fn evaluate_at_length(segments: &[PathSeg], target: f64) -> Option<(kurbo::Point, Vec2)> {
    let mut accumulated = 0.0;
    for seg in segments {
        let length = seg.arclen(BEZIER_TOLERANCE);
        if accumulated + length >= target {
            let t = seg.inv_arclen(target - accumulated, BEZIER_TOLERANCE);
            return Some((seg.eval(t), tangent(seg, t)));
        }
        accumulated += length;
    }
    segments.last().map(|seg| (seg.eval(1.0), tangent(seg, 1.0)))
}

fn tangent(seg: &PathSeg, t: f64) -> Vec2 {
    match seg {
        PathSeg::Line(line) => line.p1 - line.p0,
        PathSeg::Quad(quad) => quad.deriv().eval(t).to_vec2(),
        PathSeg::Cubic(cubic) => cubic.deriv().eval(t).to_vec2(),
    }
}

/// Position and unit normal (the tangent rotated by 90°) at `target` arc
/// length along the segments.
fn point_and_normal_at_length(
    segments: &[PathSeg],
    target: f64,
) -> Option<(Point, Vec2)> {
    let (pos, tangent) = evaluate_at_length(segments, target)?;
    let length = tangent.hypot();
    let normal = if length > 0.0 {
        Vec2::new(-tangent.y / length, tangent.x / length)
    } else {
        Vec2::new(0.0, 1.0)
    };
    Some((Point::new(Abs::raw(pos.x), Abs::raw(pos.y)), normal))
}

/// Sample the pattern at uniform arc length intervals, as raw (x, y) pairs.
fn sample_pattern_by_arc_length(pattern: &Curve, num_samples: usize) -> Vec<(f64, f64)> {
    let segments = to_segments(pattern);
    let total_length: f64 = segments.iter().map(|seg| seg.arclen(BEZIER_TOLERANCE)).sum();
    if total_length <= 0.0 || num_samples == 0 {
        return Vec::new();
    }

    (0..=num_samples)
        .filter_map(|i| {
            let target = i as f64 / num_samples as f64 * total_length;
            evaluate_at_length(&segments, target).map(|(p, _)| (p.x, p.y))
        })
        .collect()
}

fn adaptive_sample_count(curve: &Curve) -> usize {
    (BASE_SAMPLE_COUNT + curve.0.len() * SAMPLES_PER_SEGMENT).min(MAX_SAMPLE_COUNT)
}

/// Shift the pattern so its bounding box is centered on the origin. Its x
/// coordinates then read as "distance along the path" relative to the copy's
/// center and its y coordinates as "distance across".
fn center_pattern(pattern: &Curve) -> Curve {
    let mut centered = Curve(pattern.0.clone());
    if !pattern.is_empty() {
        let bbox = pattern.bbox(None);
        centered.translate(-Point::new(
            (bbox.min.x + bbox.max.x) / 2.0,
            (bbox.min.y + bbox.max.y) / 2.0,
        ));
    }
    centered
}
