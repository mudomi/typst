use std::sync::Arc;

use ecow::EcoString;
use kurbo::{Affine, CubicBez, ParamCurve, ParamCurveArclen, ParamCurveDeriv};
use typst_syntax::Span;

use crate::diag::{At, SourceResult, HintedStrResult};
use crate::foundations::{
    cast, func, scope, ty, CastInfo, Content, Dict, FromValue, IntoValue, Reflect, Repr, Type, Value,
};
use crate::layout::{Abs, Length, Point};
use crate::visualize::Curve;

const BASE_SAMPLE_COUNT: usize = 10;
const SAMPLES_PER_SEGMENT: usize = 5;
const MAX_SAMPLE_COUNT: usize = 200;

const BEZIER_TOLERANCE: f64 = 0.01;
const MIN_SCALE_FACTOR: f64 = 0.1;
const RELATIVE_EPSILON: f64 = 1e-6;

fn adaptive_sample_count(curve: &Curve) -> usize {
    let segment_count = curve.0.len();
    (BASE_SAMPLE_COUNT + segment_count * SAMPLES_PER_SEGMENT).min(MAX_SAMPLE_COUNT)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RepeatType {
    /// Scale uniformly to fill the path.
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

impl Default for RepeatType {
    fn default() -> Self {
        Self::Scale
    }
}

/// Pattern specification with optional start, repeat, and end curves.
#[derive(Debug, Clone, PartialEq, Hash)]
pub struct PatternSpec {
    pub start: Option<Content>,
    pub repeat: Option<Content>,
    pub end: Option<Content>,
}

impl Eq for PatternSpec {}

impl PatternSpec {
    pub fn from_repeat(content: Content) -> Self {
        Self {
            start: None,
            repeat: Some(content),
            end: None,
        }
    }
}

/// A tracing that applies a pattern along a stroke path.
///
/// # Examples
/// ```example
/// #let sine = curve(
///   curve.cubic(
///     (1pt * calc.pi, -calc.sqrt(3) * 2pt),
///     (1pt * calc.pi, calc.sqrt(3) * 2pt),
///     (2pt * calc.pi, 0pt)
///   )
/// )
///
/// #curve(
///   curve.line((80pt, 0pt)),
///   stroke: tracing(sine, repeat: "stretch")
/// )
/// ```
#[ty(scope, cast)]
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Tracing(Arc<TracingRepr>);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TracingRepr {
    pattern: PatternSpec,
    repeat_type: RepeatType,
    spacing: Length,
}

impl Reflect for Tracing {
    fn input() -> CastInfo {
        CastInfo::Type(Type::of::<Self>())
    }

    fn output() -> CastInfo {
        CastInfo::Type(Type::of::<Self>())
    }

    fn castable(value: &Value) -> bool {
        matches!(value, Value::Tracing(_))
    }
}

impl IntoValue for Tracing {
    fn into_value(self) -> Value {
        Value::Tracing(self)
    }
}

impl FromValue for Tracing {
    fn from_value(value: Value) -> HintedStrResult<Self> {
        match value {
            Value::Tracing(tracing) => Ok(tracing),
            _ => Err("Expected tracing".into())
        }
    }
}

impl Repr for Tracing {
    fn repr(&self) -> EcoString {
        let mut result = EcoString::from("tracing(");
        result.push_str(&format!("repeat: {:?}", self.0.repeat_type).to_lowercase());
        result.push_str(", ..)");
        result
    }
}

cast! {
    RepeatType,
    self => match self {
        Self::Scale => "scale",
        Self::Stretch => "stretch",
        Self::Preserve => "preserve",
        Self::Rotate => "rotate",
        Self::None => "none",
    }.into_value(),
    "scale" => Self::Scale,
    "stretch" => Self::Stretch,
    "preserve" => Self::Preserve,
    "rotate" => Self::Rotate,
    "none" => Self::None,
}

#[scope]
impl Tracing {
    /// Create a tracing.
    #[func(constructor)]
    pub fn construct(
        /// The pattern to trace along the path.
        ///
        /// Can be either:
        /// - A curve element that will be repeated along the path
        /// - A dictionary with optional `start`, `repeat`, and `end` keys, each containing a curve element
        pattern_arg: Value,

        /// How to repeat the pattern.
        #[named]
        repeat: Option<RepeatType>,

        /// Space between pattern copies.
        #[named]
        spacing: Option<Length>,
    ) -> SourceResult<Tracing> {
        let pattern = if let Ok(content) = pattern_arg.clone().cast::<Content>() {
            PatternSpec::from_repeat(content)
        } else if let Ok(mut dict) = pattern_arg.cast::<Dict>() {
            let start = dict.take("start").ok().and_then(|v| v.cast().ok());
            let repeat = dict.take("repeat").ok().and_then(|v| v.cast().ok());
            let end = dict.take("end").ok().and_then(|v| v.cast().ok());
            dict.finish(&["start", "repeat", "end"]).at(Span::detached())?;
            PatternSpec { start, repeat, end }
        } else {
            PatternSpec { start: None, repeat: None, end: None }
        };

        Ok(Self(Arc::new(TracingRepr {
            pattern,
            repeat_type: repeat.unwrap_or_default(),
            spacing: spacing.unwrap_or_default(),
        })))
    }
}

impl Tracing {
    pub fn pattern(&self) -> &PatternSpec {
        &self.0.pattern
    }

    pub fn params(&self) -> (RepeatType, Length) {
        (self.0.repeat_type, self.0.spacing)
    }

    /// Build arc length lookup table for a skeleton curve.
    /// This can be cached and reused across multiple pattern applications.
    pub fn build_skeleton_arc_table(&self, skeleton: &Curve, skeleton_length: f64) -> Vec<(f64, f64)> {
        build_arc_length_table(skeleton, skeleton_length)
    }

    /// Center a pattern curve so its bounding box is centered on the origin.
    /// This can be cached and reused across multiple applications.
    pub fn center_pattern_curve(&self, pattern: &Curve) -> Curve {
        center_pattern(pattern)
    }

    /// Apply a pattern along a skeleton in a specific range.
    ///
    /// Optionally accepts pre-computed `arc_table` and `centered_pattern` for better
    /// performance when calling multiple times with the same skeleton/pattern.
    pub fn apply_tracing_in_range(
        &self,
        pattern: &Curve,
        skeleton: &Curve,
        spacing_abs: f64,
        skeleton_length: f64,
        start_offset: f64,
        end_offset: f64,
        arc_table: Option<&[(f64, f64)]>,
        centered_pattern: Option<&Curve>,
    ) -> Curve {
        let mut result = Curve::new();

        let centered = centered_pattern.cloned().unwrap_or_else(|| center_pattern(pattern));
        let pattern_bbox_size = centered.bbox_size();
        let pattern_width = pattern_bbox_size.x.to_raw();

        if pattern_width <= 0.0 || skeleton_length <= 0.0 {
            return result;
        }

        let scratch;
        let table = if let Some(table) = arc_table {
            table
        } else {
            scratch = build_arc_length_table(skeleton, skeleton_length);
            &scratch
        };

        let available_length = end_offset - start_offset;

        if available_length <= 0.0 {
            return result;
        }

        let (num_copies, x_scale) = calculate_pattern_layout(
            self.0.repeat_type,
            available_length,
            pattern_width,
            spacing_abs,
        );

        let scaled_width = pattern_width * x_scale;
        let pattern_step = scaled_width + spacing_abs;

        let mut t_offset = start_offset + (scaled_width / 2.0);

        let boundary_epsilon = skeleton_length * RELATIVE_EPSILON;

        for _ in 0..num_copies {
            if t_offset + (scaled_width / 2.0) <= end_offset + boundary_epsilon {
                let transformed = self.transform_pattern_for_repeat(
                    self.0.repeat_type,
                    &centered,
                    skeleton,
                    table,
                    t_offset,
                    x_scale,
                    skeleton_length,
                );

                result.0.extend(transformed.0.iter().cloned());
            }
            t_offset += pattern_step;
        }

        result
    }

    /// Apply a pattern once at a specific position.
    ///
    /// Optionally accepts pre-computed `arc_table` and `centered_pattern` for better
    /// performance when calling multiple times with the same skeleton/pattern.
    pub fn apply_pattern_once(
        &self,
        pattern: &Curve,
        skeleton: &Curve,
        skeleton_length: f64,
        offset: f64,
        force_preserve: bool,
        arc_table: Option<&[(f64, f64)]>,
        centered_pattern: Option<&Curve>,
    ) -> Curve {
        let mut result = Curve::new();

        let centered = centered_pattern.cloned().unwrap_or_else(|| center_pattern(pattern));
        let pattern_bbox_size = centered.bbox_size();
        let pattern_width = pattern_bbox_size.x.to_raw();

        if pattern_width <= 0.0 || skeleton_length <= 0.0 {
            return result;
        }

        let scratch;
        let table = if let Some(table) = arc_table {
            table
        } else {
            scratch = build_arc_length_table(skeleton, skeleton_length);
            &scratch
        };

        let repeat_type = if force_preserve {
            RepeatType::Preserve
        } else {
            self.0.repeat_type
        };

        let transformed = self.transform_pattern_for_repeat(
            repeat_type,
            &centered,
            skeleton,
            table,
            offset,
            1.0, // No scaling for start/end
            skeleton_length,
        );

        result.0.extend(transformed.0.iter().cloned());
        result
    }

    fn transform_pattern_for_repeat(
        &self,
        repeat: RepeatType,
        pattern: &Curve,
        skeleton: &Curve,
        arc_table: &[(f64, f64)],
        t_offset: f64,
        x_scale: f64,
        skeleton_length: f64,
    ) -> Curve {
        match repeat {
            RepeatType::None | RepeatType::Rotate => transform_pattern_affine(
                pattern,
                skeleton,
                arc_table,
                t_offset,
                None,
                skeleton_length,
                repeat == RepeatType::Rotate,
            ),
            RepeatType::Scale | RepeatType::Preserve | RepeatType::Stretch => {
                let (sx, sy) = match repeat {
                    RepeatType::Scale => (x_scale, x_scale),
                    RepeatType::Preserve => (1.0, 1.0),
                    RepeatType::Stretch => (x_scale, 1.0),
                    _ => unreachable!(),
                };
                deform_pattern_along_skeleton(
                    pattern,
                    skeleton,
                    arc_table,
                    t_offset,
                    sx,
                    sy,
                    skeleton_length,
                )
            }
        }
    }

}

/// Maps pattern points along the skeleton curve with independent x/y scaling.
fn deform_pattern_along_skeleton(
    pattern: &Curve,
    skeleton: &Curve,
    arc_table: &[(f64, f64)],
    t_offset: f64,
    x_scale: f64,
    y_scale: f64,
    skeleton_length: f64,
) -> Curve {
    let mut result = Curve::new();
    let sample_count = adaptive_sample_count(pattern);
    let pattern_samples = sample_pattern_parametrically(pattern, sample_count);

    if pattern_samples.is_empty() {
        return result;
    }

    let is_closed = pattern.0.iter().any(|item| matches!(item, crate::visualize::CurveItem::Close));

    let mut is_first_point = true;

    for (pattern_x, pattern_y) in pattern_samples {
        let x = pattern_x * x_scale;
        let y = pattern_y * y_scale;
        let skeleton_arc_length = t_offset + x;

        if skeleton_arc_length >= 0.0 && skeleton_arc_length <= skeleton_length {
            if let Some((skeleton_pt, normal)) = point_and_normal_at_length(skeleton, arc_table, skeleton_arc_length) {
                let final_pt = Point::new(
                    skeleton_pt.x + normal.x * y,
                    skeleton_pt.y + normal.y * y,
                );

                if is_first_point {
                    result.move_(final_pt);
                    is_first_point = false;
                } else {
                    result.line(final_pt);
                }
            }
        }
    }

    if is_closed && !is_first_point {
        result.close();
    }

    result
}

/// Apply rotation and optional scaling while preserving the pattern's original shape.
fn transform_pattern_affine(
    pattern: &Curve,
    skeleton: &Curve,
    arc_table: &[(f64, f64)],
    t_offset: f64,
    scale_factor: Option<f64>,
    skeleton_length: f64,
    apply_rotation: bool,
) -> Curve {
    let mut result = Curve::new();

    if t_offset < 0.0 || t_offset > skeleton_length {
        return result;
    }

    if let Some((skeleton_center, skeleton_normal)) = point_and_normal_at_length(skeleton, arc_table, t_offset) {
        let mut transform = Affine::translate((skeleton_center.x.to_raw(), skeleton_center.y.to_raw()));

        if apply_rotation {
            let skeleton_tangent = Point::new(skeleton_normal.y, -skeleton_normal.x);
            let angle = skeleton_tangent.y.to_raw().atan2(skeleton_tangent.x.to_raw());
            transform *= Affine::rotate(angle);
        }

        if let Some(scale) = scale_factor {
            transform *= Affine::scale(scale);
        }

        let apply = |p: Point| {
            let transformed = transform * kurbo::Point::new(p.x.to_raw(), p.y.to_raw());
            Point::new(Abs::raw(transformed.x), Abs::raw(transformed.y))
        };

        for item in &pattern.0 {
            match item {
                crate::visualize::CurveItem::Move(p) => result.move_(apply(*p)),
                crate::visualize::CurveItem::Line(p) => result.line(apply(*p)),
                crate::visualize::CurveItem::Cubic(c1, c2, p) => {
                    result.cubic(apply(*c1), apply(*c2), apply(*p));
                }
                crate::visualize::CurveItem::Close => result.close(),
            }
        }
    }

    result
}

/// Calculate pattern layout: number of copies and scale factor to fit along the skeleton.
fn calculate_pattern_layout(
    repeat_type: RepeatType,
    skeleton_length: f64,
    pattern_width: f64,
    spacing_abs: f64,
) -> (usize, f64) {
    let effective_width = pattern_width + spacing_abs;
    if effective_width <= 0.0 {
        return (1, 1.0);
    }
    
    let num_copies = ((skeleton_length + spacing_abs) / effective_width)
        .floor()
        .max(1.0) as usize;
    
    let x_scale = match repeat_type {
        RepeatType::Scale | RepeatType::Stretch => {
            let total_gap = (num_copies - 1) as f64 * spacing_abs;
            let available_for_patterns = skeleton_length - total_gap;
            let scale = available_for_patterns / (num_copies as f64 * pattern_width);
            scale.max(MIN_SCALE_FACTOR)
        }
        RepeatType::Preserve | RepeatType::Rotate | RepeatType::None => 1.0,
    };
    
    (num_copies, x_scale)
}

/// Build arc length lookup table for the curve.
/// Returns a table that maps parameter t to arc length.
fn build_arc_length_table(curve: &Curve, total_length: f64) -> Vec<(f64, f64)> {
    if curve.is_empty() || total_length <= 0.0 {
        return Vec::new();
    }

    let sample_count = adaptive_sample_count(curve);
    let mut table = Vec::with_capacity(sample_count + 1);

    let mut accumulated_length = 0.0;
    let mut prev_point = match get_curve_start_point(curve) {
        Some(p) => p,
        None => return table,
    };

    table.push((0.0, 0.0));

    for i in 1..=sample_count {
        let t = i as f64 / sample_count as f64;
        if let Some(point) = sample_curve_at_t(curve, t, total_length) {
            accumulated_length += (point - prev_point).hypot().to_raw();
            table.push((t, accumulated_length));
            prev_point = point;
        }
    }

    if let Some((_, last_length)) = table.last_mut() {
        *last_length = total_length;
    }

    table
}

/// Find the curve parameter t for a given arc length using binary search and linear interpolation.
fn arc_length_to_parameter(arc_table: &[(f64, f64)], target_length: f64) -> f64 {
    if arc_table.is_empty() || target_length <= 0.0 {
        return 0.0;
    }

    let max_length = arc_table.last().map(|(_, l)| *l).unwrap_or(0.0);
    if target_length >= max_length {
        return 1.0;
    }

    let idx = arc_table.partition_point(|(_, len)| *len < target_length);

    if idx == 0 {
        return 0.0;
    }

    if idx >= arc_table.len() {
        return 1.0;
    }

    let (t1, len1) = arc_table[idx - 1];
    let (t2, len2) = arc_table[idx];

    if len2 - len1 > 0.0 {
        let ratio = (target_length - len1) / (len2 - len1);
        t1 + ratio * (t2 - t1)
    } else {
        t1
    }
}

/// Get the position and normal vector at a given arc length along the curve.
fn point_and_normal_at_length(curve: &Curve, arc_table: &[(f64, f64)], length: f64) -> Option<(Point, Point)> {
    if arc_table.is_empty() {
        return None;
    }

    let total_length = arc_table.last().map(|(_, l)| *l).unwrap_or(0.0);
    if total_length <= 0.0 {
        return None;
    }

    let (pos, tangent) = evaluate_curve_at_length(curve, total_length, length)?;
    let len = tangent.hypot().to_raw();
    let normal = if len > 0.0 {
        Point::new(-tangent.y / len, tangent.x / len)
    } else {
        Point::new(Abs::zero(), Abs::pt(1.0))
    };

    Some((pos, normal))
}

/// Get the position and tangent vector at a given arc length along the curve.
fn evaluate_curve_at_length(curve: &Curve, total_length: f64, target_length: f64) -> Option<(Point, Point)> {
    if total_length <= 0.0 {
        let p = get_curve_start_point(curve)?;
        return Some((p, Point::new(Abs::zero(), Abs::pt(1.0))));
    }

    let mut accumulated_length = 0.0;
    let mut current_point = get_curve_start_point(curve)?;
    let mut segment_start = current_point;

    for item in &curve.0 {
        match item {
            crate::visualize::CurveItem::Move(p) => {
                current_point = *p;
                segment_start = *p;
            }
            crate::visualize::CurveItem::Line(p) => {
                let diff = *p - segment_start;
                let segment_length = diff.hypot().to_raw();

                if accumulated_length + segment_length >= target_length {
                    let segment_t = if segment_length > 0.0 {
                        (target_length - accumulated_length) / segment_length
                    } else {
                        0.0
                    };

                    let pos = Point::new(
                        segment_start.x + diff.x * segment_t,
                        segment_start.y + diff.y * segment_t,
                    );
                    return Some((pos, diff));
                }

                accumulated_length += segment_length;
                segment_start = *p;
                current_point = *p;
            }
            crate::visualize::CurveItem::Cubic(c1, c2, p) => {
                let bez = CubicBez::new(
                    kurbo::Point::new(segment_start.x.to_raw(), segment_start.y.to_raw()),
                    kurbo::Point::new(c1.x.to_raw(), c1.y.to_raw()),
                    kurbo::Point::new(c2.x.to_raw(), c2.y.to_raw()),
                    kurbo::Point::new(p.x.to_raw(), p.y.to_raw()),
                );

                let seg_len = bez.arclen(BEZIER_TOLERANCE);

                if accumulated_length + seg_len >= target_length {
                    let target_arc = target_length - accumulated_length;
                    let segment_t = bez.inv_arclen(target_arc, BEZIER_TOLERANCE);
                    let point = bez.eval(segment_t);
                    let deriv = bez.deriv().eval(segment_t);

                    return Some((
                        Point::new(Abs::raw(point.x), Abs::raw(point.y)),
                        Point::new(Abs::raw(deriv.x), Abs::raw(deriv.y)),
                    ));
                }

                accumulated_length += seg_len;
                segment_start = *p;
                current_point = *p;
            }
            crate::visualize::CurveItem::Close => {
                if let Some(start) = get_curve_start_point(curve) {
                    let diff = start - current_point;
                    let segment_length = diff.hypot().to_raw();

                    if accumulated_length + segment_length >= target_length {
                        let segment_t = if segment_length > 0.0 {
                            (target_length - accumulated_length) / segment_length
                        } else {
                            0.0
                        };

                        let pos = Point::new(
                            current_point.x + diff.x * segment_t,
                            current_point.y + diff.y * segment_t,
                        );
                        return Some((pos, diff));
                    }

                    accumulated_length += segment_length;
                    current_point = start;
                }
            }
        }
    }

    Some((current_point, Point::new(Abs::zero(), Abs::pt(1.0))))
}

/// Sample a pattern curve at regular parameter intervals.
fn sample_pattern_parametrically(pattern: &Curve, num_samples: usize) -> Vec<(f64, f64)> {
    if pattern.is_empty() || num_samples == 0 {
        return Vec::new();
    }

    let mut samples = Vec::with_capacity(num_samples + 1);

    let pattern_bbox = pattern.bbox_size();
    let pattern_width = pattern_bbox.x.to_raw();

    if pattern_width <= 0.0 {
        return samples;
    }

    let total_length = calculate_curve_length(pattern);
    if total_length <= 0.0 {
        return samples;
    }

    for i in 0..=num_samples {
        let t = i as f64 / num_samples as f64;
        if let Some(pattern_point) = sample_curve_at_t(pattern, t, total_length) {
            samples.push((pattern_point.x.to_raw(), pattern_point.y.to_raw()));
        }
    }

    samples
}

/// Calculate the total arc length of a curve by summing all segment lengths.
/// Cubic Bezier segments are approximated using recursive subdivision.
pub fn calculate_curve_length(curve: &Curve) -> f64 {
    let mut total_length = 0.0;
    let mut current_point = Point::zero();
    let mut has_start = false;

    for item in &curve.0 {
        match item {
            crate::visualize::CurveItem::Move(p) => {
                current_point = *p;
                has_start = true;
            }
            crate::visualize::CurveItem::Line(p) => {
                if has_start {
                    let segment_length = (*p - current_point).hypot().to_raw();
                    total_length += segment_length;
                }
                current_point = *p;
                has_start = true;
            }
            crate::visualize::CurveItem::Cubic(c1, c2, p) => {
                if has_start {
                    let bez = CubicBez::new(
                        kurbo::Point::new(current_point.x.to_raw(), current_point.y.to_raw()),
                        kurbo::Point::new(c1.x.to_raw(), c1.y.to_raw()),
                        kurbo::Point::new(c2.x.to_raw(), c2.y.to_raw()),
                        kurbo::Point::new(p.x.to_raw(), p.y.to_raw()),
                    );
                    total_length += bez.arclen(BEZIER_TOLERANCE);
                }
                current_point = *p;
                has_start = true;
            }
            crate::visualize::CurveItem::Close => {
                if let Some(start) = get_curve_start_point(curve) {
                    let segment_length = (start - current_point).hypot().to_raw();
                    total_length += segment_length;
                }
            }
        }
    }

    total_length
}

/// Sample curve at normalized parameter t ∈ [0, 1].
fn sample_curve_at_t(curve: &Curve, t: f64, total_length: f64) -> Option<Point> {
    if curve.is_empty() {
        return None;
    }

    let clamped_t = t.clamp(0.0, 1.0);
    let target_length = clamped_t * total_length;
    sample_curve_at_t_with_length(curve, total_length, target_length)
}

/// Walk through curve segments to find the point at a specific arc length.
fn sample_curve_at_t_with_length(curve: &Curve, total_length: f64, target_length: f64) -> Option<Point> {
    evaluate_curve_at_length(curve, total_length, target_length).map(|(p, _)| p)
}

/// Find the first defined point in a curve (the starting point after first Move or implicit origin).
fn get_curve_start_point(curve: &Curve) -> Option<Point> {
    let cursor = Point::zero();
    for item in &curve.0 {
        match item {
            crate::visualize::CurveItem::Move(p) => return Some(*p),
            crate::visualize::CurveItem::Line(_) => return Some(cursor),
            crate::visualize::CurveItem::Cubic(_, _, _) => return Some(cursor),
            crate::visualize::CurveItem::Close => continue,
        }
    }
    None
}

/// Center a pattern by shifting it so its bounding box is centered on the origin.
fn center_pattern(pattern: &Curve) -> Curve {
    if pattern.is_empty() {
        return Curve::new();
    }

    let bbox = pattern.bbox();
    if bbox.size().x == Abs::zero() || bbox.size().y == Abs::zero() {
        return Curve(pattern.0.clone());
    }

    let center = Point::new(
        (bbox.min.x + bbox.max.x) / 2.0,
        (bbox.min.y + bbox.max.y) / 2.0,
    );
    let offset = -center;

    let mut centered = Curve(pattern.0.clone());
    centered.translate(offset);
    centered
}