use std::sync::Arc;

use ecow::EcoString;
use crate::foundations::{func, scope, ty, Content, Repr, cast, IntoValue, Value, FromValue, Reflect, CastInfo, Type};
use crate::diag::{SourceResult, HintedStrResult};
use crate::layout::{Abs, Length, Point};
use crate::visualize::Curve;

const PATTERN_SAMPLE_COUNT: usize = 100;
const ARC_LENGTH_TABLE_SAMPLES: usize = 200;
const TANGENT_DELTA: f64 = 0.001;
const BEZIER_TOLERANCE: f64 = 0.01;
const BEZIER_MAX_DEPTH: u32 = 8;
const MIN_SCALE_FACTOR: f64 = 0.1;

/// Trait for pattern transformation strategies.
trait PatternTransformer {
    fn transform(
        &self,
        pattern: &Curve,
        skeleton: &Curve,
        arc_table: &[(f64, f64)],
        t_offset: f64,
        x_scale: f64,
        skeleton_length: f64,
        pattern_width: f64,
    ) -> Curve;
}

/// Transformation for None repeat type: simple translation.
struct NoneTransformer;

impl PatternTransformer for NoneTransformer {
    fn transform(
        &self,
        pattern: &Curve,
        skeleton: &Curve,
        arc_table: &[(f64, f64)],
        t_offset: f64,
        _x_scale: f64,
        skeleton_length: f64,
        pattern_width: f64,
    ) -> Curve {
        let mut result = Curve::new();
        let pattern_center_pos = t_offset + pattern_width / 2.0;

        if pattern_center_pos < 0.0 || pattern_center_pos > skeleton_length {
            return result;
        }

        if let Some((skeleton_center, _)) = point_and_normal_at_length(skeleton, arc_table, pattern_center_pos) {
            for item in &pattern.0 {
                match item {
                    crate::visualize::CurveItem::Move(p) => {
                        result.move_(Point::new(p.x + skeleton_center.x, p.y + skeleton_center.y));
                    }
                    crate::visualize::CurveItem::Line(p) => {
                        result.line(Point::new(p.x + skeleton_center.x, p.y + skeleton_center.y));
                    }
                    crate::visualize::CurveItem::Cubic(c1, c2, p) => {
                        result.cubic(
                            Point::new(c1.x + skeleton_center.x, c1.y + skeleton_center.y),
                            Point::new(c2.x + skeleton_center.x, c2.y + skeleton_center.y),
                            Point::new(p.x + skeleton_center.x, p.y + skeleton_center.y),
                        );
                    }
                    crate::visualize::CurveItem::Close => {
                        result.close();
                    }
                }
            }
        }

        result
    }
}

/// Transformation for Rotate repeat type: rotation without deformation.
struct RotateTransformer;

impl PatternTransformer for RotateTransformer {
    fn transform(
        &self,
        pattern: &Curve,
        skeleton: &Curve,
        arc_table: &[(f64, f64)],
        t_offset: f64,
        _x_scale: f64,
        skeleton_length: f64,
        pattern_width: f64,
    ) -> Curve {
        transform_pattern_affine(pattern, skeleton, arc_table, t_offset, None, skeleton_length, pattern_width)
    }
}

/// Transformation for Scale repeat type: uniform deformation in both axes.
struct ScaleTransformer;

impl PatternTransformer for ScaleTransformer {
    fn transform(
        &self,
        pattern: &Curve,
        skeleton: &Curve,
        arc_table: &[(f64, f64)],
        t_offset: f64,
        x_scale: f64,
        skeleton_length: f64,
        _pattern_width: f64,
    ) -> Curve {
        deform_pattern_along_skeleton(pattern, skeleton, arc_table, t_offset, x_scale, x_scale, skeleton_length)
    }
}

/// Transformation for Preserve repeat type: deformation without scaling.
struct PreserveTransformer;

impl PatternTransformer for PreserveTransformer {
    fn transform(
        &self,
        pattern: &Curve,
        skeleton: &Curve,
        arc_table: &[(f64, f64)],
        t_offset: f64,
        _x_scale: f64,
        skeleton_length: f64,
        _pattern_width: f64,
    ) -> Curve {
        deform_pattern_along_skeleton(pattern, skeleton, arc_table, t_offset, 1.0, 1.0, skeleton_length)
    }
}

/// Transformation for Stretch repeat type: horizontal deformation only.
struct StretchTransformer;

impl PatternTransformer for StretchTransformer {
    fn transform(
        &self,
        pattern: &Curve,
        skeleton: &Curve,
        arc_table: &[(f64, f64)],
        t_offset: f64,
        x_scale: f64,
        skeleton_length: f64,
        _pattern_width: f64,
    ) -> Curve {
        deform_pattern_along_skeleton(pattern, skeleton, arc_table, t_offset, x_scale, 1.0, skeleton_length)
    }
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

/// Pattern specification.
#[derive(Debug, Clone, PartialEq)]
pub struct PatternSpec {
    pub content: Content,
}

impl std::hash::Hash for PatternSpec {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.content.hash(state);
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

/// Internal representation.
#[derive(Debug, Clone)]
struct TracingRepr {
    pattern: PatternSpec,
    repeat_type: RepeatType,
    spacing: Length,
}

impl PartialEq for TracingRepr {
    fn eq(&self, other: &Self) -> bool {
        self.pattern == other.pattern
            && self.repeat_type == other.repeat_type
            && self.spacing == other.spacing
    }
}

impl Eq for TracingRepr {}

impl std::hash::Hash for TracingRepr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.pattern.hash(state);
        self.repeat_type.hash(state);
        self.spacing.hash(state);
    }
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
        pattern: Content,

        /// How to repeat the pattern.
        #[named]
        repeat: Option<RepeatType>,

        /// Space between pattern copies.
        #[named]
        spacing: Option<Length>,
    ) -> SourceResult<Tracing> {
        Ok(Self(Arc::new(TracingRepr {
            pattern: PatternSpec { content: pattern },
            repeat_type: repeat.unwrap_or(RepeatType::Scale),
            spacing: spacing.unwrap_or(Length::zero()),
        })))
    }
}

impl Tracing {
    pub fn content(&self) -> &Content {
        &self.0.pattern.content
    }

    pub fn pattern(&self) -> &PatternSpec {
        &self.0.pattern
    }

    pub fn params(&self) -> (RepeatType, Length) {
        (self.0.repeat_type, self.0.spacing)
    }

    pub fn apply_tracing_algorithm(
        &self,
        pattern: &Curve,
        skeleton: &Curve,
        spacing_abs: f64,
        skeleton_length: f64,
    ) -> Curve {
        let mut result = Curve::new();

        let centered_pattern = center_pattern(pattern);
        let pattern_bbox_size = centered_pattern.bbox_size();
        let pattern_width = pattern_bbox_size.x.to_raw();

        if pattern_width <= 0.0 || skeleton_length <= 0.0 {
            return result;
        }

        let arc_table = build_arc_length_table(skeleton, skeleton_length);

        let (num_copies, x_scale) = calculate_pattern_layout(
            self.0.repeat_type,
            skeleton_length,
            pattern_width,
            spacing_abs,
        );

        let pattern_step = (pattern_width * x_scale) + spacing_abs;
        let mut t_offset = 0.0;

        for _ in 0..num_copies {
            let transformed = self.transform_pattern_for_repeat(
                self.0.repeat_type,
                &centered_pattern,
                skeleton,
                &arc_table,
                t_offset,
                x_scale,
                skeleton_length,
                pattern_width,
            );

            result.0.extend(transformed.0.iter().cloned());
            t_offset += pattern_step;
        }

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
        pattern_width: f64,
    ) -> Curve {
        let transformer: &dyn PatternTransformer = match repeat {
            RepeatType::None => &NoneTransformer,
            RepeatType::Rotate => &RotateTransformer,
            RepeatType::Scale => &ScaleTransformer,
            RepeatType::Preserve => &PreserveTransformer,
            RepeatType::Stretch => &StretchTransformer,
        };

        transformer.transform(
            pattern,
            skeleton,
            arc_table,
            t_offset,
            x_scale,
            skeleton_length,
            pattern_width,
        )
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
    let pattern_samples = sample_pattern_parametrically(pattern, PATTERN_SAMPLE_COUNT);

    if pattern_samples.is_empty() {
        return result;
    }

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
    pattern_width: f64,
) -> Curve {
    let mut result = Curve::new();

    let scale = scale_factor.unwrap_or(1.0);
    let pattern_center_pos = t_offset + (pattern_width * scale) / 2.0;

    if pattern_center_pos < 0.0 || pattern_center_pos > skeleton_length {
        return result;
    }

    if let Some((skeleton_center, skeleton_normal)) = point_and_normal_at_length(skeleton, arc_table, pattern_center_pos) {
        let skeleton_tangent = Point::new(skeleton_normal.y, -skeleton_normal.x);
        let angle = skeleton_tangent.y.to_raw().atan2(skeleton_tangent.x.to_raw());
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        for item in &pattern.0 {
            match item {
                crate::visualize::CurveItem::Move(p) => {
                    let transformed = transform_point_affine(*p, skeleton_center, cos_a, sin_a, scale_factor);
                    result.move_(transformed);
                }
                crate::visualize::CurveItem::Line(p) => {
                    let transformed = transform_point_affine(*p, skeleton_center, cos_a, sin_a, scale_factor);
                    result.line(transformed);
                }
                crate::visualize::CurveItem::Cubic(c1, c2, p) => {
                    let transformed_c1 = transform_point_affine(*c1, skeleton_center, cos_a, sin_a, scale_factor);
                    let transformed_c2 = transform_point_affine(*c2, skeleton_center, cos_a, sin_a, scale_factor);
                    let transformed_p = transform_point_affine(*p, skeleton_center, cos_a, sin_a, scale_factor);
                    result.cubic(transformed_c1, transformed_c2, transformed_p);
                }
                crate::visualize::CurveItem::Close => {
                    result.close();
                }
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
        RepeatType::Scale => {
            let total_gap = (num_copies - 1) as f64 * spacing_abs;
            let available_for_patterns = skeleton_length - total_gap;
            let scale = available_for_patterns / (num_copies as f64 * pattern_width);
            scale.max(MIN_SCALE_FACTOR)
        }
        RepeatType::Stretch => {
            let scale = skeleton_length / (num_copies as f64 * pattern_width);
            scale.max(MIN_SCALE_FACTOR)
        }
        RepeatType::Preserve | RepeatType::Rotate | RepeatType::None => 1.0,
    };
    
    (num_copies, x_scale)
}

/// Apply rotation and optional uniform scaling around a center point.
fn transform_point_affine(
    point: Point,
    final_center: Point,
    cos_a: f64,
    sin_a: f64,
    scale_factor: Option<f64>,
) -> Point {
    let mut x = point.x.to_raw();
    let mut y = point.y.to_raw();

    if let Some(scale) = scale_factor {
        x *= scale;
        y *= scale;
    }

    let rotated_x = x * cos_a - y * sin_a;
    let rotated_y = x * sin_a + y * cos_a;
    Point::new(
        final_center.x + Abs::raw(rotated_x),
        final_center.y + Abs::raw(rotated_y),
    )
}

/// Build arc length lookup table for the curve.
/// Returns a table that maps parameter t to arc length.
fn build_arc_length_table(curve: &Curve, total_length: f64) -> Vec<(f64, f64)> {
    let mut table = Vec::new();

    if curve.is_empty() || total_length <= 0.0 {
        return table;
    }

    let mut accumulated_length = 0.0;
    let mut prev_point = match get_curve_start_point(curve) {
        Some(p) => p,
        None => return table,
    };

    table.push((0.0, 0.0));

    for i in 1..=ARC_LENGTH_TABLE_SAMPLES {
        let t = i as f64 / ARC_LENGTH_TABLE_SAMPLES as f64;
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

/// Find the curve parameter t for a given arc length using linear interpolation.
fn arc_length_to_parameter(arc_table: &[(f64, f64)], target_length: f64) -> f64 {
    if arc_table.is_empty() || target_length <= 0.0 {
        return 0.0;
    }
    
    let max_length = arc_table.last().map(|(_, l)| *l).unwrap_or(0.0);
    if target_length >= max_length {
        return 1.0;
    }
    
    for window in arc_table.windows(2) {
        let (t1, len1) = window[0];
        let (t2, len2) = window[1];
        
        if target_length >= len1 && target_length <= len2 {
            if len2 - len1 > 0.0 {
                let ratio = (target_length - len1) / (len2 - len1);
                return t1 + ratio * (t2 - t1);
            }
            return t1;
        }
    }
    
    1.0
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

    let t = arc_length_to_parameter(arc_table, length);
    let skeleton_point = sample_curve_at_t(curve, t, total_length)?;

    let t1 = (t - TANGENT_DELTA).max(0.0);
    let t2 = (t + TANGENT_DELTA).min(1.0);

    let p1 = sample_curve_at_t(curve, t1, total_length)?;
    let p2 = sample_curve_at_t(curve, t2, total_length)?;

    let diff = p2 - p1;
    let length_val = diff.hypot().to_raw();

    let normal = if length_val > 0.0 {
        let tx = diff.x.to_raw() / length_val;
        let ty = diff.y.to_raw() / length_val;
        Point::new(Abs::raw(-ty), Abs::raw(tx))
    } else {
        Point::new(Abs::raw(0.0), Abs::raw(1.0))
    };

    Some((skeleton_point, normal))
}

/// Sample a pattern curve at regular parameter intervals.
fn sample_pattern_parametrically(pattern: &Curve, num_samples: usize) -> Vec<(f64, f64)> {
    let mut samples = Vec::new();

    if pattern.is_empty() || num_samples == 0 {
        return samples;
    }

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
                    let segment_length = estimate_cubic_bezier_length_recursive(current_point, *c1, *c2, *p, 0, BEZIER_MAX_DEPTH);
                    total_length += segment_length;
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

/// Approximate cubic Bezier curve length using adaptive recursive subdivision.
fn estimate_cubic_bezier_length_recursive(
    p0: Point, 
    p1: Point, 
    p2: Point, 
    p3: Point, 
    depth: u32,
    max_depth: u32
) -> f64 {
    let chord = (p3 - p0).hypot().to_raw();
    
    let poly_length = (p1 - p0).hypot().to_raw() 
                    + (p2 - p1).hypot().to_raw() 
                    + (p3 - p2).hypot().to_raw();
    
    if depth >= max_depth || (poly_length - chord).abs() < BEZIER_TOLERANCE {
        return (poly_length + chord) / 2.0;
    }
    
    let p01 = Point::new((p0.x + p1.x) / 2.0, (p0.y + p1.y) / 2.0);
    let p12 = Point::new((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0);
    let p23 = Point::new((p2.x + p3.x) / 2.0, (p2.y + p3.y) / 2.0);
    
    let p012 = Point::new((p01.x + p12.x) / 2.0, (p01.y + p12.y) / 2.0);
    let p123 = Point::new((p12.x + p23.x) / 2.0, (p12.y + p23.y) / 2.0);
    
    let p0123 = Point::new((p012.x + p123.x) / 2.0, (p012.y + p123.y) / 2.0);
    
    let left_length = estimate_cubic_bezier_length_recursive(p0, p01, p012, p0123, depth + 1, max_depth);
    let right_length = estimate_cubic_bezier_length_recursive(p0123, p123, p23, p3, depth + 1, max_depth);
    
    left_length + right_length
}

/// Sample curve at normalized parameter t ∈ [0, 1].
fn sample_curve_at_t(curve: &Curve, t: f64, total_length: f64) -> Option<Point> {
    if curve.is_empty() {
        return None;
    }

    let clamped_t = t.clamp(0.0, 1.0);
    let target_length = clamped_t * total_length;
    sample_curve_at_t_with_length(curve, clamped_t, total_length, target_length)
}

/// Walk through curve segments to find the point at a specific arc length.
fn sample_curve_at_t_with_length(curve: &Curve, _t: f64, total_length: f64, target_length: f64) -> Option<Point> {
    if total_length <= 0.0 {
        return get_curve_start_point(curve);
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
                let segment_length = (*p - segment_start).hypot().to_raw();

                if accumulated_length + segment_length >= target_length {
                    let segment_t = if segment_length > 0.0 {
                        (target_length - accumulated_length) / segment_length
                    } else {
                        0.0
                    };

                    return Some(Point::new(
                        segment_start.x + (p.x - segment_start.x) * segment_t,
                        segment_start.y + (p.y - segment_start.y) * segment_t,
                    ));
                }

                accumulated_length += segment_length;
                segment_start = *p;
                current_point = *p;
            }
            crate::visualize::CurveItem::Cubic(c1, c2, p) => {
                let seg_len = estimate_cubic_bezier_length_recursive(segment_start, *c1, *c2, *p, 0, BEZIER_MAX_DEPTH);

                if accumulated_length + seg_len >= target_length {
                    let segment_t = if seg_len > 0.0 {
                        (target_length - accumulated_length) / seg_len
                    } else {
                        0.0
                    };

                    let it = 1.0 - segment_t;
                    let x = it.powi(3) * segment_start.x.to_raw() +
                            3.0 * it.powi(2) * segment_t * c1.x.to_raw() +
                            3.0 * it * segment_t.powi(2) * c2.x.to_raw() +
                            segment_t.powi(3) * p.x.to_raw();
                    let y = it.powi(3) * segment_start.y.to_raw() +
                            3.0 * it.powi(2) * segment_t * c1.y.to_raw() +
                            3.0 * it * segment_t.powi(2) * c2.y.to_raw() +
                            segment_t.powi(3) * p.y.to_raw();

                    return Some(Point::new(Abs::raw(x), Abs::raw(y)));
                }

                accumulated_length += seg_len;
                segment_start = *p;
                current_point = *p;
            }
            crate::visualize::CurveItem::Close => {
                if let Some(start) = get_curve_start_point(curve) {
                    let segment_length = (start - current_point).hypot().to_raw();

                    if accumulated_length + segment_length >= target_length {
                        let segment_t = if segment_length > 0.0 {
                            (target_length - accumulated_length) / segment_length
                        } else {
                            0.0
                        };

                        return Some(Point::new(
                            current_point.x + (start.x - current_point.x) * segment_t,
                            current_point.y + (start.y - current_point.y) * segment_t,
                        ));
                    }

                    accumulated_length += segment_length;
                    current_point = start;
                }
            }
        }
    }

    Some(current_point)
}

/// Find the first defined point in a curve.
fn get_curve_start_point(curve: &Curve) -> Option<Point> {
    for item in &curve.0 {
        match item {
            crate::visualize::CurveItem::Move(p) => return Some(*p),
            crate::visualize::CurveItem::Line(p) => return Some(*p),
            crate::visualize::CurveItem::Cubic(_, _, p) => return Some(*p),
            crate::visualize::CurveItem::Close => continue,
        }
    }
    None
}

/// Center a pattern by shifting it so its bounding box is centered on the x-axis.
fn center_pattern(pattern: &Curve) -> Curve {
    if pattern.is_empty() {
        return Curve::new();
    }

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut has_points = false;

    for item in &pattern.0 {
        let mut record = |point: Point| {
            min_x = min_x.min(point.x.to_raw());
            min_y = min_y.min(point.y.to_raw());
            has_points = true;
        };

        match item {
            crate::visualize::CurveItem::Move(p) | crate::visualize::CurveItem::Line(p) => record(*p),
            crate::visualize::CurveItem::Cubic(c1, c2, p) => {
                record(*c1);
                record(*c2);
                record(*p);
            }
            crate::visualize::CurveItem::Close => {}
        }
    }

    if !has_points {
        return Curve(pattern.0.clone());
    }

    let bbox = pattern.bbox_size();
    let offset = Point::new(
        Abs::raw(-min_x),
        -Abs::raw(min_y) - bbox.y / 2.0,
    );

    let mut centered = Curve(pattern.0.clone());
    centered.translate(offset);
    centered
}