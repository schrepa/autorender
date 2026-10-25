//! What each expression of a function stands for, from what its operands
//! stand for: reads of the camera and of positions relative to the eye, or
//! what the detail eye's measures give and a pixel's derivatives, start
//! dependences, and decisions' operands stop there.

use naga::{
    AddressSpace, Expression, GlobalVariable, Handle, MathFunction, RelationalFunction, ScalarKind,
    Span, Type, TypeInner,
};

use super::{
    Flow, Frame, Member, Place, RELATIVE, Read, Root, SIZES, Slot, Step, Taint, Value, Watch,
    lanes_of, type_of,
};

impl Flow<'_> {
    /// What expression `handle` of the frame's function stands for; on a
    /// marked line, its value depends on nothing.
    pub(super) fn evaluate(
        &mut self,
        frame: &Frame<'_>,
        handle: Handle<Expression>,
        expression: &Expression,
    ) -> Slot {
        let span = frame.function.expressions.get_span(handle);
        let slot = match *expression {
            Expression::Access { base, index } => access(frame, base, index),
            Expression::AccessIndex { base, index } => self.access_index(frame, base, index, span),
            Expression::FunctionArgument(index) => argument(frame, index as usize),
            Expression::GlobalVariable(global) => {
                if self.module.global_variables[global].space == AddressSpace::Handle {
                    Slot::Value(Value::clean())
                } else {
                    Slot::Place(Place::of(Root::Global(global)))
                }
            }
            Expression::LocalVariable(local) => Slot::Place(Place::of(Root::Local(local.index()))),
            Expression::Load { pointer } => match frame.slot(pointer) {
                Slot::Place(place) => {
                    let place = place.clone();
                    Slot::Value(self.read(frame, &place, span))
                }
                Slot::Value(value) => Slot::Value(value.clone()),
            },
            Expression::CallResult(function) => {
                let value = frame
                    .calls
                    .get(&handle)
                    .cloned()
                    .unwrap_or_else(Value::clean);
                // Functions named for positions relative to the eye give
                // them; the detail eye's measures give sizes on screen.
                let name = self.module.functions[function].name.as_deref();
                let read = match self.watch {
                    Watch::Camera
                        if name.is_some_and(|name| {
                            name.split('_').any(|word| word == RELATIVE)
                        }) =>
                    {
                        Some(Read::Relative)
                    }
                    Watch::Sizes if name.is_some_and(|name| SIZES.contains(&name)) => {
                        Some(Read::Measure)
                    }
                    Watch::Camera | Watch::Sizes => None,
                };
                match read {
                    Some(read) => {
                        let taint = self.source(read, frame.of, &frame.name, span);
                        Slot::Value(value.with(&taint))
                    }
                    None => Slot::Value(value),
                }
            }
            // A pixel's derivatives are sizes on screen.
            Expression::Derivative { expr, .. } if self.watch == Watch::Sizes => {
                let taint = self.source(Read::Derivative, frame.of, &frame.name, span);
                Slot::Value(frame.value(expr).with(&taint))
            }
            _ => Slot::Value(self.value(frame, handle, expression)),
        };
        match slot {
            Slot::Value(value) if frame.marked[handle.index()] => Slot::Value(value.cleared()),
            slot => slot,
        }
    }

    /// What a value expression that reads no memory depends on.
    fn value(
        &self,
        frame: &Frame<'_>,
        handle: Handle<Expression>,
        expression: &Expression,
    ) -> Value {
        match *expression {
            Expression::Compose { ty, ref components } => self.compose(frame, ty, components),
            Expression::Splat { size, value } => {
                Value::Lanes(vec![frame.taint(value); usize::from(u8::from(size))])
            }
            Expression::Swizzle {
                size,
                vector,
                pattern,
            } => {
                let value = frame.value(vector);
                let count = usize::from(u8::from(size));
                Value::Lanes(
                    pattern[..count]
                        .iter()
                        .map(|&component| value.lane(component as usize))
                        .collect(),
                )
            }
            // What a texture gives is its texels as filtering averages them
            // over the pixel, wherever a size on screen moves the read (a
            // level's half texel kept inside a frame).
            Expression::ImageSample { .. } | Expression::ImageLoad { .. }
                if self.watch == Watch::Sizes =>
            {
                Value::clean()
            }
            // What a texture gives follows where it is read; how it is
            // filtered (levels, gradients) is the view's own.
            Expression::ImageSample {
                coordinate,
                array_index,
                depth_ref,
                ..
            } => Value::Whole(taints(frame, [Some(coordinate), array_index, depth_ref])),
            Expression::ImageLoad {
                coordinate,
                array_index,
                sample,
                ..
            } => Value::Whole(taints(frame, [Some(coordinate), array_index, sample])),
            Expression::Unary { expr, .. } | Expression::Derivative { expr, .. } => {
                frame.value(expr)
            }
            Expression::Binary { left, right, .. } => self.binary(frame, handle, left, right),
            // A decision's operands stop where it decides: its condition
            // here, read by the decisions' pass.
            Expression::Select { accept, reject, .. } => {
                let mut value = frame.value(accept);
                value.join(&frame.value(reject));
                value
            }
            Expression::Relational { fun, argument } => match fun {
                RelationalFunction::All | RelationalFunction::Any => {
                    Value::Whole(frame.taint(argument))
                }
                _ => frame.value(argument),
            },
            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => self.math(frame, handle, fun, [Some(arg), arg1, arg2, arg3]),
            Expression::As { expr, kind, .. } => {
                if kind != ScalarKind::Bool && self.is_bool(frame, expr) {
                    frame.value(expr).cleared()
                } else {
                    frame.value(expr)
                }
            }
            Expression::AtomicResult { .. } | Expression::WorkGroupUniformLoadResult { .. } => {
                frame
                    .calls
                    .get(&handle)
                    .cloned()
                    .unwrap_or_else(Value::clean)
            }
            _ => Value::clean(),
        }
    }

    /// A member, lane, column or element `index` of `base`: a place's part
    /// for pointers, a value's part otherwise, eye-relative where the
    /// member is named `relative` and the camera is watched.
    fn access_index(
        &mut self,
        frame: &Frame<'_>,
        base: Handle<Expression>,
        index: u32,
        span: Span,
    ) -> Slot {
        let relative = self.watch == Watch::Camera && self.names_relative(frame, base, index);
        match frame.slot(base) {
            Slot::Place(place) => {
                Slot::Place(place.step(Step::Index(index), relative, &Taint::new()))
            }
            Slot::Value(value) => {
                let part = match type_of(frame.info, self.module, base) {
                    TypeInner::Struct { .. } => value.at(&[index]),
                    _ => Value::Whole(value.lane(index as usize)),
                };
                if relative {
                    let taint = self.source(Read::Relative, frame.of, &frame.name, span);
                    Slot::Value(part.with(&taint))
                } else {
                    Slot::Value(part)
                }
            }
        }
    }

    /// Whether member `index` of the struct `base` is (or points at) is
    /// named `relative`.
    fn names_relative(&self, frame: &Frame<'_>, base: Handle<Expression>, index: u32) -> bool {
        let module = self.module;
        let mut inner = type_of(frame.info, module, base);
        if let TypeInner::Pointer { base: pointee, .. } = *inner {
            inner = &module.types[pointee].inner;
        }
        let TypeInner::Struct { members, .. } = inner else {
            return false;
        };
        members
            .get(index as usize)
            .and_then(|member| member.name.as_deref())
            == Some(RELATIVE)
    }

    /// What reading `place` at `span` gives: what its root holds there,
    /// eye-relative where a step names `relative`, depending on the indices
    /// chosen as the shader runs too.
    pub(super) fn read(&mut self, frame: &Frame<'_>, place: &Place, span: Span) -> Value {
        let value = match place.root {
            Root::Local(local) => frame.locals[local].at_steps(&place.steps),
            Root::Argument(argument) => frame
                .pointees
                .get(argument)
                .and_then(Option::as_ref)
                .map_or_else(Value::clean, |pointee| pointee.at_steps(&place.steps)),
            Root::Global(global) => self.global(frame, global, &place.steps, span),
        };
        let mut more = place.index.clone();
        if place.relative {
            more.extend(self.source(Read::Relative, frame.of, &frame.name, span));
        }
        if more.is_empty() {
            value
        } else {
            value.with(&more)
        }
    }

    /// What module variable `global` holds at `steps`: the camera's reads,
    /// what functions stored in private and workgroup variables, and
    /// nothing in other buffers.
    fn global(
        &mut self,
        frame: &Frame<'_>,
        global: Handle<GlobalVariable>,
        steps: &[Step],
        span: Span,
    ) -> Value {
        if self
            .camera
            .as_ref()
            .is_some_and(|(camera, _)| *camera == global)
        {
            return self.camera_read(frame, steps, span);
        }
        match self.module.global_variables[global].space {
            AddressSpace::Private | AddressSpace::WorkGroup => {
                let mut value = self
                    .globals
                    .get(&global)
                    .cloned()
                    .unwrap_or_else(Value::clean);
                if let Some(own) = frame.globals.get(&global) {
                    value.join(own);
                }
                value.at_steps(steps)
            }
            _ => Value::clean(),
        }
    }

    /// What reading the camera at `steps` gives.
    fn camera_read(&mut self, frame: &Frame<'_>, steps: &[Step], span: Span) -> Value {
        match steps.split_first() {
            None => {
                let members = self.camera.as_ref().map_or(0, |(_, members)| members.len());
                Value::Members(
                    (0..members)
                        .map(|member| self.camera_member(frame, member, None, span))
                        .collect(),
                )
            }
            Some((&Step::Index(member), rest)) => {
                self.camera_member(frame, member as usize, rest.first().copied(), span)
            }
            Some((&Step::Any, _)) => Value::clean(),
        }
    }

    /// What reading member `member` of the camera gives, lane `lane` of it
    /// where one is read: the eyes' first three lanes, a pixel's angle
    /// (`options.y`) and a matrix's parts are reads of the camera.
    fn camera_member(
        &mut self,
        frame: &Frame<'_>,
        member: usize,
        lane: Option<Step>,
        span: Span,
    ) -> Value {
        let meaning = self
            .camera
            .as_ref()
            .and_then(|(_, members)| members.get(member).copied())
            .unwrap_or(Member::Other);
        let (read, lanes) = match meaning {
            Member::Eye => (Read::Eye, [true, true, true, false]),
            Member::Detail => (Read::Detail, [true, true, true, false]),
            Member::Options => (Read::PixelAngle, [false, true, false, false]),
            Member::Matrix => {
                let taint = self.source(Read::ViewAxis, frame.of, &frame.name, span);
                return if lane.is_some() {
                    Value::Whole(taint)
                } else {
                    Value::Matrix(taint)
                };
            }
            Member::Other => return Value::clean(),
        };
        match lane {
            Some(Step::Index(index)) if !lanes.get(index as usize).copied().unwrap_or(false) => {
                Value::clean()
            }
            Some(_) => Value::Whole(self.source(read, frame.of, &frame.name, span)),
            None => {
                let taint = self.source(read, frame.of, &frame.name, span);
                Value::Lanes(
                    lanes
                        .iter()
                        .map(|&shown| if shown { taint.clone() } else { Taint::new() })
                        .collect(),
                )
            }
        }
    }

    /// A vector, struct, matrix or array of `components`.
    fn compose(
        &self,
        frame: &Frame<'_>,
        ty: Handle<Type>,
        components: &[Handle<Expression>],
    ) -> Value {
        match &self.module.types[ty].inner {
            TypeInner::Vector { size, .. } => {
                let count = usize::from(u8::from(*size));
                let mut lanes = Vec::with_capacity(count);
                for &component in components {
                    let value = frame.value(component);
                    match lanes_of(type_of(frame.info, self.module, component)) {
                        Some(inner) => lanes.extend((0..inner).map(|lane| value.lane(lane))),
                        None => lanes.push(value.collapse()),
                    }
                }
                if lanes.len() == count {
                    Value::Lanes(lanes)
                } else {
                    Value::Whole(lanes.into_iter().flatten().collect())
                }
            }
            TypeInner::Struct { .. } => Value::Members(
                components
                    .iter()
                    .map(|&component| frame.value(component))
                    .collect(),
            ),
            _ => Value::Whole(
                components
                    .iter()
                    .flat_map(|&component| frame.taint(component))
                    .collect(),
            ),
        }
    }

    /// An operator applied lane by lane, but for matrices, which mix lanes:
    /// a camera matrix transforms whole, and applied to a constant vector
    /// gives the view's eye or an axis.
    fn binary(
        &self,
        frame: &Frame<'_>,
        handle: Handle<Expression>,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Value {
        let (first, second) = (frame.value(left), frame.value(right));
        for (matrix, other, operand) in [(&first, &second, right), (&second, &first, left)] {
            if let Value::Matrix(taint) = matrix {
                let mut all = other.collapse();
                if frame.constant[operand.index()] {
                    all.extend(taint.iter().copied());
                }
                return Value::Whole(all);
            }
        }
        let module = self.module;
        let is_matrix = |operand: Handle<Expression>| {
            matches!(
                type_of(frame.info, module, operand),
                TypeInner::Matrix { .. }
            )
        };
        match lanes_of(type_of(frame.info, module, handle)) {
            Some(count) if !is_matrix(left) && !is_matrix(right) => Value::Lanes(
                (0..count)
                    .map(|lane| {
                        first
                            .lane(lane)
                            .union(&second.lane(lane))
                            .copied()
                            .collect()
                    })
                    .collect(),
            ),
            _ => Value::Whole(
                first
                    .collapse()
                    .union(&second.collapse())
                    .copied()
                    .collect(),
            ),
        }
    }

    /// A built-in function's result: decisions' operands stop where they
    /// decide (all of `smoothstep`'s and `step`'s, `mix`'s weight, and
    /// `saturate`'s where sizes on screen are watched); a direction
    /// (`normalize`'s) holds no size; functions of each lane go lane by
    /// lane; the rest mix everything.
    fn math(
        &self,
        frame: &Frame<'_>,
        handle: Handle<Expression>,
        fun: MathFunction,
        arguments: [Option<Handle<Expression>>; 4],
    ) -> Value {
        let operands: Vec<Handle<Expression>> = arguments.into_iter().flatten().collect();
        match fun {
            MathFunction::SmoothStep | MathFunction::Step => Value::clean(),
            MathFunction::Saturate | MathFunction::Normalize if self.watch == Watch::Sizes => {
                Value::clean()
            }
            MathFunction::Mix => self.lanewise(frame, handle, &operands[..operands.len().min(2)]),
            MathFunction::Abs
            | MathFunction::Min
            | MathFunction::Max
            | MathFunction::Clamp
            | MathFunction::Saturate
            | MathFunction::Cos
            | MathFunction::Cosh
            | MathFunction::Sin
            | MathFunction::Sinh
            | MathFunction::Tan
            | MathFunction::Tanh
            | MathFunction::Acos
            | MathFunction::Asin
            | MathFunction::Atan
            | MathFunction::Atan2
            | MathFunction::Asinh
            | MathFunction::Acosh
            | MathFunction::Atanh
            | MathFunction::Radians
            | MathFunction::Degrees
            | MathFunction::Ceil
            | MathFunction::Floor
            | MathFunction::Round
            | MathFunction::Fract
            | MathFunction::Trunc
            | MathFunction::Ldexp
            | MathFunction::Exp
            | MathFunction::Exp2
            | MathFunction::Log
            | MathFunction::Log2
            | MathFunction::Pow
            | MathFunction::Sign
            | MathFunction::Fma
            | MathFunction::Sqrt
            | MathFunction::InverseSqrt
            | MathFunction::QuantizeToF16 => self.lanewise(frame, handle, &operands),
            _ => Value::Whole(taints(frame, arguments)),
        }
    }

    /// What expression `handle`'s lanes depend on: each the same lane of
    /// every operand.
    fn lanewise(
        &self,
        frame: &Frame<'_>,
        handle: Handle<Expression>,
        operands: &[Handle<Expression>],
    ) -> Value {
        let values: Vec<Value> = operands
            .iter()
            .map(|&operand| frame.value(operand))
            .collect();
        match lanes_of(type_of(frame.info, self.module, handle)) {
            Some(count) => Value::Lanes(
                (0..count)
                    .map(|lane| values.iter().flat_map(|value| value.lane(lane)).collect())
                    .collect(),
            ),
            None => Value::Whole(values.iter().flat_map(Value::collapse).collect()),
        }
    }

    /// Whether expression `handle` is a boolean or a vector of them.
    pub(super) fn is_bool(&self, frame: &Frame<'_>, handle: Handle<Expression>) -> bool {
        matches!(
            type_of(frame.info, self.module, handle),
            TypeInner::Scalar(scalar) | TypeInner::Vector { scalar, .. }
                if scalar.kind == ScalarKind::Bool
        )
    }
}

/// A dynamic index into `base`: a place's part chosen as the shader runs for
/// pointers, which depends on the index too; for values, any element, and
/// the index.
fn access(frame: &Frame<'_>, base: Handle<Expression>, index: Handle<Expression>) -> Slot {
    match frame.slot(base) {
        Slot::Place(place) => Slot::Place(place.step(Step::Any, false, &frame.taint(index))),
        Slot::Value(value) => {
            let mut taint = match value {
                Value::Matrix(taint) => taint.clone(),
                _ => value.collapse(),
            };
            taint.extend(frame.taint(index));
            Slot::Value(Value::Whole(taint))
        }
    }
}

/// Argument `index` of the frame's function: what pointer arguments point
/// at is a place, the others values.
fn argument(frame: &Frame<'_>, index: usize) -> Slot {
    if frame.pointees.get(index).is_some_and(Option::is_some) {
        Slot::Place(Place::of(Root::Argument(index)))
    } else {
        Slot::Value(
            frame
                .arguments
                .get(index)
                .cloned()
                .unwrap_or_else(Value::clean),
        )
    }
}

/// Everything the expressions among `handles` depend on.
fn taints<const N: usize>(frame: &Frame<'_>, handles: [Option<Handle<Expression>>; N]) -> Taint {
    handles
        .into_iter()
        .flatten()
        .flat_map(|handle| frame.taint(handle))
        .collect()
}
