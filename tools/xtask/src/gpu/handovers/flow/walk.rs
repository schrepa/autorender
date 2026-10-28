use naga::{
    AddressSpace, Block, Expression, Function, GlobalVariable, Handle, MathFunction, ScalarKind,
    Span, Statement,
};

use super::{
    Call, Decision, Flow, Frame, Origin, Place, Root, Sink, Slot, Step, Summary, Taint, Value,
    Watch, bounds_loop, clean,
};

impl Flow<'_> {
    pub(super) fn walk(&mut self, frame: &mut Frame<'_>, block: &Block, decide: bool) -> bool {
        let mut grew = false;
        for (statement, span) in block.span_iter() {
            match *statement {
                Statement::Block(ref inner) => grew |= self.walk(frame, inner, decide),
                Statement::If {
                    condition,
                    ref accept,
                    ref reject,
                } => {
                    if decide {
                        let decision = if bounds_loop(accept, reject) {
                            Decision::Loop
                        } else {
                            Decision::If
                        };
                        self.condition(frame, decision, condition);
                    }
                    grew |= self.walk(frame, accept, decide);
                    grew |= self.walk(frame, reject, decide);
                }
                Statement::Switch {
                    selector,
                    ref cases,
                } => {
                    if decide {
                        self.condition(frame, Decision::Switch, selector);
                    }
                    for case in cases {
                        grew |= self.walk(frame, &case.body, decide);
                    }
                }
                Statement::Loop {
                    ref body,
                    ref continuing,
                    break_if,
                } => {
                    grew |= self.walk(frame, body, decide);
                    grew |= self.walk(frame, continuing, decide);
                    if let (true, Some(condition)) = (decide, break_if) {
                        self.condition(frame, Decision::BreakIf, condition);
                    }
                }
                Statement::Return { value: Some(value) } => {
                    let returned = frame.value(value);
                    grew |= frame.result.join(&returned);
                }
                Statement::Store { pointer, value } => {
                    if let Slot::Place(place) = frame.slot(pointer).clone() {
                        let stored = frame.value(value);
                        grew |= self.write(frame, &place, &stored);
                    }
                }
                Statement::Call {
                    function,
                    ref arguments,
                    result,
                } => grew |= self.call(frame, function, arguments, result, *span, decide),
                Statement::WorkGroupUniformLoad { pointer, result } => {
                    if let Slot::Place(place) = frame.slot(pointer).clone() {
                        let loaded = self.read(frame, &place, *span);
                        let held = frame.calls.entry(result).or_insert_with(Value::clean);
                        grew |= held.join(&loaded);
                    }
                }
                _ => {}
            }
        }
        grew
    }

    fn call(
        &mut self,
        frame: &mut Frame<'_>,
        callee: Handle<Function>,
        arguments: &[Handle<Expression>],
        result: Option<Handle<Expression>>,
        span: Span,
        decide: bool,
    ) -> bool {
        let Some(summary) = self.summaries.get(callee.index()).cloned().flatten() else {
            return false;
        };
        let mut passed = Vec::with_capacity(arguments.len());
        for &argument in arguments {
            let value = match frame.slot(argument).clone() {
                Slot::Place(place) => self.read(frame, &place, span),
                Slot::Value(value) => value,
            };
            passed.push(value);
        }
        let mut grew = false;
        if let Some(result) = result {
            let value = summary
                .result
                .map(&mut |taint| substitute(&summary, &passed, taint));
            let held = frame.calls.entry(result).or_insert_with(Value::clean);
            grew |= held.join(&value);
        }
        for (pointee, &argument) in summary.pointees.iter().zip(arguments) {
            let Some(pointee) = pointee else {
                continue;
            };
            if let Slot::Place(place) = frame.slot(argument).clone() {
                let value = pointee.map(&mut |taint| substitute(&summary, &passed, taint));
                grew |= self.write(frame, &place, &value);
            }
        }
        for (&global, value) in &summary.globals {
            let value = value.map(&mut |taint| substitute(&summary, &passed, taint));
            grew |= self.store_global(frame, global, &[], &value);
        }
        if decide {
            let call = Call {
                function: frame.name.clone(),
                span,
            };
            for (&sink, (taint, via)) in &summary.pending {
                let taint = substitute(&summary, &passed, taint);
                let mut via = via.clone();
                via.push(call.clone());
                self.feed(frame, sink, &taint, &via);
            }
        }
        grew
    }

    fn write(&mut self, frame: &mut Frame<'_>, place: &Place, value: &Value) -> bool {
        let value = if place.index.is_empty() {
            value.clone()
        } else {
            value.with(&place.index)
        };
        match place.root {
            Root::Local(local) => frame.locals[local].join_at(&place.steps, &value),
            Root::Argument(argument) => match frame.pointees.get_mut(argument) {
                Some(Some(pointee)) => pointee.join_at(&place.steps, &value),
                _ => false,
            },
            Root::Global(global) => self.store_global(frame, global, &place.steps, &value),
        }
    }

    fn store_global(
        &mut self,
        frame: &mut Frame<'_>,
        global: Handle<GlobalVariable>,
        steps: &[Step],
        value: &Value,
    ) -> bool {
        let module = self.module;
        let variable: &GlobalVariable = &module.global_variables[global];
        if !matches!(
            variable.space,
            AddressSpace::Private | AddressSpace::WorkGroup
        ) {
            return false;
        }
        let reads = value.map(&mut |taint| {
            taint
                .iter()
                .filter(|origin| matches!(origin, Origin::Source(_)))
                .copied()
                .collect()
        });
        let shared = self
            .globals
            .entry(global)
            .or_insert_with(|| clean(module, variable.ty));
        let mut grew = shared.join_at(steps, &reads);
        let own = frame
            .globals
            .entry(global)
            .or_insert_with(|| clean(module, variable.ty));
        grew |= own.join_at(steps, value);
        grew
    }

    pub(super) fn expression_decisions(&mut self, frame: &mut Frame<'_>) {
        let function = frame.function;
        for (handle, expression) in function.expressions.iter() {
            let (decision, operands): (Decision, Vec<Handle<Expression>>) = match *expression {
                Expression::Select { condition, .. } => (Decision::Select, vec![condition]),
                Expression::Math {
                    fun: MathFunction::Saturate,
                    arg,
                    ..
                } if self.watch == Watch::Sizes => (Decision::Saturate, vec![arg]),
                Expression::Math {
                    fun: MathFunction::SmoothStep,
                    arg,
                    arg1,
                    arg2,
                    ..
                } => (
                    Decision::Smoothstep,
                    [Some(arg), arg1, arg2].into_iter().flatten().collect(),
                ),
                Expression::Math {
                    fun: MathFunction::Step,
                    arg,
                    arg1,
                    ..
                } => (
                    Decision::Step,
                    [Some(arg), arg1].into_iter().flatten().collect(),
                ),
                Expression::Math {
                    fun: MathFunction::Mix,
                    arg2: Some(weight),
                    ..
                } => (Decision::MixWeight, vec![weight]),
                Expression::As { expr, kind, .. }
                    if kind != ScalarKind::Bool && self.is_bool(frame, expr) =>
                {
                    (Decision::Number, vec![expr])
                }
                _ => continue,
            };
            let taint: Taint = operands
                .iter()
                .flat_map(|&operand| frame.taint(operand))
                .collect();
            let span = function.expressions.get_span(handle);
            self.decide(frame, decision, handle.index(), span, &taint);
        }
    }

    fn condition(
        &mut self,
        frame: &mut Frame<'_>,
        decision: Decision,
        condition: Handle<Expression>,
    ) {
        let span = frame.function.expressions.get_span(condition);
        let taint = frame.taint(condition);
        self.decide(frame, decision, condition.index(), span, &taint);
    }

    fn decide(
        &mut self,
        frame: &mut Frame<'_>,
        decision: Decision,
        key: usize,
        span: Span,
        taint: &Taint,
    ) {
        let id = (frame.of, key, decision);
        self.decided.insert(id);
        if taint.is_empty() || (self.marked)(span) {
            return;
        }
        let sink = if let Some(&sink) = self.sink_ids.get(&id) {
            sink
        } else {
            self.sinks.push(Sink {
                decision,
                function: frame.name.clone(),
                span,
            });
            self.sink_ids.insert(id, self.sinks.len() - 1);
            self.sinks.len() - 1
        };
        self.feed(frame, sink, taint, &[]);
    }

    fn feed(&mut self, frame: &mut Frame<'_>, sink: usize, taint: &Taint, via: &[Call]) {
        for &origin in taint {
            match origin {
                Origin::Source(source) => {
                    self.findings
                        .entry((sink, source))
                        .or_insert_with(|| via.to_vec());
                }
                Origin::Part(_) => {
                    let (pending, _) = frame
                        .pending
                        .entry(sink)
                        .or_insert_with(|| (Taint::new(), via.to_vec()));
                    pending.insert(origin);
                }
            }
        }
    }
}

fn substitute(summary: &Summary, passed: &[Value], taint: &Taint) -> Taint {
    let mut out = Taint::new();
    for &origin in taint {
        match origin {
            Origin::Source(_) => {
                out.insert(origin);
            }
            Origin::Part(part) => {
                if let Some((argument, path)) = summary.parts.get(part)
                    && let Some(value) = passed.get(*argument)
                {
                    out.extend(value.at(path).collapse());
                }
            }
        }
    }
    out
}
