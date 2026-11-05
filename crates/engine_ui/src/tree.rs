use std::ops::Range;

use bevy_ecs::prelude::*;
use engine_core::math::{clamp, max, min};
use engine_core::{CursorMode, SceneContext, TextureHandle, View};
use engine_places::text::{FontId, Text, TextRun};
use engine_places::ui::{
    UiAction, UiAlign, UiDirection, UiInsets, UiJustify, UiKind, UiLength, UiNode, UiRoot, UiValue,
    UiValues,
};
use glam::Vec2;
use taffy::{
    AlignItems, AvailableSpace, Dimension, Display, FlexDirection, JustifyContent,
    LengthPercentage, LengthPercentageAuto, NodeId, Overflow, Point, Position, Rect, Size, Style,
    TaffyError, TaffyTree, compute_leaf_layout,
};

use crate::{UiRect, UiSettings, UiStats};

pub(crate) type Roots<'w, 's> = Query<'w, 's, (Entity, Ref<'static, UiRoot>)>;

pub(crate) struct Facts<'a> {
    enabled: bool,
    metrics: [f32; 4],
    max_nodes: usize,
    stamps: [u32; 2],
    top: Option<&'a str>,
    text: &'a Text,
    values: &'a UiValues,
    scene: Option<&'a SceneContext>,
}

impl<'a> Facts<'a> {
    pub(crate) fn of(
        text: &'a Res<'_, Text>,
        values: &'a UiValues,
        view: &View,
        settings: &UiSettings,
        scene: Option<&'a Res<'_, SceneContext>>,
        top: Option<&'a str>,
    ) -> Self {
        let display = view.display_size().as_vec2();
        Self {
            enabled: settings.enabled,
            metrics: [display.x, display.y, view.pixel_ratio(), settings.scale],
            max_nodes: settings.max_nodes as usize,
            stamps: [
                text.last_changed().get(),
                scene.map_or(0, |scene| scene.last_changed().get()),
            ],
            top,
            text,
            values,
            scene: scene.map(|scene| &**scene),
        }
    }
}

pub(crate) struct Flat {
    pub(crate) node: UiNode,
    pub(crate) parent: Option<u32>,
    pub(crate) texture: Option<TextureHandle>,
    pub(crate) text: String,
    pub(crate) fill: f32,
    pub(crate) at: [f32; 4],
    pub(crate) clip: Option<[f32; 4]>,
    pub(crate) step: u32,
    root: Entity,
    font: Option<FontId>,
    children: Vec<u32>,
    rect: [f32; 4],
    measured: [f32; 2],
    scroll: f32,
    extent: f32,
    offset: [f32; 2],
    unbound: u32,
}

struct Bound {
    text: bool,
    moved: bool,
    image: bool,
}

impl Flat {
    fn of(
        node: UiNode,
        parent: Option<u32>,
        root: Entity,
        font: Option<FontId>,
        texture: Option<TextureHandle>,
    ) -> Self {
        let text = if matches!(node.kind, UiKind::Label | UiKind::Button) {
            node.text.clone()
        } else {
            String::new()
        };
        let fill = unit(node.bar_value);
        Self {
            node,
            parent,
            texture,
            text,
            fill,
            at: [0.0; 4],
            clip: None,
            step: 0,
            root,
            font,
            children: Vec::new(),
            rect: [0.0; 4],
            measured: [0.0; 2],
            scroll: 0.0,
            extent: 0.0,
            offset: [0.0; 2],
            unbound: 0,
        }
    }

    fn shows_text(&self) -> bool {
        matches!(self.node.kind, UiKind::Label | UiKind::Button) && !self.text.is_empty()
    }

    fn missing(&self) -> u32 {
        self.unbound + u32::from(self.node.kind == UiKind::Image && self.texture.is_none())
    }

    fn bind(&mut self, values: &UiValues, scene: Option<&SceneContext>) -> Bound {
        self.unbound = 0;
        self.bind_color(values);
        Bound {
            text: self.bind_value(values),
            moved: self.bind_offset(values),
            image: self.bind_image(values, scene),
        }
    }

    fn bind_value(&mut self, values: &UiValues) -> bool {
        let Some(name) = self.node.bind.value.as_deref() else {
            return false;
        };
        let value = values.get(name);
        match self.node.kind {
            UiKind::Bar => {
                if let Some(UiValue::Number(number)) = value {
                    self.fill = unit(number);
                } else {
                    self.unbound += 1;
                    self.fill = unit(self.node.bar_value);
                }
                false
            }
            UiKind::Label | UiKind::Button => {
                let shown = shown_text(value);
                self.unbound += u32::from(shown.is_none());
                let text = shown.unwrap_or_else(|| {
                    if self.node.text.is_empty() {
                        name.to_owned()
                    } else {
                        self.node.text.clone()
                    }
                });
                let changed = text != self.text;
                self.text = text;
                changed
            }
            UiKind::Panel | UiKind::Image | UiKind::List => false,
        }
    }

    fn bind_offset(&mut self, values: &UiValues) -> bool {
        let Some(name) = self.node.bind.offset.as_deref() else {
            return false;
        };
        if let Some(UiValue::Offset(offset)) = values.get(name)
            && finite(&offset)
        {
            let moved = offset != self.offset;
            self.offset = offset;
            moved
        } else {
            self.unbound += 1;
            false
        }
    }

    fn bind_color(&mut self, values: &UiValues) {
        let Some(name) = self.node.bind.color.as_deref() else {
            return;
        };
        if let Some(UiValue::Color(color)) = values.get(name)
            && finite(&color)
        {
            match self.node.kind {
                UiKind::Label => self.node.text_color = color,
                UiKind::Bar => self.node.bar_color = color,
                UiKind::Panel | UiKind::Image | UiKind::Button | UiKind::List => {
                    self.node.color = color;
                }
            }
        } else {
            self.unbound += 1;
        }
    }

    fn bind_image(&mut self, values: &UiValues, scene: Option<&SceneContext>) -> bool {
        if self.node.kind != UiKind::Image {
            return false;
        }
        let Some(name) = self.node.bind.image.as_deref() else {
            return false;
        };
        if let Some(UiValue::Image { image, rect }) = values.get(name)
            && finite(&[rect.u, rect.v, rect.width, rect.height])
        {
            self.node.image = Some(image);
            self.node.image_rect = rect;
            let texture = texture_of(&self.node, scene);
            let changed = texture != self.texture;
            self.texture = texture;
            changed
        } else {
            self.unbound += u32::from(self.texture.is_some());
            false
        }
    }

    fn content(&self, scale: f32) -> [f32; 4] {
        let [left, top, right, bottom] = self.node.padding;
        let edge = self.node.border_width;
        [
            self.at[0] + (left + edge) * scale,
            self.at[1] + (top + edge) * scale,
            self.at[2] - (left + right + edge * 2.0) * scale,
            self.at[3] - (top + bottom + edge * 2.0) * scale,
        ]
    }

    fn measured_at(&self, text: &Text, limit: Option<f32>, scale: f32) -> [f32; 2] {
        let size = text.0.measure(&TextRun {
            text: self.text.clone(),
            font: self.font,
            size: self.node.text_size * scale,
            max_width: limit,
            align: self.node.text_align,
            ..TextRun::default()
        });
        [size.width.ceil(), size.height.ceil()]
    }

    fn fit(&self, text: &Text, scale: f32) -> [f32; 2] {
        let width = self.content(scale)[2];
        self.measured_at(text, (width > 0.0).then_some(width), scale)
    }

    fn run(&self, scale: f32) -> TextRun {
        let [x, y, width, height] = self.content(scale);
        TextRun {
            text: self.text.clone(),
            font: self.font,
            size: self.node.text_size * scale,
            color: self.node.text_color,
            position: [x, y + (height - self.measured[1]) * 0.5],
            max_width: (width > 0.0).then_some(width),
            align: self.node.text_align,
            clip: self.clip,
            step: Some(self.step),
        }
    }

    fn most_scroll(&self) -> f32 {
        max(
            self.extent - self.rect[axis_of(self.node.direction) + 2],
            0.0,
        )
    }

    fn passed_down(&self) -> ([f32; 2], Option<[f32; 4]>) {
        let mut shift = [self.at[0] - self.rect[0], self.at[1] - self.rect[1]];
        if self.node.kind != UiKind::List {
            return (shift, self.clip);
        }
        shift[axis_of(self.node.direction)] -= self.scroll;
        let clip = self.clip.map_or(self.at, |clip| meet(clip, self.at));
        (shift, Some(clip))
    }

    fn under(&self, point: Vec2) -> bool {
        inside(self.at, point) && self.clip.is_none_or(|clip| inside(clip, point))
    }

    fn solid(&self) -> bool {
        let node = &self.node;
        !matches!(node.kind, UiKind::Panel | UiKind::Label)
            || node.color[3] > 0.0
            || (node.border_width > 0.0 && node.border_color[3] > 0.0)
    }

    pub(crate) fn center(&self) -> Vec2 {
        Vec2::new(self.at[0] + self.at[2] * 0.5, self.at[1] + self.at[3] * 0.5)
    }
}

struct Mark {
    root: Entity,
    id: Option<String>,
    kind: UiKind,
    action: Option<UiAction>,
    path: Vec<usize>,
}

#[derive(Resource, Default)]
pub(crate) struct UiState {
    pub(crate) nodes: Vec<Flat>,
    pub(crate) metrics: [f32; 4],
    pub(crate) hovered: Option<u32>,
    pub(crate) pressed: Option<u32>,
    pub(crate) focused: Option<u32>,
    pub(crate) pending: Vec<String>,
    pub(crate) menu: bool,
    pub(crate) wanting: bool,
    pub(crate) held_cursor: Option<CursorMode>,
    pub(crate) textures: Vec<TextureHandle>,
    pub(crate) rects: Vec<UiRect>,
    pub(crate) runs: Vec<(u32, Option<TextureHandle>, Range<u32>)>,
    pub(crate) rects_cut: u32,
    pub(crate) events: u32,
    pub(crate) shown_roots: u32,
    seen: Vec<(Entity, u32)>,
    stamps: [u32; 2],
    top: Option<String>,
    values: u64,
    stale: bool,
    nodes_cut: u32,
    layouts: u32,
    missing: u32,
}

impl UiState {
    pub(crate) fn is_idle(&self) -> bool {
        self.seen.is_empty() && self.nodes.is_empty() && !self.wanting
    }

    pub(crate) fn stats(&self) -> UiStats {
        UiStats {
            roots: self.shown_roots,
            nodes: self.nodes.len() as u32,
            rects_drawn: self.rects.len() as u32,
            rects_cut: self.rects_cut + self.nodes_cut,
            layouts: self.layouts,
            events: self.events,
            missing_bindings: self.missing,
        }
    }

    pub(crate) fn sync(&mut self, roots: &Roots<'_, '_>, facts: &Facts<'_>) {
        if !self.same_roots(roots, facts.enabled)
            || self.stamps != facts.stamps
            || self.top.as_deref() != facts.top
        {
            self.rebuild(roots, facts);
        }
        self.refresh_values(facts);
        if self.stale || self.metrics != facts.metrics {
            self.metrics = facts.metrics;
            self.stale = false;
            self.lay_out(facts);
        }
    }

    pub(crate) fn named(&self, id: &str) -> Option<u32> {
        self.nodes
            .iter()
            .position(|flat| flat.node.id.as_deref() == Some(id))
            .map(|at| at as u32)
    }

    pub(crate) fn top_at(&self, point: Vec2, kind: UiKind) -> Option<u32> {
        self.nodes
            .iter()
            .rposition(|flat| flat.node.kind == kind && flat.under(point))
            .map(|at| at as u32)
    }

    pub(crate) fn over(&self, point: Vec2) -> bool {
        self.nodes
            .iter()
            .any(|flat| flat.solid() && flat.under(point))
    }

    pub(crate) fn scroll_by(&mut self, at: u32, by: f32) {
        let list = &mut self.nodes[at as usize];
        let scroll = clamp(list.scroll + by, 0.0, list.most_scroll());
        if scroll != list.scroll {
            list.scroll = scroll;
            self.place();
        }
    }

    pub(crate) fn reveal(&mut self, at: u32) {
        let shown = self.nodes[at as usize].rect;
        let mut holder = self.nodes[at as usize].parent;
        let mut moved = false;
        while let Some(index) = holder {
            let list = &mut self.nodes[index as usize];
            holder = list.parent;
            if list.node.kind != UiKind::List {
                continue;
            }
            let axis = axis_of(list.node.direction);
            let start = shown[axis] - list.rect[axis];
            let end = start + shown[axis + 2];
            let view = list.rect[axis + 2];
            let scroll = if start < list.scroll {
                start
            } else if end > list.scroll + view {
                end - view
            } else {
                list.scroll
            };
            let scroll = clamp(scroll, 0.0, list.most_scroll());
            moved |= scroll != list.scroll;
            list.scroll = scroll;
        }
        if moved {
            self.place();
        }
    }

    pub(crate) fn queue_text(&self, text: &Text) {
        let scale = self.metrics[3];
        for flat in &self.nodes {
            let in_view = flat.clip.is_none_or(|clip| clip[2] > 0.0 && clip[3] > 0.0);
            if flat.shows_text() && in_view {
                text.0.queue(flat.run(scale));
            }
        }
    }

    fn same_roots(&self, roots: &Roots<'_, '_>, enabled: bool) -> bool {
        if !enabled {
            return self.seen.is_empty();
        }
        roots.iter().count() == self.seen.len()
            && roots.iter().zip(&self.seen).all(|((entity, root), seen)| {
                entity == seen.0 && root.last_changed().get() == seen.1
            })
    }

    fn mark(&self, at: u32) -> Mark {
        let flat = &self.nodes[at as usize];
        let mut path = Vec::new();
        let mut child = at;
        while let Some(parent) = self.nodes[child as usize].parent {
            let siblings = &self.nodes[parent as usize].children;
            path.extend(siblings.iter().position(|&sibling| sibling == child));
            child = parent;
        }
        Mark {
            root: flat.root,
            id: flat.node.id.clone(),
            kind: flat.node.kind,
            action: flat.node.action.clone(),
            path,
        }
    }

    fn find(&self, mark: &Mark) -> Option<u32> {
        let at = if mark.id.is_some() {
            self.nodes
                .iter()
                .position(|flat| flat.root == mark.root && flat.node.id == mark.id)?
        } else {
            let start = self.nodes.iter().position(|flat| flat.root == mark.root)?;
            mark.path.iter().rev().try_fold(start, |at, &step| {
                let child = self.nodes[at].children.get(step)?;
                Some(*child as usize)
            })?
        };
        let node = &self.nodes[at].node;
        (node.kind == mark.kind && (mark.id.is_some() || node.action == mark.action))
            .then_some(at as u32)
    }

    fn rebuild(&mut self, roots: &Roots<'_, '_>, facts: &Facts<'_>) {
        let held = [self.hovered, self.pressed, self.focused].map(|at| at.map(|at| self.mark(at)));
        let scrolled: Vec<(Mark, f32)> = (0..self.nodes.len() as u32)
            .filter(|&at| self.nodes[at as usize].scroll != 0.0)
            .map(|at| (self.mark(at), self.nodes[at as usize].scroll))
            .collect();
        self.nodes.clear();
        self.seen.clear();
        self.menu = false;
        self.shown_roots = 0;
        self.nodes_cut = 0;
        self.stamps = facts.stamps;
        self.top = facts.top.map(str::to_owned);
        self.stale = true;
        if facts.enabled {
            self.seen.extend(
                roots
                    .iter()
                    .map(|(entity, root)| (entity, root.last_changed().get())),
            );
            let mut shown: Vec<_> = roots
                .iter()
                .filter(|(_, root)| root.screen.is_none() || root.screen.as_deref() == facts.top)
                .collect();
            shown.sort_by_key(|(entity, root)| (root.order, *entity));
            self.menu = shown.iter().any(|(_, root)| root.menu);
            for (entity, root) in shown {
                let font = root
                    .bytes
                    .as_ref()
                    .and_then(|bytes| facts.text.0.register_font(bytes).ok());
                let first = self.nodes.len();
                self.flatten(&root.root, None, entity, font, facts);
                for flat in &mut self.nodes[first..] {
                    flat.step = self.shown_roots;
                }
                self.shown_roots += 1;
            }
        }
        [self.hovered, self.pressed, self.focused] =
            held.map(|mark| mark.and_then(|mark| self.find(&mark)));
        for (mark, scroll) in scrolled {
            if let Some(at) = self.find(&mark) {
                self.nodes[at as usize].scroll = scroll;
            }
        }
        self.values = facts.values.revision();
        self.missing = 0;
        for flat in &mut self.nodes {
            flat.bind(facts.values, facts.scene);
            self.missing += flat.missing();
        }
        self.list_textures();
    }

    fn list_textures(&mut self) {
        self.textures.clear();
        self.textures
            .extend(self.nodes.iter().filter_map(|flat| flat.texture));
        self.textures.sort_unstable();
        self.textures.dedup();
    }

    fn flatten(
        &mut self,
        node: &UiNode,
        parent: Option<u32>,
        root: Entity,
        font: Option<FontId>,
        facts: &Facts<'_>,
    ) {
        if self.nodes.len() >= facts.max_nodes {
            self.nodes_cut += weight(node);
            return;
        }
        let at = self.nodes.len() as u32;
        let texture = texture_of(node, facts.scene);
        let flat = Flat::of(without_children(node), parent, root, font, texture);
        self.nodes.push(flat);
        if let Some(parent) = parent {
            self.nodes[parent as usize].children.push(at);
        }
        for child in &node.children {
            self.flatten(child, Some(at), root, font, facts);
        }
    }

    fn refresh_values(&mut self, facts: &Facts<'_>) {
        let revision = facts.values.revision();
        if revision == self.values {
            return;
        }
        self.values = revision;
        self.missing = 0;
        let scale = self.metrics[3];
        let (mut moved, mut images) = (false, false);
        for flat in &mut self.nodes {
            let bound = flat.bind(facts.values, facts.scene);
            if bound.text {
                let size = flat.fit(facts.text, scale);
                let fixed = !matches!(flat.node.width, UiLength::Auto)
                    && !matches!(flat.node.height, UiLength::Auto);
                self.stale |= !fixed && size != flat.measured;
                flat.measured = size;
            }
            moved |= bound.moved;
            images |= bound.image;
            self.missing += flat.missing();
        }
        if moved {
            self.place();
        }
        if images {
            self.list_textures();
        }
    }

    fn lay_out(&mut self, facts: &Facts<'_>) {
        if self.nodes.is_empty() {
            return;
        }
        self.layouts += 1;
        let _ = self.solve(facts);
        self.settle_lists();
        self.place();
        let scale = self.metrics[3];
        for flat in &mut self.nodes {
            if flat.shows_text() {
                flat.measured = flat.fit(facts.text, scale);
            }
        }
    }

    fn solve(&mut self, facts: &Facts<'_>) -> Result<(), TaffyError> {
        let [display_x, display_y, ratio, scale] = self.metrics;
        let ratio = max(ratio, f32::EPSILON);
        let (width, height) = (display_x / ratio, display_y / ratio);
        let count = self.nodes.len();
        let mut tree: TaffyTree<u32> = TaffyTree::with_capacity(count * 2);
        let mut ids = vec![NodeId::new(0); count];
        for at in (0..count).rev() {
            let flat = &self.nodes[at];
            let listed = flat
                .parent
                .is_some_and(|parent| self.nodes[parent as usize].node.kind == UiKind::List);
            let style = style_of(&flat.node, scale, listed);
            ids[at] = if flat.children.is_empty() {
                tree.new_leaf_with_context(style, at as u32)?
            } else {
                let children: Vec<NodeId> = flat
                    .children
                    .iter()
                    .map(|&child| ids[child as usize])
                    .collect();
                tree.new_with_children(style, &children)?
            };
        }
        let space = Size {
            width: AvailableSpace::Definite(width),
            height: AvailableSpace::Definite(height),
        };
        for (at, &id) in ids.iter().enumerate() {
            if self.nodes[at].parent.is_some() {
                continue;
            }
            let window = tree.new_with_children(window_style(width, height), &[id])?;
            let nodes = &self.nodes;
            tree.compute_layout_with_measure(window, space, |inputs, _, context, style| {
                compute_leaf_layout(
                    inputs,
                    style,
                    |_, _| 0.0,
                    |_, available| {
                        context.map_or(Size::ZERO, |at| {
                            text_size(&nodes[*at as usize], available, facts.text, scale)
                        })
                    },
                )
            })?;
        }
        for (at, &id) in ids.iter().enumerate() {
            let layout = tree.layout(id)?;
            let origin = self.nodes[at]
                .parent
                .map_or([0.0; 4], |parent| self.nodes[parent as usize].rect);
            self.nodes[at].rect = [
                origin[0] + layout.location.x,
                origin[1] + layout.location.y,
                layout.size.width,
                layout.size.height,
            ];
        }
        Ok(())
    }

    fn settle_lists(&mut self) {
        let scale = self.metrics[3];
        for at in 0..self.nodes.len() {
            if self.nodes[at].node.kind != UiKind::List {
                continue;
            }
            let axis = axis_of(self.nodes[at].node.direction);
            let start = self.nodes[at].rect[axis];
            let end = self.nodes[at].children.iter().fold(start, |end, &child| {
                let rect = self.nodes[child as usize].rect;
                max(end, rect[axis] + rect[axis + 2])
            });
            let list = &mut self.nodes[at];
            list.extent = end - start + list.node.padding[axis + 2] * scale;
            list.scroll = clamp(list.scroll, 0.0, list.most_scroll());
        }
    }

    fn place(&mut self) {
        for at in 0..self.nodes.len() {
            let (shift, clip) = self.nodes[at].parent.map_or(([0.0; 2], None), |parent| {
                self.nodes[parent as usize].passed_down()
            });
            let flat = &mut self.nodes[at];
            flat.at = [
                flat.rect[0] + shift[0] + flat.offset[0],
                flat.rect[1] + shift[1] + flat.offset[1],
                flat.rect[2],
                flat.rect[3],
            ];
            flat.clip = clip;
        }
    }
}

fn unit(value: f32) -> f32 {
    if value.is_finite() {
        clamp(value, 0.0, 1.0)
    } else {
        0.0
    }
}

fn finite(numbers: &[f32]) -> bool {
    numbers.iter().all(|number| number.is_finite())
}

fn shown_text(value: Option<UiValue>) -> Option<String> {
    match value {
        Some(UiValue::Text(text)) => Some(text),
        Some(UiValue::Number(number)) => Some(number.to_string()),
        Some(UiValue::Offset(_) | UiValue::Color(_) | UiValue::Image { .. }) | None => None,
    }
}

fn axis_of(direction: UiDirection) -> usize {
    match direction {
        UiDirection::Row => 0,
        UiDirection::Column => 1,
    }
}

fn inside([x, y, width, height]: [f32; 4], point: Vec2) -> bool {
    point.x >= x && point.y >= y && point.x < x + width && point.y < y + height
}

fn meet(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let left = max(a[0], b[0]);
    let top = max(a[1], b[1]);
    let right = min(a[0] + a[2], b[0] + b[2]);
    let bottom = min(a[1] + a[3], b[1] + b[3]);
    [left, top, max(right - left, 0.0), max(bottom - top, 0.0)]
}

fn weight(node: &UiNode) -> u32 {
    1 + node.children.iter().map(weight).sum::<u32>()
}

fn without_children(node: &UiNode) -> UiNode {
    UiNode {
        id: node.id.clone(),
        text: node.text.clone(),
        bind: node.bind.clone(),
        image: node.image.clone(),
        action: node.action.clone(),
        children: Vec::new(),
        ..*node
    }
}

fn texture_of(node: &UiNode, scene: Option<&SceneContext>) -> Option<TextureHandle> {
    if node.kind != UiKind::Image {
        return None;
    }
    let name = node.image.as_deref()?;
    scene?.assets.textures.get(name).copied()
}

fn text_size(flat: &Flat, available: Size<AvailableSpace>, text: &Text, scale: f32) -> Size<f32> {
    if !flat.shows_text() {
        return Size::ZERO;
    }
    let limit = match available.width {
        AvailableSpace::Definite(width) => Some(width),
        AvailableSpace::MinContent | AvailableSpace::MaxContent => None,
    };
    let [width, height] = flat.measured_at(text, limit, scale);
    Size { width, height }
}

fn window_style(width: f32, height: f32) -> Style {
    Style {
        display: Display::Flex,
        position: Position::Relative,
        size: Size {
            width: Dimension::length(width),
            height: Dimension::length(height),
        },
        flex_direction: FlexDirection::Column,
        align_items: AlignItems::START,
        justify_content: JustifyContent::START,
        ..Style::default()
    }
}

fn style_of(node: &UiNode, scale: f32, listed: bool) -> Style {
    let overflow = if node.kind == UiKind::List {
        Overflow::Hidden
    } else {
        Overflow::Visible
    };
    let gap = LengthPercentage::length(node.gap * scale);
    Style {
        display: Display::Flex,
        position: if node.absolute.is_some() {
            Position::Absolute
        } else {
            Position::Relative
        },
        inset: inset_of(node.absolute.unwrap_or_default(), scale),
        overflow: Point {
            x: overflow,
            y: overflow,
        },
        size: Size {
            width: dimension_of(node.width, scale),
            height: dimension_of(node.height, scale),
        },
        margin: margin_of(node.margin, scale),
        padding: padding_of(node.padding, scale),
        border: padding_of([node.border_width; 4], scale),
        gap: Size {
            width: gap,
            height: gap,
        },
        flex_direction: match node.direction {
            UiDirection::Row => FlexDirection::Row,
            UiDirection::Column => FlexDirection::Column,
        },
        flex_grow: node.grow,
        flex_shrink: if listed { 0.0 } else { 1.0 },
        align_items: items_of(node.align),
        justify_content: justify_of(node.justify),
        ..Style::default()
    }
}

fn dimension_of(length: UiLength, scale: f32) -> Dimension {
    match length {
        UiLength::Auto => Dimension::auto(),
        UiLength::Px(pixels) => Dimension::length(pixels * scale),
        UiLength::Percent(percent) => Dimension::percent(percent * 0.01),
    }
}

fn inset_of(insets: UiInsets, scale: f32) -> Rect<LengthPercentageAuto> {
    let side = |value: Option<f32>| {
        value.map_or_else(LengthPercentageAuto::auto, |value| {
            LengthPercentageAuto::length(value * scale)
        })
    };
    Rect {
        left: side(insets.left),
        right: side(insets.right),
        top: side(insets.top),
        bottom: side(insets.bottom),
    }
}

fn margin_of([left, top, right, bottom]: [f32; 4], scale: f32) -> Rect<LengthPercentageAuto> {
    let side = |value: f32| LengthPercentageAuto::length(value * scale);
    Rect {
        left: side(left),
        right: side(right),
        top: side(top),
        bottom: side(bottom),
    }
}

fn padding_of([left, top, right, bottom]: [f32; 4], scale: f32) -> Rect<LengthPercentage> {
    let side = |value: f32| LengthPercentage::length(value * scale);
    Rect {
        left: side(left),
        right: side(right),
        top: side(top),
        bottom: side(bottom),
    }
}

fn items_of(align: UiAlign) -> AlignItems {
    match align {
        UiAlign::Start => AlignItems::START,
        UiAlign::Center => AlignItems::CENTER,
        UiAlign::End => AlignItems::END,
        UiAlign::Stretch => AlignItems::STRETCH,
    }
}

fn justify_of(justify: UiJustify) -> JustifyContent {
    match justify {
        UiJustify::Start => JustifyContent::START,
        UiJustify::Center => JustifyContent::CENTER,
        UiJustify::End => JustifyContent::END,
        UiJustify::SpaceBetween => JustifyContent::SPACE_BETWEEN,
        UiJustify::SpaceAround => JustifyContent::SPACE_AROUND,
        UiJustify::SpaceEvenly => JustifyContent::SPACE_EVENLY,
    }
}
