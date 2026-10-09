//! Virtual geometry: a mesh as a cluster level-of-detail
//! hierarchy in a GPU-ready layout.
//!
//! An `EVG1` file is little-endian 32-bit words: a header, the bounding
//! volume hierarchy's nodes, the hierarchy's groups of clusters, the
//! clusters, a page table and the pages holding the clusters' vertices and
//! triangles. Renderers upload the sections as they are; decoding only
//! validates them. Offline tools build the hierarchy (`tools/vgbuild`).
//!
//! The first pages are pinned: they hold every group simplified with more
//! error than the pin error, which covers every surface coarsely. A file's
//! prefix, its metadata and pinned pages, decodes on its own; renderers
//! stream the other pages from the file as views need them, each validated
//! as it arrives ([`VirtualMesh::validate_page`]).
//!
//! A cluster holds up to [`CLUSTER_TRIANGLES`] triangles over up to
//! [`CLUSTER_VERTICES`] vertices. Groups of clusters were simplified together
//! into the next level's clusters; each cluster records the group it belongs
//! to (whose simplification error decides when a coarser cluster replaces it)
//! and the finer group it came from (its own error). A view draws a cluster
//! when its group's error projects above the view's threshold and its own at
//! or below it ([`VirtualMesh::cut`] in mesh space).

use glam::Vec3;

use crate::{AssetError, Mesh, MeshVertex, mesh::Aabb};

/// Magic of virtual mesh files.
pub const VIRTUAL_MESH_MAGIC: [u8; 4] = *b"EVG1";
/// Format version this build reads and writes.
pub const VIRTUAL_MESH_VERSION: u32 = 2;
/// Triangles per cluster at most: vertex shaders draw three vertices of each.
pub const CLUSTER_TRIANGLES: u32 = 128;
/// Vertices per cluster at most, addressed by 8-bit local indices.
pub const CLUSTER_VERTICES: u32 = 255;
/// Words of the header.
pub const HEADER_WORDS: usize = 32;
/// Words a page holds at most: 128 KiB, the unit renderers stream.
pub const PAGE_WORDS: usize = 32 * 1024;
/// Words of a hierarchy node: bounding sphere, largest error in its
/// subtree, and its group or children.
pub const NODE_WORDS: usize = 8;
/// Words of a group: its simplification's sphere and error, its clusters
/// and its level.
pub const GROUP_WORDS: usize = 8;
/// Words of a cluster record.
pub const CLUSTER_WORDS: usize = 12;
/// Words of a stored vertex: 16-bit position steps, an octahedral normal
/// and 16-bit texture coordinates.
pub const VERTEX_WORDS: usize = 4;
/// Marks internal hierarchy nodes, clusters of original geometry and groups
/// never simplified further.
pub const NONE: u32 = u32::MAX;

/// Header word offsets.
mod header {
    pub const MAGIC: usize = 0;
    pub const VERSION: usize = 1;
    pub const FLAGS: usize = 2;
    pub const LEVELS: usize = 3;
    pub const NODES: usize = 4;
    pub const GROUPS: usize = 5;
    pub const CLUSTERS: usize = 6;
    pub const PAGES: usize = 7;
    pub const TRIANGLES: usize = 8;
    pub const SOURCE_TRIANGLES: usize = 9;
    pub const DEPTH: usize = 10;
    pub const POSITION_MIN: usize = 12;
    pub const POSITION_STEP: usize = 15;
    pub const UV_MIN: usize = 18;
    pub const UV_STEP: usize = 20;
    pub const BOUNDS_MIN: usize = 22;
    pub const BOUNDS_MAX: usize = 25;
    pub const PINNED_PAGES: usize = 28;
    pub const PIN_ERROR: usize = 29;
    pub const PREFIX_WORDS: usize = 30;
}

/// Flag: stored vertices carry texture coordinates.
pub const FLAG_UVS: u32 = 1;
/// Triangles of a virtual mesh's fallback cut ([`Mesh::from_virtual`]) when
/// its source names none.
pub const DEFAULT_FALLBACK_TRIANGLES: u32 = 4096;
/// Flag: the mesh is closed, every edge joining two triangles, so its
/// surfaces facing a direction hide those facing away.
pub const FLAG_CLOSED: u32 = 2;

/// A sphere and the mesh-space simplification error it bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErrorSphere {
    pub center: Vec3,
    pub radius: f32,
    /// `f32::MAX` for groups never simplified further.
    pub error: f32,
}

/// Word offsets of each section in the file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Layout {
    nodes: usize,
    groups: usize,
    clusters: usize,
    page_table: usize,
    pages: usize,
    end: usize,
}

/// A validated virtual mesh: the words of the file or of its prefix, where
/// its sections start, and where to fetch its other pages from.
pub struct VirtualMesh {
    words: Box<[u32]>,
    layout: Layout,
    bounds: Aabb,
    // Pages `words` holds: every page, or the pinned ones of a prefix.
    available_pages: usize,
    source: Option<String>,
}

/// A cluster's record, decoded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClusterInfo {
    pub sphere: ErrorSphere,
    /// Normal cone axis and cutoff as 8-bit snorm (cutoff 127: no cone).
    pub cone: [i8; 4],
    /// The finer group whose simplification made it, or `None` for
    /// original geometry.
    pub refined: Option<u32>,
    pub group: u32,
    pub page: u32,
    pub vertex_offset: u32,
    pub triangle_offset: u32,
    pub vertex_count: u32,
    pub triangle_count: u32,
}

impl std::fmt::Debug for VirtualMesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VirtualMesh")
            .field("levels", &self.levels())
            .field("clusters", &self.cluster_count())
            .field("triangles", &self.triangle_count())
            .field("pages", &self.page_count())
            .field("available_pages", &self.available_pages)
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

fn invalid(reason: &'static str) -> AssetError {
    AssetError::InvalidVirtualMesh { reason }
}

fn float(word: u32) -> f32 {
    f32::from_bits(word)
}

fn vec3(words: &[u32]) -> Vec3 {
    Vec3::new(float(words[0]), float(words[1]), float(words[2]))
}

impl VirtualMesh {
    /// Bytes of the prefix of the file whose first [`HEADER_WORDS`] words
    /// are `header` (metadata and pinned pages), for hosts fetching it
    /// first; `None` when it is not an `EVG1` header of this version.
    pub fn prefix_bytes(header: &[u8]) -> Option<usize> {
        let word = |index: usize| {
            header
                .get(index * 4..index * 4 + 4)
                .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four bytes")))
        };
        (word(header::MAGIC)? == u32::from_le_bytes(VIRTUAL_MESH_MAGIC)
            && word(header::VERSION)? == VIRTUAL_MESH_VERSION)
            .then(|| word(header::PREFIX_WORDS).map(|words| words as usize * 4))
            .flatten()
    }

    /// Validates `bytes` as an `EVG1` file or its prefix: every section in
    /// range, every reference to a node, group, cluster or page in bounds,
    /// finite geometry, counts within the cluster limits, pinned groups in
    /// the pinned pages, errors growing toward coarser groups, and the
    /// triangles of every page held.
    pub fn decode(bytes: &[u8]) -> Result<Self, AssetError> {
        if !bytes.len().is_multiple_of(4) || bytes.len() < HEADER_WORDS * 4 {
            return Err(invalid("length is not a whole header of words"));
        }
        let words: Box<[u32]> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect();
        Self::from_words(words)
    }

    /// Validates `words` as [`Self::decode`] does.
    pub fn from_words(words: Box<[u32]>) -> Result<Self, AssetError> {
        if words.len() < HEADER_WORDS {
            return Err(invalid("shorter than a header"));
        }
        if words[header::MAGIC] != u32::from_le_bytes(VIRTUAL_MESH_MAGIC) {
            return Err(invalid("magic is not EVG1"));
        }
        if words[header::VERSION] != VIRTUAL_MESH_VERSION {
            return Err(invalid("unsupported version"));
        }
        if words[header::FLAGS] & !(FLAG_UVS | FLAG_CLOSED) != 0 {
            return Err(invalid("unknown flags"));
        }
        let count = |index: usize| words[index] as usize;
        let (levels, nodes, groups, clusters, pages) = (
            count(header::LEVELS),
            count(header::NODES),
            count(header::GROUPS),
            count(header::CLUSTERS),
            count(header::PAGES),
        );
        if levels == 0 || nodes < levels || groups == 0 || clusters == 0 || pages == 0 {
            return Err(invalid("empty hierarchy"));
        }
        let pinned = count(header::PINNED_PAGES);
        if pinned == 0 || pinned > pages {
            return Err(invalid("pinned pages are out of range"));
        }
        let pin_error = float(words[header::PIN_ERROR]);
        if pin_error.is_nan() || pin_error < 0.0 {
            return Err(invalid("the pin error is negative or not a number"));
        }
        let section = |start: usize, items: usize, size: usize| {
            items
                .checked_mul(size)
                .and_then(|length| start.checked_add(length))
                .filter(|&end| end <= words.len())
                .ok_or_else(|| invalid("a section runs past the end"))
        };
        let node_start = HEADER_WORDS;
        let group_start = section(node_start, nodes, NODE_WORDS)?;
        let cluster_start = section(group_start, groups, GROUP_WORDS)?;
        let page_table = section(cluster_start, clusters, CLUSTER_WORDS)?;
        let page_start = section(page_table, pages, 2)?;
        let layout = Layout {
            nodes: node_start,
            groups: group_start,
            clusters: cluster_start,
            page_table,
            pages: page_start,
            end: words.len(),
        };
        let floats = |range: std::ops::Range<usize>| {
            words[range].iter().all(|&word| float(word).is_finite())
        };
        if !floats(header::POSITION_MIN..header::BOUNDS_MAX + 3) {
            return Err(invalid("quantization or bounds are not finite"));
        }
        let steps = (header::POSITION_STEP..header::POSITION_STEP + 3)
            .chain(header::UV_STEP..header::UV_STEP + 2);
        if steps.clone().any(|index| float(words[index]) < 0.0) {
            return Err(invalid("negative quantization step"));
        }
        let bounds_min = vec3(&words[header::BOUNDS_MIN..]);
        let bounds_max = vec3(&words[header::BOUNDS_MAX..]);
        if !bounds_min.cmple(bounds_max).all()
            || bounds_min.abs().max_element() > 1.0e6
            || bounds_max.abs().max_element() > 1.0e6
        {
            return Err(invalid("bounds are inverted or beyond 1e6"));
        }
        // Pages: consecutive and at most PAGE_WORDS each, together the rest
        // of the file; the pinned ones end the prefix.
        let mut expected = 0usize;
        let mut pinned_words = 0usize;
        for page in 0..pages {
            let entry = &words[page_table + page * 2..page_table + page * 2 + 2];
            if entry[0] as usize != expected {
                return Err(invalid("pages are not consecutive"));
            }
            if entry[1] as usize > PAGE_WORDS {
                return Err(invalid("a page exceeds its size limit"));
            }
            expected += entry[1] as usize;
            if page + 1 == pinned {
                pinned_words = expected;
            }
        }
        let prefix = page_start + pinned_words;
        if count(header::PREFIX_WORDS) != prefix {
            return Err(invalid("the prefix length does not match the pinned pages"));
        }
        let available_pages = if words.len() == page_start + expected {
            pages
        } else if words.len() == prefix {
            pinned
        } else {
            return Err(invalid("neither a whole file nor its prefix"));
        };
        let mesh = Self {
            words,
            layout,
            bounds: Aabb::from_min_max(bounds_min, bounds_max),
            available_pages,
            source: None,
        };
        mesh.validate_hierarchy()?;
        Ok(mesh)
    }

    /// This mesh, fetching the pages its prefix lacks from `url` (the whole
    /// file, by byte ranges: [`Self::page_bytes`]).
    pub fn with_source(mut self, url: impl Into<String>) -> Self {
        self.source = Some(url.into());
        self
    }

    /// Where the pages this mesh lacks come from.
    pub fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }

    /// Pages whose words this mesh holds: every page, or the pinned ones.
    pub fn available_pages(&self) -> usize {
        self.available_pages
    }

    /// The first pages, holding every group simplified with more error than
    /// [`Self::pin_error`].
    pub fn pinned_pages(&self) -> usize {
        self.words[header::PINNED_PAGES] as usize
    }

    /// Groups simplified with more error than this lie in the pinned pages.
    pub fn pin_error(&self) -> f32 {
        float(self.words[header::PIN_ERROR])
    }

    /// Byte range of page `page` in the file.
    pub fn page_bytes(&self, page: usize) -> Option<std::ops::Range<u64>> {
        (page < self.page_count()).then(|| {
            let entry = self.layout.page_table + page * 2;
            let start = (self.layout.pages as u64 + u64::from(self.words[entry])) * 4;
            start..start + u64::from(self.words[entry + 1]) * 4
        })
    }

    /// Validates `words`, fetched separately, as page `page`: its size, and
    /// the local triangle indices of every cluster it holds.
    pub fn validate_page(&self, page: usize, words: &[u32]) -> Result<(), AssetError> {
        let size = self
            .page_bytes(page)
            .map(|range| ((range.end - range.start) / 4) as usize)
            .ok_or_else(|| invalid("a page is out of range"))?;
        if words.len() != size {
            return Err(invalid("a page's length does not match its table entry"));
        }
        for cluster in 0..self.cluster_count() {
            let info = self.cluster(cluster);
            if info.page as usize == page {
                Self::validate_triangles(&info, words)?;
            }
        }
        Ok(())
    }

    /// Whether every local triangle index of `info` stays within its
    /// vertices, reading `page`'s words.
    fn validate_triangles(info: &ClusterInfo, page: &[u32]) -> Result<(), AssetError> {
        let triangle_words = (info.triangle_count as usize * 3).div_ceil(4);
        let start = info.triangle_offset as usize;
        let locals = page[start..start + triangle_words]
            .iter()
            .flat_map(|word| word.to_le_bytes());
        if locals
            .take(info.triangle_count as usize * 3)
            .any(|local| u32::from(local) >= info.vertex_count)
        {
            return Err(invalid("a triangle index exceeds its cluster's vertices"));
        }
        Ok(())
    }

    fn validate_hierarchy(&self) -> Result<(), AssetError> {
        let (levels, node_count, group_count, cluster_count) = (
            self.levels() as usize,
            self.node_count(),
            self.group_count(),
            self.cluster_count(),
        );
        let sphere_ok = |sphere: &ErrorSphere| {
            sphere.center.is_finite()
                && sphere.radius.is_finite()
                && sphere.radius >= 0.0
                && sphere.error >= 0.0
                && !sphere.error.is_nan()
        };
        for node in 0..node_count {
            if !sphere_ok(&self.node_sphere(node)) {
                return Err(invalid("a node sphere is not finite"));
            }
            let words = self.node_words(node);
            if words[5] != NONE {
                if words[5] as usize >= group_count {
                    return Err(invalid("a node's group is out of range"));
                }
            } else {
                let (first, children) = (words[6] as usize, words[7] as usize);
                if children == 0
                    || first
                        .checked_add(children)
                        .is_none_or(|end| end > node_count)
                {
                    return Err(invalid("a node's children are out of range"));
                }
            }
        }
        // Walk down from the roots: every node has at most one parent, so the
        // hierarchy is a forest and traversals end.
        let mut visited = vec![false; node_count];
        let mut stack: Vec<(usize, u32)> = (0..levels).map(|root| (root, 1)).collect();
        let mut depth = 0u32;
        while let Some((node, level)) = stack.pop() {
            if std::mem::replace(&mut visited[node], true) {
                return Err(invalid("a node has more than one parent"));
            }
            depth = depth.max(level);
            let words = self.node_words(node);
            if words[5] == NONE {
                let first = words[6] as usize;
                stack.extend((first..first + words[7] as usize).map(|child| (child, level + 1)));
            }
        }
        if depth != self.words[header::DEPTH] {
            return Err(invalid("hierarchy depth does not match its nodes"));
        }
        let mut owner = vec![NONE; cluster_count];
        for group in 0..group_count {
            let words = self.group_words(group);
            if !sphere_ok(&self.group_sphere(group)) {
                return Err(invalid("a group sphere is not finite"));
            }
            let (first, count) = (words[5] as usize, words[6] as usize);
            if count == 0
                || first
                    .checked_add(count)
                    .is_none_or(|end| end > cluster_count)
            {
                return Err(invalid("a group's clusters are out of range"));
            }
            if words[7] as usize >= levels {
                return Err(invalid("a group's level is out of range"));
            }
            for owned in &mut owner[first..first + count] {
                if *owned != NONE {
                    return Err(invalid("clusters belong to one group each"));
                }
                *owned = group as u32;
            }
        }
        let (pinned, pin_error) = (self.pinned_pages(), self.pin_error());
        let mut triangles = 0u64;
        for (cluster, &owned) in owner.iter().enumerate() {
            let info = self.cluster(cluster);
            if !sphere_ok(&info.sphere) {
                return Err(invalid("a cluster sphere is not finite"));
            }
            if owned != info.group {
                return Err(invalid("a cluster's group does not list it"));
            }
            if info
                .refined
                .is_some_and(|refined| refined as usize >= group_count)
            {
                return Err(invalid("a cluster's refined group is out of range"));
            }
            // Cuts cover each surface once only while errors grow toward
            // coarser groups; the prefix alone draws every surface.
            let error = self.group_sphere(info.group as usize).error;
            if info
                .refined
                .is_some_and(|refined| self.group_sphere(refined as usize).error > error)
            {
                return Err(invalid(
                    "a cluster's group has less error than the group it came from",
                ));
            }
            if error > pin_error && info.page as usize >= pinned {
                return Err(invalid(
                    "a pinned group's cluster lies beyond the pinned pages",
                ));
            }
            if info.vertex_count == 0
                || info.vertex_count > CLUSTER_VERTICES
                || info.triangle_count == 0
                || info.triangle_count > CLUSTER_TRIANGLES
            {
                return Err(invalid("cluster sizes exceed the limits"));
            }
            let size = self
                .page_bytes(info.page as usize)
                .map(|range| ((range.end - range.start) / 4) as usize)
                .ok_or_else(|| invalid("a cluster's page is out of range"))?;
            let vertices_end =
                info.vertex_offset as usize + info.vertex_count as usize * VERTEX_WORDS;
            let triangle_words = (info.triangle_count as usize * 3).div_ceil(4);
            if vertices_end > size || info.triangle_offset as usize + triangle_words > size {
                return Err(invalid("a cluster runs past its page"));
            }
            if let Some(page) = self.page(info.page as usize) {
                Self::validate_triangles(&info, page)?;
            }
            triangles += u64::from(info.triangle_count);
        }
        if triangles != u64::from(self.words[header::TRIANGLES]) {
            return Err(invalid("triangle count does not match the clusters"));
        }
        Ok(())
    }

    fn node_words(&self, node: usize) -> &[u32] {
        let start = self.layout.nodes + node * NODE_WORDS;
        &self.words[start..start + NODE_WORDS]
    }

    fn group_words(&self, group: usize) -> &[u32] {
        let start = self.layout.groups + group * GROUP_WORDS;
        &self.words[start..start + GROUP_WORDS]
    }

    fn cluster_words(&self, cluster: usize) -> &[u32] {
        let start = self.layout.clusters + cluster * CLUSTER_WORDS;
        &self.words[start..start + CLUSTER_WORDS]
    }

    fn sphere(words: &[u32]) -> ErrorSphere {
        ErrorSphere {
            center: vec3(words),
            radius: float(words[3]),
            error: float(words[4]),
        }
    }

    /// A node's bounds with the largest error of its subtree.
    pub fn node_sphere(&self, node: usize) -> ErrorSphere {
        Self::sphere(self.node_words(node))
    }

    /// A group's simplification: its sphere and error.
    pub fn group_sphere(&self, group: usize) -> ErrorSphere {
        Self::sphere(self.group_words(group))
    }

    /// A cluster's record.
    pub fn cluster(&self, cluster: usize) -> ClusterInfo {
        let words = self.cluster_words(cluster);
        let [x, y, z, cutoff] = words[4].to_le_bytes().map(|byte| byte as i8);
        ClusterInfo {
            sphere: ErrorSphere {
                center: vec3(words),
                radius: float(words[3]),
                error: 0.0,
            },
            cone: [x, y, z, cutoff],
            refined: (words[5] != NONE).then_some(words[5]),
            group: words[6],
            page: words[7],
            vertex_offset: words[8],
            triangle_offset: words[9],
            vertex_count: words[10] & 0xffff,
            triangle_count: words[10] >> 16,
        }
    }

    /// Words of page `page`, when this mesh holds them.
    pub fn page(&self, page: usize) -> Option<&[u32]> {
        (page < self.available_pages).then(|| {
            let entry = self.layout.page_table + page * 2;
            let start = self.layout.pages + self.words[entry] as usize;
            &self.words[start..start + self.words[entry + 1] as usize]
        })
    }

    pub fn levels(&self) -> u32 {
        self.words[header::LEVELS]
    }

    /// Nodes on the longest path from a level's root to a leaf.
    pub fn depth(&self) -> u32 {
        self.words[header::DEPTH]
    }

    pub fn node_count(&self) -> usize {
        self.words[header::NODES] as usize
    }

    pub fn group_count(&self) -> usize {
        self.words[header::GROUPS] as usize
    }

    pub fn cluster_count(&self) -> usize {
        self.words[header::CLUSTERS] as usize
    }

    pub fn page_count(&self) -> usize {
        self.words[header::PAGES] as usize
    }

    /// Triangles of every cluster at every level.
    pub fn triangle_count(&self) -> u32 {
        self.words[header::TRIANGLES]
    }

    /// Triangles of the original geometry.
    pub fn source_triangle_count(&self) -> u32 {
        self.words[header::SOURCE_TRIANGLES]
    }

    /// Whether stored vertices carry texture coordinates.
    pub fn has_uvs(&self) -> bool {
        self.words[header::FLAGS] & FLAG_UVS != 0
    }

    /// Whether the mesh is closed ([`FLAG_CLOSED`]).
    pub fn closed(&self) -> bool {
        self.words[header::FLAGS] & FLAG_CLOSED != 0
    }

    /// Bounds of every level's geometry.
    pub fn bounds(&self) -> Aabb {
        self.bounds
    }

    /// Node words, [`NODE_WORDS`] per node.
    pub fn node_section(&self) -> &[u32] {
        &self.words[self.layout.nodes..self.layout.groups]
    }

    /// Group words, [`GROUP_WORDS`] per group.
    pub fn group_section(&self) -> &[u32] {
        &self.words[self.layout.groups..self.layout.clusters]
    }

    /// Cluster words, [`CLUSTER_WORDS`] per cluster.
    pub fn cluster_section(&self) -> &[u32] {
        &self.words[self.layout.clusters..self.layout.page_table]
    }

    /// The page table: each page's word offset in [`Self::page_section`]
    /// and its words.
    pub fn page_table(&self) -> &[u32] {
        &self.words[self.layout.page_table..self.layout.pages]
    }

    /// The words of every page held, consecutively.
    pub fn page_section(&self) -> &[u32] {
        &self.words[self.layout.pages..self.layout.end]
    }

    /// The words held: the whole file, or its prefix.
    pub fn words(&self) -> &[u32] {
        &self.words
    }

    pub fn byte_len(&self) -> usize {
        self.words.len() * 4
    }

    /// Quantization: position of step zero and step size per axis, texture
    /// coordinates of step zero and step size.
    pub fn quantization(&self) -> ([f32; 3], [f32; 3], [f32; 2], [f32; 2]) {
        let floats = |start: usize, count: usize| -> Vec<f32> {
            self.words[start..start + count]
                .iter()
                .map(|&word| float(word))
                .collect()
        };
        let (a, b, c, d) = (
            floats(header::POSITION_MIN, 3),
            floats(header::POSITION_STEP, 3),
            floats(header::UV_MIN, 2),
            floats(header::UV_STEP, 2),
        );
        (
            [a[0], a[1], a[2]],
            [b[0], b[1], b[2]],
            [c[0], c[1]],
            [d[0], d[1]],
        )
    }

    /// Cluster `cluster`'s vertices and local triangle indices, decoded; its
    /// page must be held ([`Self::available_pages`]).
    pub fn cluster_geometry(&self, cluster: usize) -> (Vec<MeshVertex>, Vec<u8>) {
        let info = self.cluster(cluster);
        let page = self.page(info.page as usize).expect("a held page");
        let (position_min, position_step, uv_min, uv_step) = self.quantization();
        let low = |word: u32| word & 0xffff;
        let high = |word: u32| word >> 16;
        let vertices = (0..info.vertex_count as usize)
            .map(|vertex| {
                let start = info.vertex_offset as usize + vertex * VERTEX_WORDS;
                let words = &page[start..start + VERTEX_WORDS];
                let steps = [low(words[0]), high(words[0]), low(words[1])];
                let position: [f32; 3] = std::array::from_fn(|axis| {
                    position_min[axis] + steps[axis] as f32 * position_step[axis]
                });
                let normal =
                    octahedral_normal(high(words[1]) as u16 as i16, low(words[2]) as u16 as i16);
                let uv = [
                    uv_min[0] + high(words[2]) as f32 * uv_step[0],
                    uv_min[1] + low(words[3]) as f32 * uv_step[1],
                ];
                MeshVertex {
                    position,
                    normal: normal.to_array(),
                    uv,
                }
            })
            .collect();
        let words = (info.triangle_count as usize * 3).div_ceil(4);
        let start = info.triangle_offset as usize;
        let triangles = page[start..start + words]
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .take(info.triangle_count as usize * 3)
            .collect();
        (vertices, triangles)
    }

    /// The clusters of the cut at mesh-space error `error`: every cluster
    /// whose group simplifies with more error and whose own error is at most
    /// it. Errors are monotonic up the hierarchy, so every cut covers the
    /// surface once.
    pub fn cut(&self, error: f32) -> Vec<u32> {
        (0..self.cluster_count())
            .filter(|&cluster| {
                let info = self.cluster(cluster);
                self.group_sphere(info.group as usize).error > error
                    && info
                        .refined
                        .is_none_or(|refined| self.group_sphere(refined as usize).error <= error)
            })
            .map(|cluster| cluster as u32)
            .collect()
    }

    /// Triangles of clusters `clusters`.
    pub fn triangles_of(&self, clusters: &[u32]) -> u32 {
        clusters
            .iter()
            .map(|&cluster| self.cluster(cluster as usize).triangle_count)
            .sum()
    }

    /// The most detailed cut within the pinned pages with at most
    /// `triangles` triangles (the coarsest when none is that small), as an
    /// ordinary mesh: what renderers draw where they cannot draw clusters.
    pub fn fallback(&self, triangles: u32) -> Result<Mesh, AssetError> {
        // Cuts at the pin error or above use pinned groups alone.
        let pin = self.pin_error();
        let mut errors: Vec<f32> = (0..self.group_count())
            .map(|group| self.group_sphere(group).error)
            .filter(|error| *error < f32::MAX && *error >= pin)
            .collect();
        errors.push(pin);
        errors.sort_by(f32::total_cmp);
        errors.dedup();
        // Cut sizes shrink as the error grows: the first small enough.
        let fits = |error: f32| self.triangles_of(&self.cut(error)) <= triangles;
        let index = errors.partition_point(|&error| !fits(error));
        let error = errors.get(index).copied().unwrap_or(f32::MAX / 2.0);
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for cluster in self.cut(error) {
            let (cluster_vertices, triangles) = self.cluster_geometry(cluster as usize);
            let base = vertices.len() as u32;
            vertices.extend(cluster_vertices);
            indices.extend(triangles.iter().map(|&local| base + u32::from(local)));
        }
        Mesh::new(vertices, indices)
    }
}

/// Octahedral snorm16 coordinates of a unit normal, as renderers encode
/// them.
pub fn octahedral_encode(normal: [f32; 3]) -> [i16; 2] {
    let [x, y, z] = normal;
    let length = x.abs() + y.abs() + z.abs();
    let (u, v) = if length > 0.0 {
        (x / length, y / length)
    } else {
        (0.0, 0.0)
    };
    let sign = |value: f32| if value >= 0.0 { 1.0 } else { -1.0 };
    let (u, v) = if z < 0.0 {
        ((1.0 - v.abs()) * sign(u), (1.0 - u.abs()) * sign(v))
    } else {
        (u, v)
    };
    [u, v].map(|value| (engine_math::clamp(value, -1.0, 1.0) * 32767.0).round() as i16)
}

/// The unit normal of octahedral snorm16 coordinates, as renderers decode
/// them.
pub fn octahedral_normal(x: i16, y: i16) -> Vec3 {
    let x = engine_math::clamp(f32::from(x) / 32767.0, -1.0, 1.0);
    let y = engine_math::clamp(f32::from(y) / 32767.0, -1.0, 1.0);
    let z = 1.0 - x.abs() - y.abs();
    let t = engine_math::max(-z, 0.0);
    let normal = Vec3::new(
        x + if x >= 0.0 { -t } else { t },
        y + if y >= 0.0 { -t } else { t },
        z,
    );
    normal.normalize_or(Vec3::Y)
}
