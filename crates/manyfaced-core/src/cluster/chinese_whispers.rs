//! Chinese whispers graph clustering.
//!
//! Ported from `dlib/clustering/chinese_whispers.h`, `dlib/graph_utils/edge_list_graphs.h`,
//! `sample_pair.h` and the Python binding `chinese_whispers_clustering`
//! (`tools/python/src/face_recognition.cpp`), dlib v20.0.1, Boost Software License 1.0, see
//! `THIRD_PARTY_NOTICES`. With dlib's default-seeded random generator the labels are identical
//! to dlib's.

use std::collections::{BTreeMap, TryReserveError};

use super::rand::DlibRand;

/// An undirected weighted edge (`sample_pair`): stored with `index1 <= index2`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Edge {
    index1: usize,
    index2: usize,
    weight: f64,
}

impl Edge {
    /// An edge between `a` and `b` (in any order) with weight 1, like `sample_pair(a, b)`.
    pub fn new(a: usize, b: usize) -> Self {
        Self::weighted(a, b, 1.0)
    }

    /// `sample_pair(a, b, weight)`. dlib calls the weight a "distance", but chinese whispers
    /// sums it as a vote strength.
    pub fn weighted(a: usize, b: usize, weight: f64) -> Self {
        let (index1, index2) = if a < b { (a, b) } else { (b, a) };
        Edge {
            index1,
            index2,
            weight,
        }
    }
}

/// dlib's `chinese_whispers(edges, labels, num_iterations)` with a default-seeded `dlib::rand`.
///
/// Returns one label per node `0..=max index`, numbered in order of first appearance; nodes that
/// appear in no edge keep a label of their own. An empty edge list gives no labels, like dlib.
///
/// The memory is proportional to the largest node index, not to the number of edges. If that
/// memory is not available, the result is an error (dlib raises `MemoryError`).
pub fn chinese_whispers(
    edges: &[Edge],
    num_iterations: u64,
) -> Result<Vec<usize>, TryReserveError> {
    chinese_whispers_with(edges, num_iterations, &mut DlibRand::default())
}

/// [`chinese_whispers`] with an explicit generator.
pub fn chinese_whispers_with(
    edges: &[Edge],
    num_iterations: u64,
    rnd: &mut DlibRand,
) -> Result<Vec<usize>, TryReserveError> {
    if edges.is_empty() {
        return Ok(Vec::new());
    }
    // convert_unordered_to_ordered: both directions, self loops once; then sort by index. dlib
    // uses std::sort (unstable); only the order of exactly duplicated edges could differ, which
    // can change floating point vote sums by rounding when their weights differ.
    let mut ordered: Vec<(usize, usize, f64)> = Vec::with_capacity(edges.len() * 2);
    for e in edges {
        ordered.push((e.index1, e.index2, e.weight));
        if e.index1 != e.index2 {
            ordered.push((e.index2, e.index1, e.weight));
        }
    }
    ordered.sort_by_key(|e| (e.0, e.1));

    // find_neighbor_ranges: [start, end) of each node's edges; nodes without edges get (0, 0).
    // Saturating: an index of usize::MAX makes the reservation below fail instead of wrapping.
    let num_nodes = ordered
        .iter()
        .map(|e| e.0.max(e.1))
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    let mut neighbors: Vec<(usize, usize)> = Vec::new();
    neighbors.try_reserve_exact(num_nodes)?;
    neighbors.resize(num_nodes, (0, 0));
    let (mut cur, mut start) = (0usize, 0usize);
    for (i, e) in ordered.iter().enumerate() {
        if e.0 != cur {
            neighbors[cur] = (start, i);
            start = i;
            cur = e.0;
        }
    }
    neighbors[cur] = (start, ordered.len());

    let mut labels: Vec<usize> = Vec::new();
    labels.try_reserve_exact(num_nodes)?;
    labels.extend(0..num_nodes);
    let total = (num_nodes as u64).saturating_mul(num_iterations);
    let mut counts: BTreeMap<usize, f64> = BTreeMap::new();
    for _ in 0..total {
        let idx = (rnd.get_random_64bit_number() % num_nodes as u64) as usize;
        // Sum the weights of each label among the neighbours (std::map keeps labels sorted).
        counts.clear();
        let (s, end) = neighbors[idx];
        for e in &ordered[s..end] {
            *counts.entry(labels[e.1]).or_insert(0.0) += e.2;
        }
        // The most common label; ties go to the smallest label (strict `>`).
        let mut best_score = f64::NEG_INFINITY;
        let mut best_label = labels[idx];
        for (&label, &score) in &counts {
            if score > best_score {
                best_score = score;
                best_label = label;
            }
        }
        labels[idx] = best_label;
    }

    // Remap the labels to 0..k in order of first appearance.
    let mut remap: BTreeMap<usize, usize> = BTreeMap::new();
    for &l in &labels {
        let next = remap.len();
        remap.entry(l).or_insert(next);
    }
    Ok(labels.iter().map(|l| remap[l]).collect())
}

/// dlib's Python `chinese_whispers_clustering(descriptors, threshold)`: connect every pair
/// `i <= j` (self pairs included) whose Euclidean distance is below `threshold`, then run
/// [`chinese_whispers`] with 100 iterations.
///
/// `threshold` is an `f32` because the binding takes a C++ `float`; distances are computed in
/// `f64` as `sqrt(sum((a - b)^2))` in element order, like `length(a - b)` on `matrix<double>`.
/// As in dlib, a descriptor whose distance to itself is not below the threshold (e.g. NaN) gets
/// no self edge, and if it is the last one the result is shorter than the input.
pub fn chinese_whispers_clustering<D: AsRef<[f64]>>(
    descriptors: &[D],
    threshold: f32,
) -> Vec<usize> {
    let threshold = f64::from(threshold);
    let mut edges = Vec::new();
    for i in 0..descriptors.len() {
        for j in i..descriptors.len() {
            let (a, b) = (descriptors[i].as_ref(), descriptors[j].as_ref());
            let mut sum = 0.0f64;
            for (x, y) in a.iter().zip(b) {
                let d = x - y;
                sum += d * d;
            }
            if sum.sqrt() < threshold {
                edges.push(Edge::new(i, j));
            }
        }
    }
    // There is one node for each descriptor, so the memory for the nodes is always available.
    chinese_whispers(&edges, 100).expect("one node for each descriptor")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_cliques() {
        let edges = [
            Edge::new(0, 1),
            Edge::new(1, 2),
            Edge::new(0, 2),
            Edge::new(3, 4),
            Edge::new(4, 5),
            Edge::new(3, 5),
        ];
        let labels = chinese_whispers(&edges, 100).unwrap();
        assert_eq!(labels.len(), 6);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
        assert_eq!(labels[0], 0, "labels are numbered by first appearance");
    }

    #[test]
    fn isolated_and_empty() {
        assert!(chinese_whispers(&[], 100).unwrap().is_empty());
        // Node 1 has no edges: it keeps its own label.
        let labels = chinese_whispers(&[Edge::new(0, 0), Edge::new(2, 2)], 100).unwrap();
        assert_eq!(labels, [0, 1, 2]);
    }

    #[test]
    fn clustering_descriptors() {
        let d = [
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![5.0, 5.0],
            vec![5.0, 5.1],
            vec![10.0, 0.0],
        ];
        let labels = chinese_whispers_clustering(&d, 0.5);
        assert_eq!(labels, [0, 0, 1, 1, 2]);
        assert!(chinese_whispers_clustering::<Vec<f64>>(&[], 0.5).is_empty());
        // A NaN last descriptor has no self edge, so it is dropped like in dlib.
        let d = [vec![0.0], vec![f64::NAN]];
        assert_eq!(chinese_whispers_clustering(&d, 0.5), [0]);
    }

    #[test]
    fn huge_node_index_is_an_error() {
        // The node table needs usize::MAX entries: the allocation must fail, not abort.
        assert!(chinese_whispers(&[Edge::new(0, usize::MAX - 1)], 100).is_err());
        assert!(chinese_whispers(&[Edge::new(0, usize::MAX)], 100).is_err());
    }
}
