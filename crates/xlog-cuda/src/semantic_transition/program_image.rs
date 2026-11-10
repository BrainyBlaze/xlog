//! Selected decision-forest geometry inside the joint task-ground publication.
use super::{identity_words, publication_input_error, SemanticTransitionError};
use crate::{SemanticProgramAdmission, SemanticProgramNode};
use sha2::{Digest, Sha256};

pub(super) const IMAGE_GEOMETRY_WORDS: usize = 32;
pub(super) const IMAGE_FRAME_HEADER_WORDS: usize = 48;
pub(super) const IMAGE_LINEAGE_WORDS: usize = 56;
pub(super) const IMAGE_GEOMETRY_MAGIC: u64 = 0x584c4f47494d4747;
pub(super) const IMAGE_FRAME_MAGIC: u64 = 0x584c4f47494d4746;
pub(super) const IMAGE_GENOME_DOMAIN: &[u8] = b"xlog.program-genome.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ProgramImageLayout {
    pub roots: usize,
    pub nodes: usize,
    pub guards: usize,
    pub leaves: usize,
    pub blocks: usize,
    pub mutations: usize,
    pub local_roots: usize,
    pub local_nodes: usize,
    pub local_guards: usize,
    pub local_leaves: usize,
    pub local_blocks: usize,
    pub local_mutations: usize,
    pub facts: usize,
    pub library: usize,
    pub statements: usize,
    pub entry_supports: usize,
    pub root_offset: usize,
    pub node_offset: usize,
    pub rule_offset: usize,
    pub block_offset: usize,
    pub activity_offset: usize,
    pub lineage_offset: usize,
    pub frame_bytes: usize,
}

fn overflow() -> SemanticTransitionError {
    publication_input_error("program image geometry exceeds its finite native domain")
}
fn word(bytes: &[u8], index: usize) -> Result<u64, SemanticTransitionError> {
    let begin = index.checked_mul(8).ok_or_else(overflow)?;
    let value = bytes
        .get(begin..begin.checked_add(8).ok_or_else(overflow)?)
        .ok_or_else(overflow)?;
    Ok(u64::from_le_bytes(value.try_into().unwrap()))
}
fn put(bytes: &mut [u8], index: usize, value: u64) {
    bytes[index * 8..index * 8 + 8].copy_from_slice(&value.to_le_bytes());
}

impl ProgramImageLayout {
    fn geometry(
        roots: usize,
        nodes: usize,
        guards: usize,
        leaves: usize,
        blocks: usize,
        mutations: usize,
        local: [usize; 6],
        facts: usize,
        library: usize,
        statements: usize,
    ) -> Result<Self, SemanticTransitionError> {
        let counts = [
            roots, nodes, guards, leaves, blocks, mutations, facts, library, statements,
        ];
        if counts.iter().any(|&n| n > u32::MAX as usize)
            || (roots == 0) != (nodes == 0)
            || (roots != 0 && (guards == 0 || leaves == 0 || blocks == 0 || facts == 0))
            || local[5] > mutations
        {
            return Err(overflow());
        }
        for (&begin, &count) in local[..5].iter().zip(&counts[..5]) {
            if (count == 0 && begin != 0) || (count != 0 && begin >= count) {
                return Err(overflow());
            }
        }
        if library != 0 && library != super::rule_control::RULE_LIBRARY_COUNT {
            return Err(overflow());
        }
        let root_offset = IMAGE_FRAME_HEADER_WORDS * 8;
        let node_offset = roots
            .checked_mul(8)
            .and_then(|n| root_offset.checked_add(n))
            .ok_or_else(overflow)?;
        let rule_offset = nodes
            .checked_mul(64)
            .and_then(|n| node_offset.checked_add(n))
            .ok_or_else(overflow)?;
        let block_offset = mutations
            .checked_mul(std::mem::size_of::<crate::SemanticProgramRule>())
            .and_then(|n| rule_offset.checked_add(n))
            .ok_or_else(overflow)?;
        let activity_offset = mutations
            .checked_mul(4)
            .and_then(|n| block_offset.checked_add(n))
            .ok_or_else(overflow)?;
        let lineage_offset = blocks
            .checked_mul(8)
            .and_then(|n| activity_offset.checked_add(n))
            .ok_or_else(overflow)?;
        let frame_bytes = mutations
            .checked_mul(IMAGE_LINEAGE_WORDS * 8)
            .and_then(|n| lineage_offset.checked_add(n))
            .ok_or_else(overflow)?;
        Ok(Self {
            roots,
            nodes,
            guards,
            leaves,
            blocks,
            mutations,
            local_roots: local[0],
            local_nodes: local[1],
            local_guards: local[2],
            local_leaves: local[3],
            local_blocks: local[4],
            local_mutations: local[5],
            facts,
            library,
            statements,
            entry_supports: 0,
            root_offset,
            node_offset,
            rule_offset,
            block_offset,
            activity_offset,
            lineage_offset,
            frame_bytes,
        })
    }

    pub fn new(
        program: Option<&SemanticProgramAdmission>,
    ) -> Result<Self, SemanticTransitionError> {
        let Some(program) = program else {
            return Self::geometry(0, 0, 0, 0, 0, 0, [0; 6], 0, 0, 0);
        };
        let image = &program.image;
        Self::geometry(
            image.roots.len(),
            image.nodes.len(),
            image.guards.len(),
            image.leaves.len(),
            image.rule_blocks.len(),
            image.mutation_capacity as usize,
            [
                image.local_root_begin as usize,
                image.local_node_begin as usize,
                image.local_guard_begin as usize,
                image.local_leaf_begin as usize,
                image.local_block_begin as usize,
                image.local_mutation_capacity as usize,
            ],
            program.fact_capacity().map_err(publication_input_error)?,
            0,
            program.observation_facts.len(),
        )
    }

    pub fn with_library(mut self, library: usize) -> Result<Self, SemanticTransitionError> {
        if library != 0 && library != super::rule_control::RULE_LIBRARY_COUNT {
            return Err(overflow());
        }
        self.library = library;
        Ok(self)
    }

    pub fn with_entry_supports(mut self, capacity: usize) -> Result<Self, SemanticTransitionError> {
        if capacity > u32::MAX as usize || self.roots == 0 && capacity != 0 {
            return Err(overflow());
        }
        self.entry_supports = capacity;
        Ok(self)
    }

    pub fn footer(self) -> [u64; IMAGE_GEOMETRY_WORDS] {
        let mut words = [0; IMAGE_GEOMETRY_WORDS];
        words[..17].copy_from_slice(&[
            IMAGE_GEOMETRY_MAGIC,
            1,
            self.roots as u64,
            self.nodes as u64,
            self.guards as u64,
            self.leaves as u64,
            self.blocks as u64,
            self.mutations as u64,
            self.local_roots as u64,
            self.local_nodes as u64,
            self.local_guards as u64,
            self.local_leaves as u64,
            self.local_blocks as u64,
            self.local_mutations as u64,
            self.facts as u64,
            self.library as u64,
            self.frame_bytes as u64,
        ]);
        words[17] = self.statements as u64;
        words[18] = self.entry_supports as u64;
        words
    }

    pub fn from_footer(bytes: &[u8]) -> Result<Self, SemanticTransitionError> {
        if bytes.len() != IMAGE_GEOMETRY_WORDS * 8
            || word(bytes, 0)? != IMAGE_GEOMETRY_MAGIC
            || word(bytes, 1)? != 1
            || bytes[19 * 8..31 * 8].iter().any(|&byte| byte != 0)
        {
            return Err(overflow());
        }
        let mut values = [0usize; 15];
        for (index, value) in values.iter_mut().enumerate() {
            *value = usize::try_from(word(bytes, index + 2)?).map_err(|_| overflow())?;
        }
        let result = Self::geometry(
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5],
            [
                values[6], values[7], values[8], values[9], values[10], values[11],
            ],
            values[12],
            values[13],
            usize::try_from(word(bytes, 17)?).map_err(|_| overflow())?,
        )?
        .with_entry_supports(usize::try_from(word(bytes, 18)?).map_err(|_| overflow())?)?;
        if result.frame_bytes != values[14] {
            return Err(overflow());
        }
        Ok(result)
    }

    pub fn genome(self, frame: &[u8]) -> Result<[u64; 4], SemanticTransitionError> {
        if frame.len() != self.frame_bytes {
            return Err(overflow());
        }
        let mut hash = Sha256::new();
        hash.update(IMAGE_GENOME_DOMAIN);
        hash.update(&frame[6 * 8..10 * 8]);
        hash.update(&frame[4 * 8..5 * 8]);
        hash.update(&frame[self.root_offset..self.lineage_offset]);
        Ok(identity_words(hash.finalize().into()))
    }

    pub fn initial_frame(
        self,
        program: Option<&SemanticProgramAdmission>,
        owner: Option<[u64; 4]>,
    ) -> Result<Vec<u8>, SemanticTransitionError> {
        let mut bytes = vec![0; self.frame_bytes];
        put(&mut bytes, 0, IMAGE_FRAME_MAGIC);
        put(&mut bytes, 1, 1);
        let Some(program) = program else {
            return Ok(bytes);
        };
        if Self::new(Some(program))?
            .with_library(self.library)?
            .with_entry_supports(self.entry_supports)?
            != self
        {
            return Err(overflow());
        }
        let owner = owner.ok_or_else(overflow)?;
        for (index, value) in owner.into_iter().enumerate() {
            put(&mut bytes, 6 + index, value);
        }
        for (index, &root) in program.image.roots.iter().enumerate() {
            put(&mut bytes, self.root_offset / 8 + index, u64::from(root));
        }
        for (index, node) in program.image.nodes.iter().enumerate() {
            let begin = self.node_offset / 8 + 8 * index;
            match node {
                SemanticProgramNode::Unused => {}
                SemanticProgramNode::Branch { guard, children } => {
                    put(&mut bytes, begin, 1);
                    put(&mut bytes, begin + 1, u64::from(*guard));
                    for (edge, &child) in children.iter().enumerate() {
                        put(&mut bytes, begin + 2 + edge, u64::from(child));
                    }
                }
                SemanticProgramNode::Leaf { effect } => {
                    put(&mut bytes, begin, 2);
                    put(&mut bytes, begin + 6, u64::from(*effect));
                }
            }
        }
        let genome = self.genome(&bytes)?;
        for (index, value) in genome.into_iter().enumerate() {
            put(&mut bytes, 10 + index, value);
            put(&mut bytes, 26 + index, value);
        }
        Ok(bytes)
    }

    pub fn validate_frame(self, bytes: &[u8]) -> Result<(), SemanticTransitionError> {
        if bytes.len() != self.frame_bytes
            || word(bytes, 0)? != IMAGE_FRAME_MAGIC
            || word(bytes, 1)? != 1
            || bytes[34 * 8..48 * 8].iter().any(|&v| v != 0)
        {
            return Err(overflow());
        }
        let generation = word(bytes, 2)?;
        let count = usize::try_from(word(bytes, 3)?).map_err(|_| overflow())?;
        let rules = usize::try_from(word(bytes, 4)?).map_err(|_| overflow())?;
        let inherited = usize::try_from(word(bytes, 5)?).map_err(|_| overflow())?;
        if count > self.mutations
            || rules > self.mutations
            || inherited > count
            || generation < (count - inherited) as u64
        {
            return Err(overflow());
        }
        if self.roots == 0 {
            if bytes[16..].iter().any(|&value| value != 0) {
                return Err(overflow());
            }
            return Ok(());
        }
        let digest = self.genome(bytes)?;
        if (0..4).any(|index| word(bytes, 10 + index).ok() != Some(digest[index]))
            || bytes[6 * 8..10 * 8].iter().all(|&value| value == 0)
        {
            return Err(overflow());
        }
        let node =
            |index: usize, field: usize| word(bytes, self.node_offset / 8 + index * 8 + field);
        let mut active = 0;
        for root in 0..self.roots {
            let value = word(bytes, self.root_offset / 8 + root)?;
            if value == u64::from(u32::MAX) {
                continue;
            }
            if value >= self.nodes as u64 || node(value as usize, 0)? == 0 {
                return Err(overflow());
            }
            active += 1;
        }
        if active == 0 {
            return Err(overflow());
        }
        for index in 0..self.nodes {
            match node(index, 0)? {
                0 if (1..8).all(|field| node(index, field).ok() == Some(0)) => {}
                1 if node(index, 1)? < self.guards as u64
                    && node(index, 6)? == 0
                    && node(index, 7)? == 0 =>
                {
                    for field in 2..6 {
                        let child = node(index, field)?;
                        if child >= self.nodes as u64 || node(child as usize, 0)? == 0 {
                            return Err(overflow());
                        }
                    }
                }
                2 if node(index, 6)? < self.leaves as u64
                    && [1, 2, 3, 4, 5, 7]
                        .into_iter()
                        .all(|field| node(index, field).ok() == Some(0)) => {}
                _ => return Err(overflow()),
            }
        }
        // Validate every initialized node, including currently unreachable ones,
        // so a later root edit cannot activate a latent cycle.
        let mut colors = vec![0u8; self.nodes];
        let mut stack = Vec::with_capacity(self.nodes.checked_mul(5).ok_or_else(overflow)?);
        for start in 0..self.nodes {
            if colors[start] != 0 || node(start, 0)? == 0 {
                continue;
            }
            stack.push((start, false));
            while let Some((index, leave)) = stack.pop() {
                if leave {
                    colors[index] = 2;
                    continue;
                }
                match colors[index] {
                    1 => return Err(overflow()),
                    2 => continue,
                    _ => {}
                }
                colors[index] = 1;
                stack.push((index, true));
                if node(index, 0)? == 1 {
                    for field in (2..6).rev() {
                        stack.push((node(index, field)? as usize, false));
                    }
                }
            }
        }
        if bytes[self.rule_offset + rules * 100..self.block_offset]
            .iter()
            .any(|&v| v != 0)
            || bytes[self.block_offset + rules * 4..self.activity_offset]
                .iter()
                .any(|&v| v != 0)
            || bytes[self.lineage_offset + count * IMAGE_LINEAGE_WORDS * 8..]
                .iter()
                .any(|&v| v != 0)
        {
            return Err(overflow());
        }
        let permitted = if self.library == 0 {
            0
        } else {
            (1u64 << self.library) - 1
        };
        for block in 0..self.blocks {
            if word(bytes, self.activity_offset / 8 + block)? & !permitted != 0 {
                return Err(overflow());
            }
        }
        let mut parent = [0u64; 4];
        for index in 0..4 {
            parent[index] = word(bytes, 26 + index)?;
        }
        let mut previous_generation = generation
            .checked_sub((count - inherited) as u64)
            .ok_or_else(overflow)?;
        for entry in inherited..count {
            let begin = self.lineage_offset / 8 + entry * IMAGE_LINEAGE_WORDS;
            previous_generation = previous_generation.checked_add(1).ok_or_else(overflow)?;
            if word(bytes, begin)? != previous_generation
                || word(bytes, begin + 2)? >= 2
                || !(2..=6).contains(&word(bytes, begin + 3)?)
                || (0..4).any(|i| word(bytes, begin + 22 + i).ok() != Some(parent[i]))
                || (0..4).any(|i| word(bytes, begin + 30 + i).ok() != Some(parent[i]))
                || (0..4).any(|i| word(bytes, begin + 34 + i).ok() != word(bytes, 6 + i).ok())
                || (47..IMAGE_LINEAGE_WORDS).any(|i| word(bytes, begin + i).ok() != Some(0))
            {
                return Err(overflow());
            }
            let row_begin = begin * 8;
            let mutation: [u64; 4] =
                identity_words(Sha256::digest(&bytes[row_begin..row_begin + 38 * 8]).into());
            if (0..4).any(|i| word(bytes, begin + 43 + i).ok() != Some(mutation[i])) {
                return Err(overflow());
            }
            for index in 0..4 {
                parent[index] = word(bytes, begin + 26 + index)?;
            }
        }
        if parent != digest {
            return Err(overflow());
        }
        for rule in 0..rules {
            let begin = self.block_offset + 4 * rule;
            let block = u32::from_le_bytes(bytes[begin..begin + 4].try_into().unwrap());
            if block as usize >= self.blocks {
                return Err(overflow());
            }
        }
        Ok(())
    }

    /// Reconstruct only canonical origin bytes; selected semantics stay native.
    pub fn origin_genome(
        self,
        program: &SemanticProgramAdmission,
        owner: [u64; 4],
        donor: Option<(Self, &[u8])>,
    ) -> Result<[u64; 4], SemanticTransitionError> {
        let (source, bytes) = donor.unwrap_or((self, &[]));
        let inherited = donor.is_some();
        if inherited
            && (self.local_roots != source.roots
                || self.local_nodes != source.nodes
                || self.local_guards != source.guards
                || self.local_leaves != source.leaves
                || self.local_blocks != source.blocks
                || source.mutations > self.mutations)
        {
            return Err(overflow());
        }
        let rules = if inherited {
            word(bytes, 4)? as usize
        } else {
            0
        };
        let mut hash = Sha256::new();
        hash.update(IMAGE_GENOME_DOMAIN);
        for value in owner {
            hash.update(value.to_le_bytes());
        }
        hash.update((rules as u64).to_le_bytes());
        if inherited {
            hash.update(&bytes[source.root_offset..source.node_offset]);
        }
        for &root in &program.image.roots[if inherited { self.local_roots } else { 0 }..] {
            hash.update(u64::from(root).to_le_bytes());
        }
        if inherited {
            hash.update(&bytes[source.node_offset..source.rule_offset]);
        }
        for node in &program.image.nodes[if inherited { self.local_nodes } else { 0 }..] {
            let mut words = [0u64; 8];
            match node {
                SemanticProgramNode::Unused => {}
                SemanticProgramNode::Branch { guard, children } => {
                    words[0] = 1;
                    words[1] = u64::from(*guard);
                    for (index, &child) in children.iter().enumerate() {
                        words[2 + index] = u64::from(child);
                    }
                }
                SemanticProgramNode::Leaf { effect } => {
                    words[0] = 2;
                    words[6] = u64::from(*effect);
                }
            }
            for value in words {
                hash.update(value.to_le_bytes());
            }
        }
        if inherited {
            hash.update(&bytes[source.rule_offset..source.rule_offset + rules * 100]);
        }
        for _ in rules..self.mutations {
            hash.update([0u8; 100]);
        }
        if inherited {
            hash.update(&bytes[source.block_offset..source.block_offset + rules * 4]);
        }
        for _ in rules..self.mutations {
            hash.update([0u8; 4]);
        }
        if inherited {
            hash.update(&bytes[source.activity_offset..source.lineage_offset]);
        }
        for _ in (if inherited { self.local_blocks } else { 0 })..self.blocks {
            hash.update([0u8; 8]);
        }
        Ok(identity_words(hash.finalize().into()))
    }

    pub fn require_donor_prefix(
        self,
        target: &[u8],
        source: Self,
        bytes: &[u8],
        program: &SemanticProgramAdmission,
        owner: [u64; 4],
    ) -> Result<(), SemanticTransitionError> {
        source.validate_frame(bytes)?;
        self.validate_frame(target)?;
        let rules = word(bytes, 4)? as usize;
        let count = word(bytes, 3)? as usize;
        let genesis = self.origin_genome(program, owner, Some((source, bytes)))?;
        if word(target, 5)? != count as u64
            || word(target, 4)? < rules as u64
            || word(target, 2)?.checked_sub(word(target, 3)? - count as u64)
                != Some(word(bytes, 2)?)
            || (0..4).any(|i| {
                word(target, 22 + i).ok() != word(bytes, 10 + i).ok()
                    || word(target, 26 + i).ok() != Some(genesis[i])
            })
            || target[self.root_offset..self.root_offset + source.roots * 8]
                != bytes[source.root_offset..source.node_offset]
            || target[self.node_offset..self.node_offset + source.nodes * 64]
                != bytes[source.node_offset..source.rule_offset]
            || target[self.rule_offset..self.rule_offset + rules * 100]
                != bytes[source.rule_offset..source.rule_offset + rules * 100]
            || target[self.block_offset..self.block_offset + rules * 4]
                != bytes[source.block_offset..source.block_offset + rules * 4]
            || target[self.activity_offset..self.activity_offset + source.blocks * 8]
                != bytes[source.activity_offset..source.lineage_offset]
            || target[self.lineage_offset..self.lineage_offset + count * IMAGE_LINEAGE_WORDS * 8]
                != bytes
                    [source.lineage_offset..source.lineage_offset + count * IMAGE_LINEAGE_WORDS * 8]
        {
            return Err(publication_input_error(
                "restored image changed its original selected donor topology or lineage",
            ));
        }
        Ok(())
    }

    /// Carry the complete selected donor genome and its history into the
    /// disjoint source half of one newly admitted evaluation image.
    pub fn inherit_source(
        self,
        target: &mut [u8],
        source_layout: Self,
        source: &[u8],
    ) -> Result<(), SemanticTransitionError> {
        source_layout.validate_frame(source)?;
        if target.len() != self.frame_bytes
            || self.local_roots != source_layout.roots
            || self.local_nodes != source_layout.nodes
            || self.local_guards != source_layout.guards
            || self.local_leaves != source_layout.leaves
            || self.local_blocks != source_layout.blocks
            || source_layout.mutations > self.mutations
        {
            return Err(overflow());
        }
        target[self.root_offset..self.root_offset + source_layout.roots * 8]
            .copy_from_slice(&source[source_layout.root_offset..source_layout.node_offset]);
        target[self.node_offset..self.node_offset + source_layout.nodes * 64]
            .copy_from_slice(&source[source_layout.node_offset..source_layout.rule_offset]);
        let rules = usize::try_from(word(source, 4)?).map_err(|_| overflow())?;
        let count = usize::try_from(word(source, 3)?).map_err(|_| overflow())?;
        target[self.rule_offset..self.rule_offset + rules * 100].copy_from_slice(
            &source[source_layout.rule_offset..source_layout.rule_offset + rules * 100],
        );
        target[self.block_offset..self.block_offset + rules * 4].copy_from_slice(
            &source[source_layout.block_offset..source_layout.block_offset + rules * 4],
        );
        target[self.activity_offset..self.activity_offset + source_layout.blocks * 8]
            .copy_from_slice(&source[source_layout.activity_offset..source_layout.lineage_offset]);
        target[self.lineage_offset..self.lineage_offset + count * IMAGE_LINEAGE_WORDS * 8]
            .copy_from_slice(
                &source[source_layout.lineage_offset
                    ..source_layout.lineage_offset + count * IMAGE_LINEAGE_WORDS * 8],
            );
        put(target, 2, word(source, 2)?);
        put(target, 3, count as u64);
        put(target, 4, rules as u64);
        put(target, 5, count as u64);
        for index in 0..4 {
            put(target, 22 + index, word(source, 10 + index)?);
        }
        let digest = self.genome(target)?;
        for (index, value) in digest.into_iter().enumerate() {
            put(target, 10 + index, value);
            put(target, 26 + index, value);
        }
        self.validate_frame(target)
    }
}
