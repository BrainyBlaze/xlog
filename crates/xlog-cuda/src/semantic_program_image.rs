//! Immutable decision-forest images shared by cold lowering and resident execution.

use crate::SemanticProgramFact;

/// One node in the preallocated image pool. Unused slots own no semantic node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SemanticProgramNode {
    Unused,
    Branch { guard: u32, children: [u32; 4] },
    Leaf { effect: u32 },
}

/// A leaf's staged effect. Evidence references the image's original guard roster.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SemanticProgramLeaf {
    Abstain,
    Evidence(Vec<(u32, u32)>),
    RuleBlock(u32),
}

/// One immutable, context-qualified image of the rule arena and decision forest.
///
/// Indices address this image, never a process pointer or a Python object. The
/// resident selected bank owns the mutable successor, its generation and lineage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticProgramImageAdmission {
    pub roots: Vec<u32>,
    pub nodes: Vec<SemanticProgramNode>,
    pub guards: Vec<SemanticProgramFact>,
    pub guard_records: Vec<u32>,
    pub leaves: Vec<SemanticProgramLeaf>,
    pub rule_blocks: Vec<Vec<u32>>,
    pub mutation_capacity: u32,
    pub local_mutation_capacity: u32,
    pub local_root_begin: u32,
    pub local_node_begin: u32,
    pub local_guard_begin: u32,
    pub local_leaf_begin: u32,
    pub local_block_begin: u32,
}

pub(crate) const PROGRAM_IMAGE_HEADER_WORDS: usize = 16;
pub(crate) const PROGRAM_IMAGE_NODE_WORDS: usize = 8;
pub(crate) const PROGRAM_IMAGE_LEAF_WORDS: usize = 4;
pub(crate) const PROGRAM_IMAGE_NULL: u32 = u32::MAX;

impl SemanticProgramImageAdmission {
    /// The flat representation is one ordinary forest leaf invoking one block.
    /// Its rules still execute through the same forest interpreter.
    pub fn flat(
        guards: &[SemanticProgramFact],
        guard_records: &[u32],
        rules: usize,
    ) -> Result<Self, &'static str> {
        let rules = u32::try_from(rules).map_err(|_| "program rule arena exceeds u32")?;
        Ok(Self {
            roots: vec![0],
            nodes: vec![SemanticProgramNode::Leaf { effect: 1 }],
            guards: guards.to_vec(),
            guard_records: guard_records.to_vec(),
            leaves: vec![
                SemanticProgramLeaf::Abstain,
                SemanticProgramLeaf::RuleBlock(0),
            ],
            rule_blocks: vec![(0..rules).collect()],
            mutation_capacity: 0,
            local_mutation_capacity: 0,
            local_root_begin: 0,
            local_node_begin: 0,
            local_guard_begin: 0,
            local_leaf_begin: 0,
            local_block_begin: 0,
        })
    }

    /// Reserve physical slots from the actual finite mutation budget. A reserved
    /// slot is not a root or node until the native decoder initializes it.
    pub fn reserve_mutations(&mut self, mutations: u32) -> Result<(), &'static str> {
        if self.mutation_capacity != 0
            || self.roots.iter().any(|&root| root == PROGRAM_IMAGE_NULL)
            || self.nodes.iter().any(|node| matches!(node, SemanticProgramNode::Unused))
        {
            return Err("program mutation storage is already bound");
        }
        let roots = self
            .roots
            .len()
            .checked_add(mutations as usize)
            .filter(|&count| count < u32::MAX as usize)
            .ok_or("program root pool exceeds u32")?;
        let nodes = self
            .nodes
            .len()
            .checked_add(mutations as usize)
            .filter(|&count| count < u32::MAX as usize)
            .ok_or("program node pool exceeds u32")?;
        self.roots.resize(roots, PROGRAM_IMAGE_NULL);
        self.nodes.resize(nodes, SemanticProgramNode::Unused);
        self.mutation_capacity = mutations;
        self.local_mutation_capacity = mutations;
        Ok(())
    }

    pub fn validate(
        &self,
        predicate_arities: &[u32],
        rule_count: usize,
    ) -> Result<(), &'static str> {
        for count in [
            self.roots.len(),
            self.nodes.len(),
            self.guards.len(),
            self.leaves.len(),
            self.rule_blocks.len(),
        ] {
            if count == 0 || count >= u32::MAX as usize {
                return Err("program image has an empty or unaddressable roster");
            }
        }
        if self.guards.len() != self.guard_records.len()
            || self.local_root_begin as usize >= self.roots.len()
            || self.local_node_begin as usize >= self.nodes.len()
            || self.local_guard_begin as usize >= self.guards.len()
            || self.local_leaf_begin as usize >= self.leaves.len()
            || self.local_block_begin as usize >= self.rule_blocks.len()
        {
            return Err("program image changes its original context ranges");
        }
        for fact in &self.guards {
            if !(1..=2).contains(&fact.arity)
                || predicate_arities.get(fact.predicate as usize) != Some(&fact.arity)
                || (fact.arity == 1 && fact.second != 0)
            {
                return Err("program guard is not an admitted typed query");
            }
        }
        for block in &self.rule_blocks {
            if block.iter().any(|&rule| rule as usize >= rule_count)
                || block.iter().enumerate().any(|(index, rule)| block[..index].contains(rule))
            {
                return Err("program rule block changes its original rule references");
            }
        }
        for leaf in &self.leaves {
            match leaf {
                SemanticProgramLeaf::Abstain => {}
                SemanticProgramLeaf::RuleBlock(block) => {
                    if *block as usize >= self.rule_blocks.len() {
                        return Err("program leaf names an absent rule block");
                    }
                }
                SemanticProgramLeaf::Evidence(events) => {
                    if events.is_empty()
                        || events.iter().any(|&(guard, polarity)| {
                            guard as usize >= self.guards.len() || !(1..=2).contains(&polarity)
                        })
                        || events.iter().enumerate().any(|(index, event)| events[..index].contains(event))
                    {
                        return Err("program evidence leaf has an invalid support event");
                    }
                }
            }
        }
        for node in &self.nodes {
            match node {
                SemanticProgramNode::Unused => {}
                SemanticProgramNode::Branch { guard, children } => {
                    if *guard as usize >= self.guards.len()
                        || children.iter().any(|&child| {
                            self.nodes.get(child as usize).is_none_or(|node| {
                                matches!(node, SemanticProgramNode::Unused)
                            })
                        })
                    {
                        return Err("program branch has an absent guard or child");
                    }
                }
                SemanticProgramNode::Leaf { effect } => {
                    if *effect as usize >= self.leaves.len() {
                        return Err("program node names an absent leaf effect");
                    }
                }
            }
        }
        if !self.roots.iter().any(|&root| root != PROGRAM_IMAGE_NULL)
            || self.roots.iter().any(|&root| {
                root != PROGRAM_IMAGE_NULL
                    && self.nodes.get(root as usize).is_none_or(|node| {
                        matches!(node, SemanticProgramNode::Unused)
                    })
            })
        {
            return Err("program image has no live tree or an absent root node");
        }
        // Validate all initialized nodes, including a not-yet-rooted invention.
        // A completed guard walk can therefore never conceal a dangling cycle.
        let mut colors = vec![0u8; self.nodes.len()];
        let stack_capacity = self.nodes.len().checked_mul(5)
            .ok_or("program traversal allocation exceeds addressable storage")?;
        let mut stack = Vec::with_capacity(stack_capacity);
        for start in 0..self.nodes.len() {
            if colors[start] != 0 || matches!(self.nodes[start], SemanticProgramNode::Unused) {
                continue;
            }
            stack.push((start, false));
            while let Some((node, leaving)) = stack.pop() {
                if leaving {
                    colors[node] = 2;
                    continue;
                }
                if colors[node] == 2 {
                    continue;
                }
                if colors[node] == 1 {
                    return Err("program image contains a decision cycle");
                }
                colors[node] = 1;
                stack.push((node, true));
                if let SemanticProgramNode::Branch { children, .. } = &self.nodes[node] {
                    for &child in children.iter().rev() {
                        if colors[child as usize] == 1 {
                            return Err("program image contains a decision cycle");
                        }
                        if colors[child as usize] == 0 {
                            stack.push((child as usize, false));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Compose a disjoint target after the complete source pool. All source
    /// logical slots, including reserved ones, keep their original coordinates.
    pub(crate) fn evaluation_context(
        source: &Self,
        target: &Self,
        source_rule_count: usize,
        target_record_offset: u32,
        map_fact: impl Fn(SemanticProgramFact) -> Result<SemanticProgramFact, &'static str>,
    ) -> Result<Self, &'static str> {
        let count = |value: usize| {
            u32::try_from(value).map_err(|_| "program context exceeds u32")
        };
        let node_offset = count(source.nodes.len())?;
        let guard_offset = count(source.guards.len())?;
        let leaf_offset = count(source.leaves.len())?;
        let block_offset = count(source.rule_blocks.len())?;
        let rule_offset = count(source_rule_count)?;
        let add = |value: u32, offset: u32| {
            value.checked_add(offset).ok_or("program context index exceeds u32")
        };
        let mut result = source.clone();
        result.local_root_begin = count(source.roots.len())?;
        result.local_node_begin = node_offset;
        result.local_guard_begin = guard_offset;
        result.local_leaf_begin = leaf_offset;
        result.local_block_begin = block_offset;
        result.mutation_capacity = source
            .mutation_capacity
            .checked_add(target.mutation_capacity)
            .ok_or("program context mutation budget exceeds u32")?;
        for &root in &target.roots {
            result.roots.push(if root == PROGRAM_IMAGE_NULL {
                PROGRAM_IMAGE_NULL
            } else {
                add(root, node_offset)?
            });
        }
        for node in &target.nodes {
            result.nodes.push(match node {
                SemanticProgramNode::Unused => SemanticProgramNode::Unused,
                SemanticProgramNode::Branch { guard, children } => {
                    let mut mapped = [0; 4];
                    for (output, &child) in mapped.iter_mut().zip(children) {
                        *output = add(child, node_offset)?;
                    }
                    SemanticProgramNode::Branch {
                        guard: add(*guard, guard_offset)?,
                        children: mapped,
                    }
                }
                SemanticProgramNode::Leaf { effect } => SemanticProgramNode::Leaf {
                    effect: add(*effect, leaf_offset)?,
                },
            });
        }
        for &fact in &target.guards {
            result.guards.push(map_fact(fact)?);
        }
        for &record in &target.guard_records {
            result.guard_records.push(add(record, target_record_offset)?);
        }
        for leaf in &target.leaves {
            result.leaves.push(match leaf {
                SemanticProgramLeaf::Abstain => SemanticProgramLeaf::Abstain,
                SemanticProgramLeaf::RuleBlock(block) => {
                    SemanticProgramLeaf::RuleBlock(add(*block, block_offset)?)
                }
                SemanticProgramLeaf::Evidence(events) => SemanticProgramLeaf::Evidence(
                    events.iter().map(|&(guard, polarity)| {
                        Ok((add(guard, guard_offset)?, polarity))
                    }).collect::<Result<_, &'static str>>()?
                ),
            });
        }
        for block in &target.rule_blocks {
            result.rule_blocks.push(
                block.iter().map(|&rule| add(rule, rule_offset)).collect::<Result<_, _>>()?
            );
        }
        result.local_mutation_capacity = target.local_mutation_capacity;
        Ok(result)
    }

    pub(crate) fn allocation_bytes(&self) -> Result<usize, &'static str> {
        let overflow = || "program image allocation exceeds addressable storage";
        let mut bytes = std::mem::size_of::<Self>();
        for (count, width) in [
            (self.roots.len(), std::mem::size_of::<u32>()),
            (self.nodes.len(), std::mem::size_of::<SemanticProgramNode>()),
            (self.guards.len(), std::mem::size_of::<SemanticProgramFact>()),
            (self.guard_records.len(), std::mem::size_of::<u32>()),
            (self.leaves.len(), std::mem::size_of::<SemanticProgramLeaf>()),
            (self.rule_blocks.len(), std::mem::size_of::<Vec<u32>>()),
            (self.nodes.len(), std::mem::size_of::<u8>()),
        ] {
            bytes = count.checked_mul(width).and_then(|n| bytes.checked_add(n))
                .ok_or_else(overflow)?;
        }
        // One deferred leave and four child visits per initialized branch.
        bytes = self.nodes.len().checked_mul(5)
            .and_then(|n| n.checked_mul(std::mem::size_of::<(usize, bool)>()))
            .and_then(|n| bytes.checked_add(n)).ok_or_else(overflow)?;
        for leaf in &self.leaves {
            if let SemanticProgramLeaf::Evidence(events) = leaf {
                bytes = events.len().checked_mul(std::mem::size_of::<(u32, u32)>())
                    .and_then(|n| bytes.checked_add(n)).ok_or_else(overflow)?;
            }
        }
        for block in &self.rule_blocks {
            bytes = block.len().checked_mul(std::mem::size_of::<u32>())
                .and_then(|n| bytes.checked_add(n)).ok_or_else(overflow)?;
        }
        Ok(bytes)
    }

    pub(crate) fn word_count(&self) -> Option<usize> {
        let events = self.leaves.iter().try_fold(0usize, |sum, leaf| {
            sum.checked_add(match leaf {
                SemanticProgramLeaf::Evidence(events) => events.len(),
                _ => 0,
            })
        })?;
        let references = self.rule_blocks.iter().try_fold(0usize, |sum, block| {
            sum.checked_add(block.len())
        })?;
        PROGRAM_IMAGE_HEADER_WORDS
            .checked_add(self.roots.len())?
            .checked_add(self.nodes.len().checked_mul(PROGRAM_IMAGE_NODE_WORDS)?)?
            .checked_add(self.guards.len().checked_mul(5)?)?
            .checked_add(self.leaves.len().checked_mul(PROGRAM_IMAGE_LEAF_WORDS)?)?
            .checked_add(events.checked_mul(2)?)?
            .checked_add(self.rule_blocks.len().checked_mul(2)?)?
            .checked_add(references)
    }

    pub(crate) fn visit_words(&self, mut visit: impl FnMut(u64)) {
        let events: usize = self.leaves.iter().map(|leaf| match leaf {
            SemanticProgramLeaf::Evidence(events) => events.len(),
            _ => 0,
        }).sum();
        let references: usize = self.rule_blocks.iter().map(Vec::len).sum();
        for word in [
            0x584c4f47464f5253,
            1,
            self.roots.len() as u64,
            self.nodes.len() as u64,
            self.guards.len() as u64,
            self.leaves.len() as u64,
            self.rule_blocks.len() as u64,
            events as u64,
            references as u64,
            u64::from(self.mutation_capacity),
            u64::from(self.local_root_begin),
            u64::from(self.local_node_begin),
            u64::from(self.local_guard_begin),
            u64::from(self.local_leaf_begin),
            u64::from(self.local_block_begin),
            u64::from(self.local_mutation_capacity),
        ] {
            visit(word);
        }
        for &root in &self.roots {
            visit(u64::from(root));
        }
        for node in &self.nodes {
            let mut words = [0; PROGRAM_IMAGE_NODE_WORDS];
            match node {
                SemanticProgramNode::Unused => {}
                SemanticProgramNode::Branch { guard, children } => {
                    words[0] = 1;
                    words[1] = u64::from(*guard);
                    for (word, &child) in words[2..6].iter_mut().zip(children) {
                        *word = u64::from(child);
                    }
                }
                SemanticProgramNode::Leaf { effect } => {
                    words[0] = 2;
                    words[6] = u64::from(*effect);
                }
            }
            for word in words {
                visit(word);
            }
        }
        for (&fact, &record) in self.guards.iter().zip(&self.guard_records) {
            for word in [fact.predicate, fact.arity, fact.first, fact.second, record] {
                visit(u64::from(word));
            }
        }
        let mut event_offset = 0usize;
        for leaf in &self.leaves {
            let words = match leaf {
                SemanticProgramLeaf::Abstain => [0, 0, 0, 0],
                SemanticProgramLeaf::RuleBlock(block) => [2, u64::from(*block), 0, 0],
                SemanticProgramLeaf::Evidence(events) => {
                    let words = [1, event_offset as u64, events.len() as u64, 0];
                    event_offset += events.len();
                    words
                }
            };
            for word in words {
                visit(word);
            }
        }
        for leaf in &self.leaves {
            if let SemanticProgramLeaf::Evidence(events) = leaf {
                for &(guard, polarity) in events {
                    visit(u64::from(guard));
                    visit(u64::from(polarity));
                }
            }
        }
        let mut reference_offset = 0usize;
        for block in &self.rule_blocks {
            visit(reference_offset as u64);
            visit(block.len() as u64);
            reference_offset += block.len();
        }
        for block in &self.rule_blocks {
            for &rule in block {
                visit(u64::from(rule));
            }
        }
    }
}
