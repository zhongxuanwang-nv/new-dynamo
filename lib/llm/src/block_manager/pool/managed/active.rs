// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::block_manager::block::locality::LocalityProvider;

use super::*;

/// Manages active blocks being used by sequences
pub struct ActiveBlockPool<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    pub(super) map: HashMap<SequenceHash, Weak<MutableBlock<S, L, M>>>,
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> Default for ActiveBlockPool<S, L, M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> ActiveBlockPool<S, L, M> {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    pub fn register(
        &mut self,
        mut block: MutableBlock<S, L, M>,
    ) -> Result<ImmutableBlock<S, L, M>, BlockPoolError> {
        if !block.state().is_registered() {
            return Err(BlockPoolError::InvalidMutableBlock(
                "block is not registered".to_string(),
            ));
        }

        let block_id = block.block_id();
        let sequence_hash = block.sequence_hash().map_err(|_| {
            BlockPoolError::InvalidMutableBlock("block has no sequence hash".to_string())
        })?;

        tracing::error!(
            block_id = block_id,
            sequence_hash = sequence_hash,
            active_pool_size = self.map.len(),
            "ACTIVE_POOL_DEBUG: 📥 register() called - attempting to add block to active pool"
        );

        // Set the parent of the block if it has one.
        // This is needed to ensure the lifetime of the parent is at least as long as the child.
        if let Ok(Some(parent)) = block.parent_sequence_hash()
            && let Some(parent_block) = self.match_sequence_hash(parent)
        {
            tracing::error!(
                block_id = block_id,
                parent_hash = parent,
                parent_block_id = parent_block.block_id(),
                "ACTIVE_POOL_DEBUG: 🔗 Setting parent block"
            );
            block.set_parent(parent_block.mutable_block().clone());
        }

        let shared = Arc::new(block);

        match self.map.entry(sequence_hash) {
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                let weak = entry.get();
                if let Some(arc) = weak.upgrade() {
                    tracing::error!(
                        block_id = block_id,
                        sequence_hash = sequence_hash,
                        "ACTIVE_POOL_DEBUG: ♻️ REUSED - block already in active pool (upgraded weak ref)"
                    );
                    Ok(ImmutableBlock::new(arc))
                } else {
                    // Weak reference is no longer alive, update it in the map
                    tracing::error!(
                        block_id = block_id,
                        sequence_hash = sequence_hash,
                        "ACTIVE_POOL_DEBUG: 🔄 REPLACED - weak ref dead, replacing in active pool"
                    );
                    entry.insert(Arc::downgrade(&shared));
                    Ok(ImmutableBlock::new(shared))
                }
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                tracing::error!(
                    block_id = block_id,
                    sequence_hash = sequence_hash,
                    new_pool_size = self.map.len() + 1,
                    "ACTIVE_POOL_DEBUG: ✨ NEW - block added to active pool"
                );
                entry.insert(Arc::downgrade(&shared));
                Ok(ImmutableBlock::new(shared))
            }
        }
    }

    pub fn remove(&mut self, block: &mut Block<S, L, M>) {
        let block_id = block.block_id();
        if let Ok(sequence_hash) = block.sequence_hash()
            && let Some(weak) = self.map.get(&sequence_hash)
        {
            if let Some(_arc) = weak.upgrade() {
                tracing::error!(
                    block_id = block_id,
                    sequence_hash = sequence_hash,
                    "ACTIVE_POOL_DEBUG: 🔒 STILL REFERENCED - block has other refs, resetting state only"
                );
                block.reset();
                return;
            }
            tracing::error!(
                block_id = block_id,
                sequence_hash = sequence_hash,
                remaining_pool_size = self.map.len() - 1,
                "ACTIVE_POOL_DEBUG: 📤 REMOVED - block removed from active pool (no more refs)"
            );
            self.map.remove(&sequence_hash);
        } else {
            tracing::error!(
                block_id = block_id,
                "ACTIVE_POOL_DEBUG: ⚠️ NOT FOUND - block not in active pool map"
            );
        }
    }

    pub fn match_sequence_hash(
        &mut self,
        sequence_hash: SequenceHash,
    ) -> Option<ImmutableBlock<S, L, M>> {
        if let Some(weak) = self.map.get(&sequence_hash) {
            if let Some(arc) = weak.upgrade() {
                tracing::error!(
                    sequence_hash = sequence_hash,
                    block_id = arc.block_id(),
                    "ACTIVE_POOL_DEBUG: 🎯 MATCH HIT - found block in active pool"
                );
                Some(ImmutableBlock::new(arc))
            } else {
                // Weak reference is no longer alive, remove it from the map
                tracing::error!(
                    sequence_hash = sequence_hash,
                    "ACTIVE_POOL_DEBUG: 💀 STALE REF - weak ref expired, removing from map"
                );
                self.map.remove(&sequence_hash);
                None
            }
        } else {
            tracing::trace!(
                sequence_hash = sequence_hash,
                "ACTIVE_POOL_DEBUG: ❌ MATCH MISS - block not in active pool"
            );
            None
        }
    }

    pub fn status(&self) -> usize {
        self.map.keys().len()
    }
}
