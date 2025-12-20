// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod layer;
pub mod manager;
pub mod reserved;
pub mod reuse;
pub mod sequence;
pub mod storage;

// #[cfg(feature = "cuda_kv")]
// pub mod storage;

use reserved::*;

use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    sync::{atomic::AtomicU64, Arc, RwLock},
};

use async_trait::async_trait;
use derive_getters::Dissolve;
use dynamo_runtime::{
    raise,
    utils::pool::{PoolExt, PoolItem, PoolValue, Returnable, SharedPoolItem},
    Result,
};

use crate::tokens::{PartialTokenBlock, SequenceHash, TokenBlock, Tokens};

use tracing as log;

pub type UniqueBlock = PoolItem<KvBlock>;
pub type SharedBlock = SharedPoolItem<KvBlock>;

#[derive(Default)]
pub struct KvBlock {
    token_block: TokenBlock,
    priority: u32,
    return_tick: u64,
    /// Remaining reuses: higher values mean higher priority (less likely to be evicted)
    /// This is the primary sorting key for eviction
    remaining_reuses: u32,
}

// pub struct KvStorage {
//     data: u64,
//     size: usize,

//     layer_idx: usize,
//     block_idx: usize,

//     /// The layout of the tensor
//     layout: layer::KvLayer,
// }

impl KvBlock {
    /// Creates a new KvBlock with the given token block
    pub fn new(token_block: TokenBlock) -> Self {
        Self {
            token_block,
            priority: 0,
            return_tick: 0,
            remaining_reuses: 0,
            // storage: None,
        }
    }

    /// Updates the token block
    pub fn update_token_block(&mut self, token_block: TokenBlock) {
        self.token_block = token_block;
    }

    /// Updates remaining_reuses using max() logic
    pub fn update_remaining_reuses(&mut self, new_remaining_reuses: u32) {
        let final_value = self.remaining_reuses.max(new_remaining_reuses);
        log::error!(
            old_remaining_reuses = self.remaining_reuses,
            new_remaining_reuses = new_remaining_reuses,
            final_remaining_reuses = final_value,
            sequence_hash = self.token_block.sequence_hash(),
            "KV_REUSE: Updating KvBlock remaining_reuses with max()"
        );
        self.remaining_reuses = final_value;
    }

    /// Gets the remaining reuses for this block
    pub fn remaining_reuses(&self) -> u32 {
        self.remaining_reuses
    }

    /// Resets the block to its initial state
    pub(crate) fn reset(&mut self) {
        self.token_block = TokenBlock::default();
        self.priority = 0;
        self.return_tick = 0;
        self.remaining_reuses = 0;
        // self.storage = None;
        // self.storage_state = StorageState::Absent;
    }
}

impl Returnable for KvBlock {
    fn on_return(&mut self) {}
}

pub struct KvBlockConfig {}
