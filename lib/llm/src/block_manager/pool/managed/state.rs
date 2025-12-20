// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::block_manager::{
    block::{BlockState, PrivateBlockExt, registry::BlockRegistrationError},
    events::Publisher,
};

use super::*;

use active::ActiveBlockPool;
use inactive::InactiveBlockPool;

impl<S: Storage, L: LocalityProvider + 'static, M: BlockMetadata> State<S, L, M> {
    pub fn new(
        event_manager: Arc<dyn EventManager>,
        return_tx: tokio::sync::mpsc::UnboundedSender<Block<S, L, M>>,
        global_registry: GlobalRegistry,
        async_runtime: Handle,
    ) -> Self {
        Self {
            active: ActiveBlockPool::new(),
            inactive: InactiveBlockPool::new(),
            registry: BlockRegistry::new(event_manager.clone(), global_registry, async_runtime),
            return_tx,
            event_manager,
        }
    }

    pub async fn handle_priority_request(
        &mut self,
        req: PriorityRequest<S, L, M>,
        return_rx: &mut tokio::sync::mpsc::UnboundedReceiver<Block<S, L, M>>,
    ) {
        match req {
            PriorityRequest::AllocateBlocks(req) => {
                let (count, resp_tx) = req.dissolve();
                let blocks = self.allocate_blocks(count);
                if resp_tx.send(blocks).is_err() {
                    tracing::error!("failed to send response to allocate blocks");
                }
            }
            PriorityRequest::RegisterBlocks(req) => {
                let ((blocks, duplication_setting), resp_tx) = req.dissolve();
                let immutable_blocks = self
                    .register_blocks(blocks, duplication_setting, return_rx)
                    .await;
                if resp_tx.send(immutable_blocks).is_err() {
                    tracing::error!("failed to send response to register blocks");
                }
            }
            PriorityRequest::MatchSequenceHashes(req) => {
                let (sequence_hashes, resp_tx) = req.dissolve();
                let immutable_blocks = self.match_sequence_hashes(sequence_hashes, return_rx).await;
                if resp_tx.send(Ok(immutable_blocks)).is_err() {
                    tracing::error!("failed to send response to match sequence hashes");
                }
            }
            PriorityRequest::TouchBlocks(req) => {
                let (sequence_hashes, resp_tx) = req.dissolve();
                self.touch_blocks(&sequence_hashes, return_rx).await;
                if resp_tx.send(Ok(())).is_err() {
                    tracing::error!("failed to send response to touch blocks");
                }
            }
            PriorityRequest::Reset(req) => {
                let (_req, resp_tx) = req.dissolve();
                let result = self.inactive.reset();
                if resp_tx.send(result).is_err() {
                    tracing::error!("failed to send response to reset");
                }
            }
            PriorityRequest::ReturnBlock(req) => {
                let (returnable_blocks, resp_tx) = req.dissolve();
                for block in returnable_blocks {
                    self.return_block(block);
                }
                if resp_tx.send(Ok(())).is_err() {
                    tracing::error!("failed to send response to return block");
                }
            }
        }
    }

    pub fn handle_control_request(&mut self, req: ControlRequest<S, L, M>) {
        match req {
            ControlRequest::AddBlocks(blocks) => {
                let (blocks, resp_rx) = blocks.dissolve();
                self.inactive.add_blocks(blocks);
                if resp_rx.send(()).is_err() {
                    tracing::error!("failed to send response to add blocks");
                }
            }
            ControlRequest::Status(req) => {
                let (_, resp_rx) = req.dissolve();
                if resp_rx.send(Ok(self.status())).is_err() {
                    tracing::error!("failed to send response to status");
                }
            }
            ControlRequest::ResetBlocks(req) => {
                let (sequence_hashes, resp_rx) = req.dissolve();
                if resp_rx
                    .send(Ok(self.try_reset_blocks(&sequence_hashes)))
                    .is_err()
                {
                    tracing::error!("failed to send response to reset blocks");
                }
            }
        }
    }

    pub fn handle_return_block(&mut self, block: Block<S, L, M>) {
        self.return_block(block);
    }

    /// We have a strong guarantee that the block will be returned to the pool in the near future.
    /// The caller must take ownership of the block
    async fn wait_for_returned_block(
        &mut self,
        sequence_hash: SequenceHash,
        return_rx: &mut tokio::sync::mpsc::UnboundedReceiver<Block<S, L, M>>,
    ) -> Block<S, L, M> {
        while let Some(block) = return_rx.recv().await {
            if matches!(block.state(), BlockState::Registered(handle, _) if handle.sequence_hash() == sequence_hash)
            {
                return block;
            }
            self.handle_return_block(block);
        }

        unreachable!("this should be unreachable");
    }

    pub fn allocate_blocks(
        &mut self,
        count: usize,
    ) -> Result<Vec<MutableBlock<S, L, M>>, BlockPoolError> {
        let available_blocks = self.inactive.available_blocks() as usize;
        let active_count = self.active.status();

        tracing::error!(
            requested = count,
            available = available_blocks,
            active_blocks = active_count,
            "POOL_DEBUG: 📦 allocate_blocks called - pool state before allocation"
        );

        if available_blocks < count {
            tracing::error!(
                requested = count,
                available = available_blocks,
                active_blocks = active_count,
                "POOL_DEBUG: ❌ NOT ENOUGH BLOCKS - allocation failed"
            );
            return Err(BlockPoolError::NotEnoughBlocksAvailable(
                count,
                available_blocks,
            ));
        }

        let mut blocks = Vec::with_capacity(count);
        let mut allocated_ids = Vec::with_capacity(count);

        for _ in 0..count {
            if let Some(block) = self.inactive.acquire_free_block() {
                allocated_ids.push(block.block_id());
                blocks.push(MutableBlock::new(block, self.return_tx.clone()));
            }
        }

        tracing::error!(
            count = blocks.len(),
            block_ids = ?allocated_ids,
            remaining_available = self.inactive.available_blocks(),
            "POOL_DEBUG: ✅ INACTIVE→MUTABLE - blocks allocated from inactive pool"
        );

        Ok(blocks)
    }

    #[tracing::instrument(level = "debug", skip_all, fields(blocks = ?blocks))]
    pub async fn register_blocks(
        &mut self,
        blocks: Vec<MutableBlock<S, L, M>>,
        duplication_setting: BlockRegistrationDuplicationSetting,
        return_rx: &mut tokio::sync::mpsc::UnboundedReceiver<Block<S, L, M>>,
    ) -> Result<Vec<ImmutableBlock<S, L, M>>, BlockPoolError> {
        assert!(!blocks.is_empty(), "no blocks to register");

        let expected_len = blocks.len();
        let block_ids: Vec<_> = blocks.iter().map(|b| b.block_id()).collect();

        tracing::error!(
            num_blocks = expected_len,
            block_ids = ?block_ids,
            active_before = self.active.status(),
            inactive_before = self.inactive.available_blocks(),
            "POOL_DEBUG: 📝 register_blocks called - registering blocks to active pool"
        );

        let mut immutable_blocks = Vec::new();

        // raii object that will collect all the publish handles and publish them when the object is dropped
        let mut publish_handles = self.publisher();

        for mut block in blocks.into_iter() {
            let sequence_hash = block.sequence_hash()?;
            let block_id = block.block_id();

            // If the block is already registered, acquire a clone of the immutable block
            if let Some(immutable) = self.active.match_sequence_hash(sequence_hash) {
                tracing::error!(
                    block_id = block_id,
                    sequence_hash = sequence_hash,
                    "POOL_DEBUG: 🔄 ALREADY ACTIVE - block already in active pool"
                );
                let immutable = if duplication_setting
                    == BlockRegistrationDuplicationSetting::Allowed
                {
                    immutable.with_duplicate(block.into()).expect("incompatible immutable block; only primary should be returned from match_sequence_hash")
                } else {
                    // immediate return the block to the pool if duplicates are disabled
                    if let Some(blocks) = block.try_take_block(private::PrivateToken) {
                        self.inactive.return_blocks(blocks);
                    }
                    immutable
                };

                immutable_blocks.push(immutable);
                continue;
            }

            let mut offload = true;

            let (mutable, duplicate) =
                if let Some(raw_block) = self.inactive.match_sequence_hash(sequence_hash) {
                    // We already have a match, so our block is a duplicate.
                    assert!(matches!(raw_block.state(), BlockState::Registered(_, _)));
                    (
                        MutableBlock::new(raw_block, self.return_tx.clone()),
                        Some(block),
                    )
                } else {
                    // Attempt to register the block
                    // On the very rare chance that the block is registered, but in the process of being returned,
                    // we will wait for it to be returned and then register it.
                    let result = block.register(&mut self.registry);

                    match result {
                        Ok(handle) => {
                            // Only create our publish handle if this block is new, and not transfered.
                            if let Some(handle) = handle {
                                publish_handles.take_handle(handle);
                            }
                            (block, None)
                        }
                        Err(BlockRegistrationError::BlockAlreadyRegistered(_)) => {
                            // Block is already registered, wait for it to be returned
                            // Return the original block as the primary, and the block we passed in as the duplicate.
                            offload = false;
                            let raw_block =
                                self.wait_for_returned_block(sequence_hash, return_rx).await;
                            (
                                MutableBlock::new(raw_block, self.return_tx.clone()),
                                Some(block),
                            )
                        }
                        Err(e) => {
                            return Err(BlockPoolError::FailedToRegisterBlock(e.to_string()));
                        }
                    }
                };

            let mut immutable = self.active.register(mutable)?;

            match duplication_setting {
                BlockRegistrationDuplicationSetting::Allowed => {
                    if let Some(duplicate) = duplicate {
                        immutable = immutable
                            .with_duplicate(duplicate.into())
                            .expect("incompatible immutable block; only primary should be returned from ActiveBlockPool::register");
                    }
                }
                BlockRegistrationDuplicationSetting::Disabled => {
                    if let Some(block) = duplicate
                        && let Some(raw_blocks) = block.try_take_block(private::PrivateToken)
                    {
                        self.inactive.return_blocks(raw_blocks);
                    }
                }
            }

            if offload && let Some(priority) = immutable.metadata().offload_priority() {
                immutable.enqueue_offload(priority).await.unwrap();
            }

            immutable_blocks.push(immutable);
        }

        assert_eq!(immutable_blocks.len(), expected_len);

        let registered_ids: Vec<_> = immutable_blocks.iter().map(|b| b.block_id()).collect();
        tracing::error!(
            num_registered = immutable_blocks.len(),
            block_ids = ?registered_ids,
            active_after = self.active.status(),
            inactive_after = self.inactive.available_blocks(),
            "POOL_DEBUG: ✅ register_blocks complete - blocks now in active pool"
        );

        Ok(immutable_blocks)
    }

    async fn match_sequence_hashes(
        &mut self,
        sequence_hashes: Vec<SequenceHash>,
        return_rx: &mut tokio::sync::mpsc::UnboundedReceiver<Block<S, L, M>>,
    ) -> Vec<ImmutableBlock<S, L, M>> {
        tracing::error!(
            num_hashes = sequence_hashes.len(),
            active_blocks = self.active.status(),
            available_blocks = self.inactive.available_blocks(),
            "POOL_DEBUG: 🔍 match_sequence_hashes called"
        );

        let mut immutable_blocks = Vec::new();
        let mut matched_from_active = 0;
        let mut matched_from_inactive = 0;
        let mut waited_for_return = 0;

        for sequence_hash in &sequence_hashes {
            if !self.registry.is_registered(*sequence_hash) {
                tracing::error!(
                    sequence_hash = *sequence_hash,
                    "POOL_DEBUG: 🔍 hash not registered, stopping match"
                );
                break;
            }

            // the block is registered, so to get it from either the:
            // 1. active pool
            // 2. inactive pool
            // 3. return channel

            if let Some(immutable) = self.active.match_sequence_hash(*sequence_hash) {
                matched_from_active += 1;
                tracing::error!(
                    sequence_hash = *sequence_hash,
                    block_id = immutable.block_id(),
                    "POOL_DEBUG: 🎯 ACTIVE HIT - block already in active pool"
                );
                immutable_blocks.push(immutable);
                continue;
            }

            let raw_block =
                if let Some(raw_block) = self.inactive.match_sequence_hash(*sequence_hash) {
                    matched_from_inactive += 1;
                    tracing::error!(
                        sequence_hash = *sequence_hash,
                        block_id = raw_block.block_id(),
                        "POOL_DEBUG: 🎯 INACTIVE HIT - moving block from inactive to active"
                    );
                    raw_block
                } else {
                    waited_for_return += 1;
                    tracing::error!(
                        sequence_hash = *sequence_hash,
                        "POOL_DEBUG: ⏳ WAITING - block in flight, waiting for return..."
                    );
                    self.wait_for_returned_block(*sequence_hash, return_rx)
                        .await
                };

            // this assert allows us to skip the error checking on the active pool registration step
            assert!(matches!(raw_block.state(), BlockState::Registered(_, _)));

            let mutable = MutableBlock::new(raw_block, self.return_tx.clone());

            let immutable = self
                .active
                .register(mutable)
                .expect("unable to register block; should never happen");

            tracing::error!(
                sequence_hash = *sequence_hash,
                block_id = immutable.block_id(),
                "POOL_DEBUG: ➡️ INACTIVE→ACTIVE - block promoted to active pool"
            );

            immutable_blocks.push(immutable);
        }

        tracing::error!(
            total_matched = immutable_blocks.len(),
            from_active = matched_from_active,
            from_inactive = matched_from_inactive,
            waited_for = waited_for_return,
            "POOL_DEBUG: 🔍 match_sequence_hashes complete"
        );

        immutable_blocks
    }

    async fn touch_blocks(
        &mut self,
        sequence_hashes: &[SequenceHash],
        return_rx: &mut tokio::sync::mpsc::UnboundedReceiver<Block<S, L, M>>,
    ) {
        for sequence_hash in sequence_hashes {
            if !self.registry.is_registered(*sequence_hash) {
                break;
            }

            let block = if let Some(block) = self.inactive.match_sequence_hash(*sequence_hash) {
                block
            } else if self.active.match_sequence_hash(*sequence_hash).is_none() {
                self.wait_for_returned_block(*sequence_hash, return_rx)
                    .await
            } else {
                continue;
            };

            self.inactive.return_block(block);
        }
    }

    /// Returns a block to the inactive pool
    pub fn return_block(&mut self, mut block: Block<S, L, M>) {
        let block_id = block.block_id();
        let sequence_hash = block.sequence_hash().ok();
        let state_before = format!("{:?}", block.state());

        tracing::error!(
            block_id = block_id,
            sequence_hash = ?sequence_hash,
            state = %state_before,
            active_before = self.active.status(),
            inactive_before = self.inactive.available_blocks(),
            "POOL_DEBUG: ⬅️ RETURN BLOCK - returning block to pool"
        );

        self.active.remove(&mut block);

        let was_reset = block.state().is_reset();
        self.inactive.return_block(block);

        tracing::error!(
            block_id = block_id,
            was_reset = was_reset,
            active_after = self.active.status(),
            inactive_after = self.inactive.available_blocks(),
            "POOL_DEBUG: ⬅️ ACTIVE→INACTIVE - block returned to inactive pool"
        );
    }

    fn publisher(&self) -> Publisher {
        Publisher::new(self.event_manager.clone())
    }

    fn status(&self) -> BlockPoolStatus {
        let active = self.active.status();
        let (inactive, empty) = self.inactive.status();
        BlockPoolStatus {
            active_blocks: active,
            inactive_blocks: inactive,
            empty_blocks: empty,
        }
    }

    fn try_reset_blocks(&mut self, sequence_hashes: &[SequenceHash]) -> ResetBlocksResponse {
        let mut reset_blocks = Vec::new();
        let mut not_found = Vec::new();
        let mut not_reset = Vec::new();

        for sequence_hash in sequence_hashes {
            if !self.registry.is_registered(*sequence_hash) {
                not_found.push(*sequence_hash);
                continue;
            }

            if let Some(mut block) = self.inactive.match_sequence_hash(*sequence_hash) {
                reset_blocks.push(*sequence_hash);
                block.reset();
                self.inactive.return_block(block);
            } else {
                not_reset.push(*sequence_hash);
            }
        }

        ResetBlocksResponse {
            reset_blocks,
            not_found,
            not_reset,
        }
    }
}
