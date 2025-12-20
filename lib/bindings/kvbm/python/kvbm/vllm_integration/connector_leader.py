# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Implementation of vLLM KV cache manager protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request


# from kvbm.vllm_integration.kv_cache_utils import KvbmCacheBlocks
# from kvbm.vllm_integration.rust import BlockManager, KvbmRequest
# from kvbm.vllm_integration.rust import KvConnectorLeader as RustKvConnectorLeader
# from kvbm.vllm_integration.rust import (
#     KvConnectorMetadata as RustKvConnectorMetadata,
# )
# from kvbm.vllm_integration.rust import SchedulerOutput as RustSchedulerOutput

from kvbm import KvbmLeader
from kvbm.utils import is_dyn_runtime_enabled
from kvbm.vllm_integration.rust import KvbmRequest
from kvbm.vllm_integration.rust import KvConnectorLeader as RustKvConnectorLeader
from kvbm.vllm_integration.rust import SchedulerOutput as RustSchedulerOutput

DistributedRuntime = None
if is_dyn_runtime_enabled():
    from dynamo.runtime import DistributedRuntime


class DynamoConnectorMetadata(KVConnectorMetadata):
    def __init__(self, metadata: bytes):
        assert isinstance(metadata, bytes)
        self.metadata = metadata


class KvConnectorLeader:
    """
    Implements the vLLM KV cache manager protocol.

    This class is a wrapper around the Rust KvbmCacheManager class.
    It is used to convert the Rust KvbmCacheManager into a Python class
    that can be used in the vLLM KV cache manager protocol.
    """

    def __init__(self, vllm_config: "VllmConfig", engine_id: str, **kwargs):
        drt: Optional[object] = kwargs.get("drt")

        if drt is None and is_dyn_runtime_enabled():
            drt = DistributedRuntime.detached()
        else:
            drt = None

        self.drt = drt
        self.vllm_config = vllm_config
        world_size = vllm_config.parallel_config.world_size

        leader = KvbmLeader(world_size, drt=self.drt)

        print(f"KvConnectorLeader initialized with engine_id: {engine_id}")
        # Get kv event consolidator endpoints from vllm_config (pre-computed in main.py)
        consolidator_vllm_endpoint = None
        consolidator_output_endpoint = None
        self._consolidator_output_port = None

        if (
            hasattr(vllm_config, "consolidator_endpoints")
            and vllm_config.consolidator_endpoints
        ):
            # Unpack all three endpoints
            # [0]: vllm_endpoint (for consolidator to subscribe to vLLM)
            # [1]: output_bind_endpoint (for consolidator to bind/publish)
            # [2]: output_connect_endpoint (for clients to connect)
            (
                consolidator_vllm_endpoint,
                consolidator_output_endpoint,
                _consolidator_output_connect_endpoint,  # Not needed here
            ) = vllm_config.consolidator_endpoints
            self._consolidator_output_port = int(
                consolidator_output_endpoint.split(":")[-1]
            )

            # Pass endpoints to Rust
            self._connector = RustKvConnectorLeader(
                engine_id,
                self.drt,
                vllm_config.cache_config.block_size,
                leader,
                consolidator_vllm_endpoint=consolidator_vllm_endpoint,
                consolidator_output_endpoint=consolidator_output_endpoint,
            )
        else:
            # No kv event consolidator - pass None to Rust
            self._connector = RustKvConnectorLeader(
                engine_id,
                self.drt,
                vllm_config.cache_config.block_size,
                leader,
                consolidator_vllm_endpoint=None,
                consolidator_output_endpoint=None,
            )

    # KV Connector

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request (GPU cache hits from vLLM)

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded from the
                  external KV cache beyond what is already computed.
                - `True` if external KV cache tokens will be loaded
                  asynchronously (between scheduler steps).
        """
        self._create_slot(request)

        # Track device (GPU) cache hits - num_computed_tokens represents tokens already in GPU cache
        # vLLM has already matched these against its GPU prefix cache before calling us
        if num_computed_tokens > 0:
            try:
                self._connector.record_device_cache_hits(request.request_id, num_computed_tokens)
            except Exception:
                pass  # Silently ignore - cache stats are optional

        return self._connector.get_num_new_matched_tokens(
            request.request_id,
            request.num_tokens,
            num_computed_tokens,
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        block_ids = blocks.get_block_ids()[0]
        self._connector.update_state_after_alloc(
            request.request_id, block_ids, num_external_tokens
        )

    def record_device_cache_hits(self, request_id: str, num_tokens: int) -> None:
        """Record device (GPU) cache hits for a request."""
        self._connector.record_device_cache_hits(request_id, num_tokens)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> bytes:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        output = RustSchedulerOutput()

        for req in scheduler_output.scheduled_new_reqs:
            output.add_new_request(
                req.req_id,
                req.prompt_token_ids,
                req.block_ids[0],
                req.num_computed_tokens,
            )

        for (
            req_id,
            resumed_from_preemption,
            new_token_ids,
            new_block_ids,
            num_computed_tokens,
        ) in zip(
            scheduler_output.scheduled_cached_reqs.req_ids,
            scheduler_output.scheduled_cached_reqs.resumed_from_preemption,
            scheduler_output.scheduled_cached_reqs.new_token_ids,
            scheduler_output.scheduled_cached_reqs.new_block_ids,
            scheduler_output.scheduled_cached_reqs.num_computed_tokens,
        ):
            if new_block_ids is not None:
                output.add_cached_request(
                    request_id=req_id,
                    resumed_from_preemption=resumed_from_preemption,
                    new_token_ids=new_token_ids,
                    new_block_ids=new_block_ids[0],
                    num_computed_tokens=num_computed_tokens,
                )
            else:
                output.add_cached_request(
                    request_id=req_id,
                    resumed_from_preemption=resumed_from_preemption,
                    new_token_ids=new_token_ids,
                    new_block_ids=[],
                    num_computed_tokens=num_computed_tokens,
                )

        output.add_num_scheduled_tokens(scheduler_output.num_scheduled_tokens)

        assert (
            scheduler_output.total_num_scheduled_tokens
            == output.get_num_scheduled_tokens()
        ), "Total number of scheduled tokens does not match"

        return self._connector.build_connector_metadata(output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        # Get cache stats BEFORE calling request_finished (which may modify/remove slot state)
        kv_transfer_params = None

        try:
            cache_stats = self._connector.get_cache_stats(request.request_id)

            if cache_stats is not None:
                device_blocks, host_blocks, disk_blocks = cache_stats

                kv_transfer_params = {
                    "cache_hit_breakdown": {
                        "device_blocks": int(device_blocks),
                        "host_blocks": int(host_blocks),
                        "disk_blocks": int(disk_blocks),
                    }
                }
        except Exception:
            # Don't fail the request if cache stats retrieval fails
            pass

        # note our worker can communicate with us oob and we can use that to know
        # ahead of time if the request is finished.
        status = self._connector.request_finished(request.request_id, block_ids)

        return status, kv_transfer_params

    # Utility functions

    def _create_slot(self, request: Request) -> None:
        """Create a slot for the request"""

        if self._connector.has_slot(request.request_id):
            return None

        if bool(getattr(request, "mm_features", None)) or bool(
            getattr(request, "mm_positions", None)
        ):
            raise ValueError("Unsupported request - requires mm extra keys")

        all_token_ids = request.all_token_ids

        # extract the critial aspects of the request that effect how the tokens are hashed
        request = KvbmRequest(
            request_id=request.request_id,
            lora_name=request.lora_request.lora_name()
            if request.lora_request
            else None,
            salt_hash=request.cache_salt,
        )

        self._connector.create_slot(request, all_token_ids)
