from cutlass import cute, pipeline


@cute.jit
def make_tma_pipeline(
    mbar_ptr: cute.Pointer, num_stages: int, num_bytes: int, cta_layout_vmnk: cute.Layout, mcast_size: int, num_consumer_warps: int
):
    num_producers = 1
    num_consumers = num_consumer_warps * mcast_size  # IMPORTANT!!!

    producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_producers)
    consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_consumers)
    # NOTE: CTA layout is only used for syncing
    return pipeline.PipelineTmaAsync.create(
        barrier_storage=mbar_ptr,
        num_stages=num_stages,
        tx_count=num_bytes,
        producer_group=producer_group,
        consumer_group=consumer_group,
        cta_layout_vmnk=cta_layout_vmnk,
    )
