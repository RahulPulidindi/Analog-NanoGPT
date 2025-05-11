from aihwkit.simulator.configs import SoftBoundsReferenceDevice, SingleRPUConfig, InferenceRPUConfig

# rpu_config = SingleRPUConfig(device=SoftBoundsReferenceDevice())
rpu_config = InferenceRPUConfig()
print(rpu_config)
