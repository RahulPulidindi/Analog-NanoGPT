from aihwkit.simulator.configs import SoftBoundsReferenceDevice, SingleRPUConfig

rpu_config = SingleRPUConfig(device=SoftBoundsReferenceDevice())
print(rpu_config)
