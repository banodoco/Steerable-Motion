# ComfyUI-Advanced-ControlNet
These custom nodes allow for scheduling ControlNet strength across latents in the same batch (WORKING) and across timesteps (IN PROGRESS).

Custom weights can also be applied to ControlNets and T2IAdapters to mimic the "My prompt is more important" functionality in AUTOMATIC1111's ControlNet extension.

TODO:
- Other handy nodes
- Finish and update this README for other workflows

## Workflows

### AnimateDiff Workflows
***Latent Keyframes*** identify which latents in a batch the ControlNet should apply to, and at what strength. They connect to a ***Timestep Keyframe*** to identify at what point in the generation to kick in (for basic use, start_percent on the Timestep Keyframe should be 0.0). Latent Keyframe nodes can be chained to apply the ControlNet to multiple keyframes at various strengths.

