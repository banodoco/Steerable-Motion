# Steerable Motion, ComfyUI custom nodes & workflows node for steering videos with batches of images

Steerable Motion is a set of ComfyUI nodes and workflows for travelling between images.

## Installation in Comfy

1. If you haven't already, install [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and [Comfy Manager](https://github.com/ltdrdata/ComfyUI-Manager) - you can find instructions on their pages.
2. When the workflow opens, download the dependent nodes by pressing "Install Missing Custom Nodes" in Comfy Manager. Search and download the required models from Comfy Manager also.

## Wan

The Wan approach uses VACE to create anchor images and continuations from previous images, which are chained together at the end:

![Main example](demo/wan_example.gif)


### Sample workflow for Wan

You can find a workflow [here](demo/Vace_Travel.json) to get started.

## Animatediff

The Animatediff approach uses a combination of IP-Adapter and SparseCtrl to travel between images:

![Main example](demo/main_example.gif)


### 5 basic workflows for Animatediff

Below are 5 basic workflows - each with their own weird and unique characteristics - all with differing levels of adherence and different types of motion - most of the changes come from tweaking the IPA configuration and switching out base models:

- [Smooth n' Steady](https://raw.githubusercontent.com/banodoco/steerable-motion/main/demo/steerable-motion_smooth-n-steady.json): tends to have nice smooth motion - good starting point
- [Rad Attack](https://raw.githubusercontent.com/banodoco/steerable-motion/main/demo/steerable-motion_rad-attack.json): probably the best for realistic motion
- [Slurshy Realistiche](https://raw.githubusercontent.com/banodoco/steerable-motion/main/demo/steerable-motion_slurshy-realistiche.json): moves in a slightly realistic manner but is a little bit slurshy
- [Chocky Realistiche](https://raw.githubusercontent.com/banodoco/steerable-motion/main/demo/steerable-motion_chocky-realistiche.json): realistic-ish but very blocky
- [Liquidy Loop](https://raw.githubusercontent.com/banodoco/steerable-motion/main/demo/steerable-motion_liquidy-loop.json): smooth and liquidy

You can see each in acton below:

![basic workflows](demo/basic_workflows.gif)

## Philosophy for getting the most from these

This isn't an approach like text to video that will perform well out of the box, it's more like a paint brush - an artistic tool that you need to figure out how to get the best from. 

Through trial and error, you'll need to build an understanding of how the motion and settings work, what its limitations are, which inputs images work best with it, etc.

It won't work for everything but if you can figure out how to wield it, this approach can provide enough control for you to make beautiful things that match your imagination precisely.

In both cases, tweaking the settings can greatly influence the motion - for example, below you can see two examples of the same images animated - but with the one setting tweaked, the length of each frame's influence:

![Tweaking settings example](demo/tweaking_settings.gif)

## Want to give feedback, or join a community who are pushing open source models to their artistic and technical limits?

You're very welcome to drop into our Discord [here](https://discord.com/invite/8Wx9dFu5tP).

## Credits

For Animatediff, the code draws heavily from Cubiq's [IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus), while the workflow uses Kosinkadink's [Animatediff Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved) and [ComfyUI-Advanced-ControlNet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet), Fizzledorf's [Fizznodes](https://github.com/FizzleDorf/ComfyUI_FizzNodes), Fannovel16's [Frame Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation) and more. 

For Wan, it's built on top of the work of Kijai's wonderful [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) and of course the VACE and Wan teams. 
