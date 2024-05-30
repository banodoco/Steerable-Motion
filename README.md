# Steerable Motion, a ComfyUI custom node for steering videos with batches of images

Steerable Motion is a ComfyUI node for batch creative interpolation. Our goal is to feature the best quality and most precise and powerful methods for steering motion with images as video models evolve.	This node is best used via [Dough](https://github.com/banodoco/dough) - a creative tool which simplifies the settings and provides a nice creative flow - or in Discord - by joining this channel.

![Main example](https://github.com/banodoco/steerable-motion/blob/main/demo/main_example.gif)

## Installation in Comfy

1. If you haven't already, install [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and [Comfy Manager](https://github.com/ltdrdata/ComfyUI-Manager) - you can find instructions on their pages.
2. Download [this workflow](https://raw.githubusercontent.com/banodoco/steerable-motion/main/demo/steerable-motion_smooth-n-steady.json) and drop it into ComfyUI - or you can use one of the workflows others in the community made below.
3. When the workflow opens, download the dependent nodes by pressing "Install Missing Custom Nodes" in Comfy Manager. Search and download the required models from Comfy Manager also - make sure that the models you download have the same name as the ones in the workflow - or you're confident that they're the same.

## Usage

The main settings are:

- Key frame position: how many frames to generate between each main key frame you provide.
- Length of influence: what range of frames to apply the IP-Adapter (IPA) influence to.
- Strength of influence: what the low-point and high-point of each frame should be.
- Image adherence: how much we should force adherence to the input images.

Other than image adherence which is set for the entire generation these are set linearly - the same for each frame - or dynamically - varying them for each frame - you can find detailed instructions on how to tweak these settings inside the workflow above.

Tweaking the settings can greatly influence the motion - for example, below you can see two examples of the same images animated - but with the one setting tweaked, the length of each frame's influence:

![Tweaking settings example](https://github.com/banodoco/steerable-motion/blob/main/demo/tweaking_settings.gif)

## Philosophy for getting the most from this

This isn’t a tool like text to video that will perform well out of the box, it’s more like a paint brush - an artistic tool that you need to figure out how to get the best from. 

Through trial and error, you'll need to build an understanding of how the motion and settings work, what its limitations are, which inputs images work best with it, etc.

It won't work for everything but if you can figure out how to wield it, this approach can provide enough control for you to make beautiful things that match your imagination precisely.

## 5 basic workflows to get started

Below are 5 basic workflows - each with their own weird and unique characteristics - all with differing levels of adherence and different types of motion - most of the changes come from tweaking the IPA configuration and switching out base models:

- [Smooth n' Steady](https://raw.githubusercontent.com/banodoco/steerable-motion/main/demo/steerable-motion_smooth-n-steady.json): tends to have nice smooth motion - good starting point
- [Rad Attack](https://raw.githubusercontent.com/banodoco/steerable-motion/main/demo/steerable-motion_rad-attack.json): probably the best for realistic motion
- [Slurshy Realistiche](https://raw.githubusercontent.com/banodoco/steerable-motion/main/demo/steerable-motion_slurshy-realistiche.json): moves in a slightly realistic manner but is a little bit slurshy
- [Chocky Realistiche](https://raw.githubusercontent.com/banodoco/steerable-motion/main/demo/steerable-motion_chocky-realistiche.json): realistic-ish but very blocky
- [Liquidy Loop](https://raw.githubusercontent.com/banodoco/steerable-motion/main/demo/steerable-motion_liquidy-loop.json): smooth and liquidy

You can see each in acton below:

![basic workflows](https://github.com/banodoco/steerable-motion/blob/main/demo/basic_workflows.gif)

## 2 examples of things others have built on top of this:

The workflows I share above are just basic examples of it in action - below are two other workflows people in our community have created on top of this node that leverage the same underlying mechanism in creative and interesting ways:

### Looped LCM by @idgallagher

First, [@idgallagher](https://twitter.com/idgallagher) uses LCM and different settings to achieve a really interesting realistic motion effect. You can grab it [here](https://github.com/IDGallagher/storage/blob/main/chiff_distilled_sm.json) and see an example output here:

![Flipping Sigmas](https://github.com/banodoco/steerable-motion/blob/main/demo/flipping_sigmas.gif)

### Smooth & Deep by @Superbeasts.ai:

Next, [Superbeasts.ai](https://www.instagram.com/superbeasts) uses depth maps to control the motion in different layers - creating a smoother motion effect. You can grab this workflow [here](https://github.com/banodoco/Steerable-Motion/blob/main/demo/SuperBeasts-POM-SmoothBatchCreative-V1.3.1.json) and see an example of it in action here:

![Superbeasts Example](https://github.com/banodoco/steerable-motion/blob/main/demo/superbeasts.gif)

I believe that that there are endless ways to expand upon and extend the ideas in this node - if you do anything cool, please share!

## Want to give feedback, or join a community who are pushing open source models to their artistic and technical limits?

You're very welcome to drop into our Discord [here](https://discord.com/invite/8Wx9dFu5tP).

## Credits

This code draws heavily from Cubiq's [IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus), while the workflow uses Kosinkadink's [Animatediff Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved) and [ComfyUI-Advanced-ControlNet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet), Fizzledorf's [Fizznodes](https://github.com/FizzleDorf/ComfyUI_FizzNodes), Fannovel16's [Frame Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation) and more. Thanks to all and of course the Animatediff team, Controlnet, others, and of course our supportive community!

