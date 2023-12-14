# Steerable Motion, a ComfyUI custom node for steering videos with batches of images

Steerable Motion is a ComfyUI node for batch creative interpolation. Our goal is to feature the best methods for steering motion with images as video models evolve.	

![Main example](https://github.com/peteromallet/ComfyUI-Creative-Interpolation/blob/main/demo/main_example.gif)

## Installation

1. If you haven't already, install [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and [Comfy Manager](https://github.com/ltdrdata/ComfyUI-Manager) - you can find instructions on their pages.
2. Search "Steerable Motion" in Comfy Manager and download the node.
3. Download [this workflow](https://raw.githubusercontent.com/banodoco/steerable-motion/main/demo/creative_interpolation_example.json) and drop it into ComfyUI.
4. When the workflow opens, download the dependent nodes by pressing "Install Missing Custom Nodes" in Comfy Manager. Search and download the required models from Comfy Manager also - make sure that the models you download have the same name as the ones in the workflow - or you're confident that they're the same.

## Usage

The main settings are:

- Key frame position: how many frames to generate between each main key frame you provide.
- Length of influence: what range of frames to apply ControlNet (CN) and IP-Adapter (IPA) to.
- Strength of influence: how strong the control of IPA and CN should be.
- Relative IPA strength & influence: whether to make IPA's influence stronger or weaker than CN's.

These are set linearly - the same for each frame - or  dynamically - varying them for each frame - you can find detailed instructions on how to tweak these settings inside the workflow above.

Tweaking the settings can greatly influence the motion - you can see two examples of the same images with slightly different settings below to get an idea of this - the settings are visualised in the graph:

![Tweaking settings example](https://github.com/peteromallet/ComfyUI-Creative-Interpolation/blob/main/demo/tweaking_settings.gif)

This also works well for moving between dramatically different images - like in the example below:

![Different images example](https://github.com/peteromallet/ComfyUI-Creative-Interpolation/blob/main/demo/different_images.gif)

## Philosophy for getting the most from this

This isn’t a tool like text to video that will perform well out of the box - it’s more like a paint brush, an artistic tool that you need to figure out how to get the best from. 

Through trial and error, you'll need to build an understanding of how the motion and settings work, what its limitations are, which inputs images work best with it, etc.

If you can figure out how to wield it, this approach can provide enough control for you to make beautiful things that match your imagination precisely.

## Coming Soon

- Implement more powerful video models and approaches to increase lengh, coherence, etc.
- Implement video ControlNets (upcoming from both Stability and Animatediff authors) for greater adherence to input images and better motion.
- Increase frame count to unlimited - currently limited to 12 due to RAM constraints with current approach.
- Fix issues with colour drain, etc.
- Implement LCM, Turbo, etc. to increase speed.

## Want to give feedback, or join a community who are pushing open source models to their artistic and technical limits?

You're very welcome to drop into our Discord [here](https://discord.com/invite/8Wx9dFu5tP).

## Credits

This code draws heavily from [Kosinkadink's ComfyUI-Advanced-ControlNet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet) and Cubiq's [IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus), while the workflows uses [Kosinkadink's Animatediff Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved), Fizzledorf's [Fizznodes](https://github.com/FizzleDorf/ComfyUI_FizzNodes), Fannovel16's [Frame Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation) and more. Thanks to all and of course the Animatediff team, Controlnet, others, and of course our supportive community!
