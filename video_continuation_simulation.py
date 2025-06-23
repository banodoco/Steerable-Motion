import matplotlib.pyplot as plt
import numpy as np

def simulate_video_continuation_comprehensive(input_frames, control_frames, total_output_frames, overlap_frames, 
                                            when_to_start_control_frames, when_to_start_masks, end_frame=None):
    """
    Comprehensive simulation of VideoContinuationGenerator logic
    """
    print(f"\n=== Simulation: control='{when_to_start_control_frames}', masks='{when_to_start_masks}' ===")
    
    # Step 1: Calculate actual overlap frames
    actual_overlap_frames = min(overlap_frames, len(input_frames), total_output_frames)
    
    # Step 2: Prepare start frames (from overlap)
    overlap_start_idx = len(input_frames) - actual_overlap_frames
    start_frames = input_frames[overlap_start_idx:overlap_start_idx + actual_overlap_frames]
    
    # Step 3: Prepare end frame
    num_end_frames = 1 if end_frame is not None and total_output_frames > actual_overlap_frames else 0
    end_frames = [end_frame] if num_end_frames > 0 else []
    
    # Step 4: Calculate middle frames needed
    num_middle_frames = total_output_frames - actual_overlap_frames - num_end_frames
    
    # Step 5: Fill middle frames based on control frame mode
    middle_frames = []
    control_frame_info = ""
    
    if num_middle_frames > 0 and control_frames:
        if when_to_start_control_frames == "beginning":
            # Use control frames from the beginning of the sequence (C0, C1, C2...)
            if len(control_frames) < num_middle_frames:
                middle_frames = control_frames + ['EMPTY'] * (num_middle_frames - len(control_frames))
                control_frame_info = f"Using first {len(control_frames)} control frames + {num_middle_frames - len(control_frames)} empty"
            else:
                middle_frames = control_frames[:num_middle_frames]
                control_frame_info = f"Using first {num_middle_frames} control frames (C0-C{num_middle_frames-1})"
        else:  # "after overlap_frames"
            # Skip the first overlap_frames control images to avoid duplication
            duplicate_count = min(actual_overlap_frames, len(control_frames))
            available_after_dup = len(control_frames) - duplicate_count
            
            if available_after_dup < num_middle_frames:
                selected_control = control_frames[duplicate_count:]
                padding_needed = num_middle_frames - len(selected_control)
                middle_frames = selected_control + ['EMPTY'] * padding_needed
                control_frame_info = f"Skipped first {duplicate_count} control frames, using C{duplicate_count}-C{duplicate_count + len(selected_control) - 1} + {padding_needed} empty"
            else:
                middle_frames = control_frames[duplicate_count:duplicate_count + num_middle_frames]
                control_frame_info = f"Skipped first {duplicate_count} control frames, using C{duplicate_count}-C{duplicate_count + num_middle_frames - 1}"
    else:
        middle_frames = ['EMPTY'] * num_middle_frames
        control_frame_info = "No control frames, all empty"
    
    # Step 6: Create masks based on mask mode
    masks = [1.0] * total_output_frames  # 1.0 = inpaint, 0.0 = known
    
    if when_to_start_masks == "beginning":
        # Set known frames (overlap and end) to 0.0, rest stay as 1.0 (inpaint)
        for i in range(actual_overlap_frames):
            masks[i] = 0.0  # overlap frames are known
        for i in range(total_output_frames - num_end_frames, total_output_frames):
            masks[i] = 0.0  # end frames are known
        mask_info = "Standard: overlap and end frames known, middle frames inpaint"
    else:  # "after overlap_frames"
        # Follow control frame logic
        for i in range(actual_overlap_frames):
            masks[i] = 0.0  # overlap frames are known
        for i in range(total_output_frames - num_end_frames, total_output_frames):
            masks[i] = 0.0  # end frames are known
            
        # For middle section, follow the same logic as control frames
        if control_frames and num_middle_frames > 0:
            duplicate_count = min(actual_overlap_frames, len(control_frames))
            available_after_dup = len(control_frames) - duplicate_count
            if available_after_dup >= num_middle_frames:
                # If we have enough control frames after skipping, set those middle frames as known (0.0)
                middle_start = actual_overlap_frames
                middle_end = middle_start + num_middle_frames
                for i in range(middle_start, middle_end):
                    masks[i] = 0.0
                mask_info = "Follows control logic: overlap, control-covered middle, and end frames known"
            else:
                mask_info = "Follows control logic: overlap and end frames known, partial middle coverage"
        else:
            mask_info = "No control frames: only overlap and end frames known"
    
    # Step 7: Assemble final video
    final_video = start_frames + middle_frames + end_frames
    
    print(f"Control: {control_frame_info}")
    print(f"Masks: {mask_info}")
    print(f"Final video: {final_video}")
    print(f"Masks: {['INPAINT' if m > 0.5 else 'KNOWN' for m in masks]}")
    
    return {
        'start_frames': start_frames,
        'middle_frames': middle_frames,
        'end_frames': end_frames,
        'final_video': final_video,
        'masks': masks,
        'actual_overlap_frames': actual_overlap_frames,
        'num_middle_frames': num_middle_frames,
        'num_end_frames': num_end_frames,
        'control_frame_info': control_frame_info,
        'mask_info': mask_info
    }

def visualize_comprehensive_simulation(results_list, scenario_names):
    """
    Create a comprehensive visual representation showing both frames and masks
    """
    fig, axes = plt.subplots(len(results_list), 2, figsize=(20, 4 * len(results_list)))
    if len(results_list) == 1:
        axes = axes.reshape(1, -1)
    
    colors = {
        'INPUT': '#87CEEB',      # Sky blue
        'CONTROL': '#98FB98',    # Pale green  
        'END': '#FFA500',        # Orange
        'EMPTY': '#D3D3D3'       # Light gray
    }
    
    mask_colors = {
        'KNOWN': '#4169E1',      # Royal blue
        'INPAINT': '#FF6347'     # Tomato red
    }
    
    for i, (results, scenario_name) in enumerate(zip(results_list, scenario_names)):
        # Plot frames (left column)
        ax_frames = axes[i, 0]
        final_video = results['final_video']
        
        frame_colors = []
        frame_labels = []
        
        for frame in final_video:
            if isinstance(frame, str):
                if frame == 'EMPTY':
                    frame_colors.append(colors['EMPTY'])
                    frame_labels.append('EMPTY')
                elif frame == 'END':
                    frame_colors.append(colors['END'])
                    frame_labels.append('END')
                else:
                    frame_colors.append(colors['CONTROL'])
                    frame_labels.append(frame)
            else:
                # Input frame (number)
                frame_colors.append(colors['INPUT'])
                frame_labels.append(f'I{frame}')
        
        x_positions = range(len(final_video))
        bars_frames = ax_frames.bar(x_positions, [1] * len(final_video), 
                                   color=frame_colors, edgecolor='black', linewidth=0.5)
        
        # Add frame labels
        for j, (bar, label) in enumerate(zip(bars_frames, frame_labels)):
            ax_frames.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                          label, ha='center', va='center', fontsize=8, rotation=90)
        
        ax_frames.set_title(f'{scenario_name}\nFrames: {results["control_frame_info"]}')
        ax_frames.set_ylabel('Frame Content')
        ax_frames.set_xlabel('Frame Position')
        ax_frames.set_ylim(0, 1.2)
        ax_frames.set_xticks(x_positions)
        ax_frames.set_xticklabels([str(i) for i in x_positions])
        
        # Plot masks (right column)
        ax_masks = axes[i, 1]
        masks = results['masks']
        
        mask_color_list = [mask_colors['KNOWN'] if m < 0.5 else mask_colors['INPAINT'] for m in masks]
        mask_label_list = ['KNOWN' if m < 0.5 else 'INPAINT' for m in masks]
        
        bars_masks = ax_masks.bar(x_positions, [1] * len(masks), 
                                 color=mask_color_list, edgecolor='black', linewidth=0.5)
        
        # Add mask labels
        for j, (bar, label) in enumerate(zip(bars_masks, mask_label_list)):
            ax_masks.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                         label, ha='center', va='center', fontsize=8, rotation=90)
        
        ax_masks.set_title(f'Masks: {results["mask_info"]}')
        ax_masks.set_ylabel('Mask Type')
        ax_masks.set_xlabel('Frame Position')
        ax_masks.set_ylim(0, 1.2)
        ax_masks.set_xticks(x_positions)
        ax_masks.set_xticklabels([str(i) for i in x_positions])
        
        # Add legends
        if i == 0:  # Only add legend to first row
            # Frame legend
            frame_legend_elements = []
            for frame_type, color in colors.items():
                frame_legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=frame_type))
            ax_frames.legend(handles=frame_legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
            
            # Mask legend
            mask_legend_elements = []
            for mask_type, color in mask_colors.items():
                mask_legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=mask_type))
            ax_masks.legend(handles=mask_legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    plt.savefig('video_continuation_comprehensive_simulation.png', dpi=150, bbox_inches='tight')
    plt.show()

def run_comprehensive_simulation():
    # Test setup
    input_frames = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10 input frames
    control_frames = [f'C{i}' for i in range(20)]   # 20 control frames (enough for all scenarios)
    total_output_frames = 17  # (17-1) % 4 == 0
    overlap_frames = 3
    end_frame = 'END'
    
    print("=" * 80)
    print("VIDEO CONTINUATION GENERATOR COMPREHENSIVE SIMULATION")
    print("=" * 80)
    print(f"Input frames: {input_frames}")
    print(f"Control frames: {control_frames}")
    print(f"Total output frames: {total_output_frames}")
    print(f"Overlap frames: {overlap_frames}")
    print(f"End frame: {end_frame}")
    print(f"Middle frames needed: {total_output_frames - overlap_frames - 1} = {total_output_frames - overlap_frames - 1}")
    
    # Test all combinations
    scenarios = [
        ("beginning", "beginning", "Control: Beginning, Masks: Beginning"),
        ("beginning", "after overlap_frames", "Control: Beginning, Masks: After Overlap"),
        ("after overlap_frames", "beginning", "Control: After Overlap, Masks: Beginning"),
        ("after overlap_frames", "after overlap_frames", "Control: After Overlap, Masks: After Overlap"),
    ]
    
    results_list = []
    scenario_names = []
    
    for control_mode, mask_mode, display_name in scenarios:
        result = simulate_video_continuation_comprehensive(
            input_frames, control_frames, total_output_frames, 
            overlap_frames, control_mode, mask_mode, end_frame
        )
        results_list.append(result)
        scenario_names.append(display_name)
    
    # Additional test: Insufficient control frames scenario
    print("\n" + "="*50)
    print("INSUFFICIENT CONTROL FRAMES TEST")
    print("="*50)
    
    short_control_frames = ['C0', 'C1', 'C2', 'C3', 'C4']  # Only 5 control frames
    print(f"Short control frames: {short_control_frames}")
    
    for control_mode, mask_mode, display_name in scenarios:
        result = simulate_video_continuation_comprehensive(
            input_frames, short_control_frames, total_output_frames, 
            overlap_frames, control_mode, mask_mode, end_frame
        )
        results_list.append(result)
        scenario_names.append(f"{display_name} (Short Control)")
    
    visualize_comprehensive_simulation(results_list, scenario_names)

if __name__ == "__main__":
    run_comprehensive_simulation() 