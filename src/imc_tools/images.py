import logging
import os.path
from pathlib import Path

import numpy as np
import tqdm
import cellpose.models

from PIL import Image
from matplotlib import pyplot as plt
from readimc import MCDFile
from imcsegpipe.utils import filter_hot_pixels

logger = logging.getLogger(__name__)


def remove_outliers(arr, percentile=99.5):
    arr = arr.copy()
    max_value = np.percentile(arr, percentile)
    arr[arr > max_value] = max_value
    return arr


def extract_channels(mcd_fn,
                             selected_channels,
                             slide_idx=0,
                                acquisition_idx=0):
    acquisition_arr = None

    with MCDFile(mcd_fn) as f:
        for slide_idx, slide in tqdm.tqdm(
            enumerate(f.slides),
            position=0,
            desc="Slides"
        ):
            for acquisition_idx, acquisition in tqdm.tqdm(
                enumerate(slide.acquisitions),
                position=1,
                desc="Acquisitions"
            ):
                if slide_idx == slide_idx and acquisition_idx == acquisition_idx:
                    channel_names = [(a if a else b) for (a, b) in
                                     zip(acquisition.channel_labels, acquisition.channel_names)]

                    acquisition_arr = f.read_acquisition(acquisition)
                    break

    if acquisition_arr is None:
        raise ValueError(f"Could not find slide {slide_idx} acquisition {acquisition_idx}")

    for channel in selected_channels:
        if channel not in channel_names:
            raise ValueError(f"Channel {channel} not found in {channel_names}")

    image_arrays = []
    for channel_idx, channel in enumerate(channel_names):
        if channel in selected_channels:
            channel_arr = Image.fromarray(acquisition_arr[channel_idx, ...]).convert('L')
            array = np.array(channel_arr)
            image_arrays.append(array)

    # Verify that all images have the same shape
    shape_check = image_arrays[0].shape
    for array in image_arrays[1:]:
        if array.shape != shape_check:
            raise ValueError("All images must have the same dimensions.")

    if len(image_arrays) < 3:
        # add zero layers
        for i in range(3 - len(image_arrays)):
            image_arrays.append(np.zeros(shape_check))

    # Step 3: Stack the N arrays to make an N-channel image
    combined_array = np.stack(image_arrays, axis=2)

    combined_array = filter_hot_pixels(combined_array, thres=50)

    return  np.uint8(combined_array)

def extract_channels_to_tiff(mcd_fn,
                         output_file,
                         selected_channels,
                         slide_idx=0,
                         acquisition_idx=0):

    combined_array = extract_channels(mcd_fn,
                                selected_channels,
                                slide_idx=slide_idx,
                                acquisition_idx=acquisition_idx)

    # Step 4: Convert the N-channel array back to a Pillow image
    # Note: Standard image formats may not support N-channel images.
    # Here, we're relying on TIFF's flexibility.
    combined_image = Image.fromarray(np.uint8(combined_array))

    # Step 5: Save the new image as a TIFF file
    combined_image.save(output_file)


def plot_channels(mcd_fn, output_dir):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    with MCDFile(mcd_fn) as f:
        for slide_idx, slide in tqdm.tqdm(
            enumerate(f.slides),
            position=0,
            desc="Slides"
        ):
            for acquisition_idx, acquisition in tqdm.tqdm(
                enumerate(slide.acquisitions),
                position=1,
                desc="Acquisitions"
            ):
                channel_names = [(a if a else b) for (a, b) in
                                 zip(acquisition.channel_labels, acquisition.channel_names)]

                img = f.read_acquisition(acquisition)

                for idx, channel_name in tqdm.tqdm(
                    enumerate(channel_names),
                    position=2,
                    desc="Channels"
                ):
                    arr = img[idx, ...].copy()

                    max_val = np.percentile(img[idx, ...], 99)

                    arr[arr >= max_val] = max_val

                    plt.figure(figsize=(10, 10))
                    plt.imshow(arr, aspect="equal")
                    plt.colorbar()
                    plt.savefig(os.path.join(output_dir,
                                             f"slide{slide_idx}_acquisition{acquisition_idx}_{channel_name}.png"))
                    plt.close()


def plot_channel_onto_panorama(mcd_file,
                               output_dir,
                               panorama_idx=0,
                               overlay_color=(57, 255, 20),
                               rotation_angle=0,
                               flip_axis=None,
                               channel_label="DNA1"):
    """
    Plot a channel onto the panorama image

    :param mcd_file: Input MCD file for IMC experiment
    :param output_dir: Output dir where multiple images will be saved
    :param panorama_idx: If the MCD file happens to contain multiple panoramas,
                         this is the index of the panorama to use
    :param overlay_color: Color to use for transparent overlay of the channel
    :param rotation_angle: Rotate the background image by this much before overlay
    :param flip_axis: Flip background on this axis before overlay
    :param channel_label: Name of channel label to plot
    """
    with MCDFile(mcd_file) as f:
        for slide_idx, slide in tqdm.tqdm(
            enumerate(f.slides),
            position=0,
            desc="Slides"
        ):
            instrument_panorama = [x for x in slide.panoramas if x.metadata["Type"] == "Instrument"][panorama_idx]

            max_x = max([
                float(instrument_panorama.metadata["SlideX1PosUm"]),
                float(instrument_panorama.metadata["SlideX2PosUm"]),
                float(instrument_panorama.metadata["SlideX3PosUm"]),
                float(instrument_panorama.metadata["SlideX4PosUm"]),
                ])

            min_x = min([
                float(instrument_panorama.metadata["SlideX1PosUm"]),
                float(instrument_panorama.metadata["SlideX2PosUm"]),
                float(instrument_panorama.metadata["SlideX3PosUm"]),
                float(instrument_panorama.metadata["SlideX4PosUm"]),
                ])

            max_y = max([
                float(instrument_panorama.metadata["SlideY1PosUm"]),
                float(instrument_panorama.metadata["SlideY2PosUm"]),
                float(instrument_panorama.metadata["SlideY3PosUm"]),
                float(instrument_panorama.metadata["SlideY4PosUm"]),
                ])

            min_y = min([
                float(instrument_panorama.metadata["SlideY1PosUm"]),
                float(instrument_panorama.metadata["SlideY2PosUm"]),
                float(instrument_panorama.metadata["SlideY3PosUm"]),
                float(instrument_panorama.metadata["SlideY4PosUm"]),
                ])

            panorama_arr = f.read_panorama(instrument_panorama)
            panorama_dims = panorama_arr.shape[:2]

            # convert x pixels to um
            x_um_per_pixel = panorama_dims[1] / (max_x - min_x)
            y_um_per_pixel = panorama_dims[0] / (max_y - min_y)

            background = Image.fromarray(panorama_arr).convert("RGBA")

            # rotate background
            background = background.rotate(rotation_angle, expand=True)

            if flip_axis == 0:
                # flip on x axis
                background = background.transpose(Image.FLIP_LEFT_RIGHT)
            elif flip_axis == 1:
                # flip on y axis
                background = background.transpose(Image.FLIP_TOP_BOTTOM)


            combined = background
            for acquisition_idx, acquisition in tqdm.tqdm(
                enumerate(slide.acquisitions),
                position=1,
                desc="Acquisitions"
            ):
                channel_names = [(a if a else b) for (a, b) in
                                 zip(acquisition.channel_labels, acquisition.channel_names)]

                if channel_label not in channel_names:
                    raise ValueError(f"Channel {channel_label} not found in {channel_names}")

                channel_idx = channel_names.index(channel_label)

                img = f.read_acquisition(acquisition)

                acquisition_arr = remove_outliers(img[channel_idx, ...])

                # Source: https://software.docs.hubmapconsortium.org/assays/imc.html
                # "ROIStartXPosUm" and "ROIStartYPosUm"	Start X and Y-coordinates of the region of interest (µm).
                # Note: This value must be divided by 1000 to correct for a bug (missing decimal point) in the Fluidigm software.
                x1 = float(acquisition.metadata["ROIStartXPosUm"])/1000.0
                y1 = float(acquisition.metadata["ROIStartYPosUm"])/1000.0
                x2 = float(acquisition.metadata["ROIEndXPosUm"])
                y2 = float(acquisition.metadata["ROIEndYPosUm"])

                x_min_acq = min(x1, x2)
                x_max_acq = max(x1, x2)
                y_min_acq = min(y1, y2)
                y_max_acq = max(y1, y2)

                if min_x < x_max_acq < max_x and min_y < y_max_acq < max_y:
                    pass
                else:
                    logger.warning(f"Acquisition {acquisition_idx} is outside of the panorama bounds, will ignore")
                    logger.warning(f"Bounding box of acquisition: {x_min_acq}, {x_max_acq}, {y_min_acq}, {y_max_acq}")
                    logger.warning(f"Bounding box of panorama: {min_x}, {max_x}, {min_y}, {max_y}")
                    continue

                x_pixel, y_pixel = int((x_min_acq-min_x) / x_um_per_pixel), int((max_y-y_max_acq) / y_um_per_pixel)
                x_extent, y_extent = int((x_max_acq - x_min_acq) / x_um_per_pixel), int((y_max_acq - y_min_acq) / y_um_per_pixel)

                overlay = Image.fromarray(acquisition_arr).convert("L")

                # Resize the overlay image
                overlay = overlay.resize((x_extent, y_extent))

                overlay = greyscale_to_colored_transparency(overlay, color=overlay_color)

                # Prepare a blank (alpha 0) image with the same size as the background
                temp = Image.new('RGBA', background.size, (0, 0, 0, 0))

                # Paste the resized overlay onto the temporary image
                temp.paste(overlay, (x_pixel, y_pixel))

                temp.save(os.path.join(output_dir, f"slide{slide_idx}_acquisition{acquisition_idx}_{channel_label}.png"))

                # Composite the temporary image onto the background
                combined = Image.alpha_composite(combined, temp)

            combined.save(os.path.join(output_dir, f"panorama_overlay_{slide_idx}_{channel_label}.png"))


def greyscale_to_colored_transparency(greyscale_image, color=(57, 255, 20)):
    arr = np.array(greyscale_image)
    alpha_channel = arr  # Alpha from grayscale image
    rgba_overlay = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    rgba_overlay[:, :, 0] = color[0]
    rgba_overlay[:, :, 1] = color[1]
    rgba_overlay[:, :, 2] = color[2]
    rgba_overlay[:, :, 3] = alpha_channel
    return Image.fromarray(rgba_overlay, 'RGBA')



def cellpose_segment(mcd_file: str,
                     cellpose_model: cellpose.models.Cellpose,
                     channels_to_use: list,

                     ):
    pass