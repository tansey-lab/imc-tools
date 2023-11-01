import os
import re
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from readimc import MCDFile

from imc_tools.images import black_to_alpha, logger, \
    AcquisitionOutOfBoundsError, remove_outliers, greyscale_to_colored_transparency


def get_acquisition(slide_index, acquisition_index, mcd_file):
    for slide_idx, slide in enumerate(mcd_file.slides):
        if slide_idx == slide_index:
            for acquisition_idx, acquisition in enumerate(slide.acquisitions):
                if acquisition_idx == acquisition_index:
                    return slide, acquisition
    raise ValueError(f"Could not find slide index {slide_index} acquisition index {acquisition_index} in {mcd_file}")


def get_panorama(slide_index, panorama_index, mcd_file):
    for slide_idx, slide in enumerate(mcd_file.slides):
        for panorama_idx, panorama in enumerate(slide.panoramas):
            if panorama_idx == panorama_index and slide_idx == slide_index:
                if panorama.metadata["Type"] == "Instrument":
                    return panorama
    raise ValueError(f"Could not find panorama index {panorama_index} in {mcd_file}")


def extract_channels_to_tiff(
    mcd_fn, output_file, selected_channels, slide_idx=0, acquisition_idx=0
):
    with MCDFile(mcd_fn) as f:
        slide, acquisition = get_acquisition(slide_idx, acquisition_idx, f)

        acquisition_arr = f.read_acquisition(acquisition)

        combined_array = extract_channels(
            acquisition, acquisition_arr, selected_channels
        )

        # Step 4: Convert the N-channel array back to a Pillow image
        # Note: Standard image formats may not support N-channel images.
        # Here, we're relying on TIFF's flexibility.
        combined_image = Image.fromarray(np.uint8(combined_array))

        # Step 5: Save the new image as a TIFF file
        combined_image.save(output_file)


def extract_channel_to_arr(
    mcd_fn, selected_channel, slide_idx=0, acquisition_idx=0
):
    with MCDFile(mcd_fn) as f:
        slide, acquisition = get_acquisition(slide_idx, acquisition_idx, f)

        acquisition_arr = f.read_acquisition(acquisition)
        return extract_channel(
            acquisition, acquisition_arr, selected_channel
        )


def extract_all_channels_to_arr(
    mcd_fn, slide_idx=0, acquisition_idx=0
):
    with MCDFile(mcd_fn) as f:
        slide, acquisition = get_acquisition(slide_idx, acquisition_idx, f)
        acquisition_arr = f.read_acquisition(acquisition)
        return extract_all_used_channels(
            acquisition, acquisition_arr
        )


def overlay_rgb_on_panorama(
    mcd_file: str, red=None, green=None, blue=None,
    slide_index=0,
    panorama_index=0
):
    with MCDFile(mcd_file) as f:
        slide, panorama = get_panorama(mcd_file=f,
                                       slide_index=slide_index,
                                       panorama_index=panorama_index)

        panorama_img = Image.fromarray(f.read_panorama(panorama)).convert("RGBA")

        roi_images = []

        for acquisition_idx, acquisition in tqdm.tqdm(
            enumerate(slide.acquisitions),
            position=1,
            total=len(slide.acquisitions),
            desc="Acquisitions",
        ):
            acquisition_arr = f.read_acquisition(acquisition)
            rbg_img = Image.fromarray(
                extract_channels_to_rgb(
                    acquisition, acquisition_arr, red=red, green=green, blue=blue
                )
            ).convert("RGBA")

            rbg_img = black_to_alpha(rbg_img)

            roi_images.append(rbg_img)

            panorama_img = resize_and_overlay_roi_in_panorama(
                panorama_object=panorama,
                panorama_image=panorama_img,
                acquisition_object=acquisition,
                roi_image=rbg_img,
            )

        return panorama_img, roi_images


def plot_mask_on_to_panorama(
    panorama_object, panorama_image, mask_arr, acquisition, alpha_channel=87
):
    # list of rainbow colors
    colors = [
        (255, 0, 0, alpha_channel),  # red
        (255, 127, 0, alpha_channel),  # orange
        (255, 255, 0, alpha_channel),  # yellow
        (0, 255, 0, alpha_channel),  # green
        (0, 0, 255, alpha_channel),  # blue
        (75, 0, 130, alpha_channel),  # indigo
        (148, 0, 211, alpha_channel),  # violet
    ]

    # for each unique value in mask arr, map to a color from the list in the image
    mask_color_map = {}
    for idx, val in enumerate(np.unique(mask_arr)):
        if val == 0:
            mask_color_map[val] = (0, 0, 0, 0)
        else:
            mask_color_map[val] = colors[idx % len(colors)]

    output_image_array = np.zeros(
        (mask_arr.shape[0], mask_arr.shape[1], 4), dtype=np.uint8
    )

    # convert mask arr to RGBA image
    for x in range(mask_arr.shape[0]):
        for y in range(mask_arr.shape[1]):
            output_image_array[x, y, :] = mask_color_map[mask_arr[x, y]]

    mask = black_to_alpha(Image.fromarray(output_image_array, "RGBA"))

    return resize_and_overlay_roi_in_panorama(
        panorama_object=panorama_object,
        panorama_image=panorama_image,
        acquisition_object=acquisition,
        roi_image=mask,
    )


def resize_and_overlay_roi_in_panorama(
    panorama_object, panorama_image, acquisition_object, roi_image
):
    if panorama_image.mode != "RGBA":
        raise ValueError("Panorama image must be RGBA")
    if roi_image.mode != "RGBA":
        raise ValueError("ROI image must be RGBA")

    max_x = max(
        [
            float(panorama_object.metadata["SlideX1PosUm"]),
            float(panorama_object.metadata["SlideX2PosUm"]),
            float(panorama_object.metadata["SlideX3PosUm"]),
            float(panorama_object.metadata["SlideX4PosUm"]),
        ]
    )

    min_x = min(
        [
            float(panorama_object.metadata["SlideX1PosUm"]),
            float(panorama_object.metadata["SlideX2PosUm"]),
            float(panorama_object.metadata["SlideX3PosUm"]),
            float(panorama_object.metadata["SlideX4PosUm"]),
        ]
    )

    max_y = max(
        [
            float(panorama_object.metadata["SlideY1PosUm"]),
            float(panorama_object.metadata["SlideY2PosUm"]),
            float(panorama_object.metadata["SlideY3PosUm"]),
            float(panorama_object.metadata["SlideY4PosUm"]),
        ]
    )

    min_y = min(
        [
            float(panorama_object.metadata["SlideY1PosUm"]),
            float(panorama_object.metadata["SlideY2PosUm"]),
            float(panorama_object.metadata["SlideY3PosUm"]),
            float(panorama_object.metadata["SlideY4PosUm"]),
        ]
    )

    # convert x pixels to um
    x_um_per_pixel = panorama_image.size[0] / (max_x - min_x)
    y_um_per_pixel = panorama_image.size[1] / (max_y - min_y)

    # Source: https://software.docs.hubmapconsortium.org/assays/imc.html
    # "ROIStartXPosUm" and "ROIStartYPosUm"	Start X and Y-coordinates of the region of interest (µm).
    # Note: This value must be divided by 1000 to correct for a bug (missing decimal point) in the Fluidigm software.
    x1 = float(acquisition_object.metadata["ROIStartXPosUm"]) / 1000.0
    y1 = float(acquisition_object.metadata["ROIStartYPosUm"]) / 1000.0
    x2 = float(acquisition_object.metadata["ROIEndXPosUm"])
    y2 = float(acquisition_object.metadata["ROIEndYPosUm"])

    x_min_acq = min(x1, x2)
    x_max_acq = max(x1, x2)
    y_min_acq = min(y1, y2)
    y_max_acq = max(y1, y2)

    if min_x < x_max_acq < max_x and min_y < y_max_acq < max_y:
        pass
    else:
        logger.warning(f"Acquisition is outside of the panorama bounds")
        logger.warning(
            f"Bounding box of acquisition: {x_min_acq}, {x_max_acq}, {y_min_acq}, {y_max_acq}"
        )
        logger.warning(f"Bounding box of panorama: {min_x}, {max_x}, {min_y}, {max_y}")
        raise AcquisitionOutOfBoundsError(
            "Acquisition is outside of the panorama bounds"
        )

    x_pixel, y_pixel = int((x_min_acq - min_x) / x_um_per_pixel), int(
        (max_y - y_max_acq) / y_um_per_pixel
    )
    x_extent, y_extent = int((x_max_acq - x_min_acq) / x_um_per_pixel), int(
        (y_max_acq - y_min_acq) / y_um_per_pixel
    )

    # Resize the overlay image
    overlay = roi_image.resize((x_extent, y_extent))

    # Prepare a blank (alpha 0) image with the same size as the background
    temp = Image.new("RGBA", panorama_image.size, (0, 0, 0, 0))

    # Paste the resized overlay onto the temporary image
    temp.paste(overlay, (x_pixel, y_pixel))

    return Image.alpha_composite(panorama_image, temp)


def extract_channels_to_rgb(
    acquisition, acquisition_arr, red=None, green=None, blue=None
):
    output_arr = np.zeros(
        (acquisition_arr.shape[1], acquisition_arr.shape[2], 3), dtype=np.uint8
    )

    for idx, color_channel in enumerate([red, green, blue]):
        if isinstance(color_channel, str):
            arr = extract_channel(acquisition, acquisition_arr, color_channel)

        elif isinstance(color_channel, list):
            arr = extract_maximum_projection_of_channels(
                acquisition, acquisition_arr, color_channel
            )
        else:
            continue

        output_arr[:, :, idx] = normalize_and_scale_to_image_range(arr)

    return output_arr


def is_likely_unused_channel(channel_label, channel_name):
    if not channel_label:
        return True

    if channel_label.strip().lower() == channel_name.strip().lower():
        return True

    channel_label_numbers = re.findall("\d+", channel_label)
    channel_name_numbers = re.findall("\d+", channel_name)

    if not len(channel_label_numbers) == 1 and len(channel_name_numbers) == 1:
        return False

    channel_label_alphas = re.findall("[A-Za-z]+", channel_label)
    channel_name_alphas = re.findall("[A-Za-z]+", channel_name)

    if not len(channel_label_alphas) == 1 and len(channel_name_alphas) == 1:
        return False

    if (channel_label_alphas[0].lower() == channel_name_alphas[0].lower() and
        channel_label_numbers[0].lower() == channel_name_numbers[0].lower()):
        return True

    return False


def get_channels(acquisition):
    for idx, (label, name) in enumerate(zip(acquisition.channel_labels, acquisition.channel_names)):
        if is_likely_unused_channel(label, name):
            continue
        yield idx, label


def extract_channel(acquisition, acquisition_arr, selected_channel):
    channel_names = [x[1] for x in get_channels(acquisition)]

    if selected_channel not in channel_names:
        raise ValueError(f"Channel {selected_channel} not found in {channel_names}")

    channel_idx = [idx for (idx, name) in get_channels(acquisition) if name == selected_channel][0]

    array = remove_outliers(acquisition_arr[channel_idx, ...])
    return array


def extract_channels(acquisition, acquisition_arr, selected_channels):
    image_arrays = []
    for channel in selected_channels:
        arr = extract_channel(acquisition, acquisition_arr, channel)
        image_arrays.append(arr)

    # Step 3: Stack the N arrays to make an N-channel image
    combined_array = np.stack(image_arrays, axis=2)

    return np.uint8(combined_array)


def extract_all_used_channels(acquisition, acquisition_arr):
    all_channels = [c[1] for c in get_channels(acquisition)]

    return extract_channels(acquisition, acquisition_arr, all_channels)


def extract_maximum_projection_of_channels(
    acquisition, acquisition_arr, selected_channels
):
    arr = extract_channels(acquisition, acquisition_arr, selected_channels)
    return np.max(arr, axis=2)


def normalize_and_scale_to_image_range(arr):
    # scale so max value is 1.0
    arr = arr / np.max(arr)

    # scale so max value is 255
    arr = arr * 255
    return arr.astype(np.uint8)



def plot_channel_onto_panorama(
    mcd_file,
    output_dir,
    slide_index=0,
    panorama_index=0,
    overlay_color=(57, 255, 20),
    rotation_angle=0,
    flip_axis=None,
    channel_label="DNA1",
):
    """
    Plot a channel onto the panorama image

    :param mcd_file: Input MCD file for IMC experiment
    :param output_dir: Output dir where multiple images will be saved
    :param slide_index: If the MCD file happens to contain multiple slides,
                         this is the index of the slide to use
    :param panorama_index: If the MCD file happens to contain multiple panoramas,
                         this is the index of the panorama to use
    :param overlay_color: Color to use for transparent overlay of the channel
    :param rotation_angle: Rotate the background image by this much before overlay
    :param flip_axis: Flip background on this axis before overlay
    :param channel_label: Name of channel label to plot
    """
    with MCDFile(mcd_file) as f:
        slide, panorama = get_panorama(
            mcd_file=f,
            slide_index=slide_index,
            panorama_index=panorama_index,
        )

        max_x = max(
            [
                float(panorama.metadata["SlideX1PosUm"]),
                float(panorama.metadata["SlideX2PosUm"]),
                float(panorama.metadata["SlideX3PosUm"]),
                float(panorama.metadata["SlideX4PosUm"]),
            ]
        )

        min_x = min(
            [
                float(panorama.metadata["SlideX1PosUm"]),
                float(panorama.metadata["SlideX2PosUm"]),
                float(panorama.metadata["SlideX3PosUm"]),
                float(panorama.metadata["SlideX4PosUm"]),
            ]
        )

        max_y = max(
            [
                float(panorama.metadata["SlideY1PosUm"]),
                float(panorama.metadata["SlideY2PosUm"]),
                float(panorama.metadata["SlideY3PosUm"]),
                float(panorama.metadata["SlideY4PosUm"]),
            ]
        )

        min_y = min(
            [
                float(panorama.metadata["SlideY1PosUm"]),
                float(panorama.metadata["SlideY2PosUm"]),
                float(panorama.metadata["SlideY3PosUm"]),
                float(panorama.metadata["SlideY4PosUm"]),
            ]
        )

        panorama_arr = f.read_panorama(panorama)
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
            enumerate(slide.acquisitions), position=1, desc="Acquisitions"
        ):
            channel_labels = [x[1] for x in get_channels(acquisition)]

            if channel_label not in channel_labels:
                raise ValueError(
                    f"Channel {channel_label} not found in {channel_labels}"
                )

            channel_idx = [idx for idx, name in get_channels(acquisition) if name == channel_label][0]

            img = f.read_acquisition(acquisition)

            acquisition_arr = remove_outliers(img[channel_idx, ...])

            # Source: https://software.docs.hubmapconsortium.org/assays/imc.html
            # "ROIStartXPosUm" and "ROIStartYPosUm"	Start X and Y-coordinates of the region of interest (µm).
            # Note: This value must be divided by 1000 to correct for a bug (missing decimal point) in the Fluidigm software.
            x1 = float(acquisition.metadata["ROIStartXPosUm"]) / 1000.0
            y1 = float(acquisition.metadata["ROIStartYPosUm"]) / 1000.0
            x2 = float(acquisition.metadata["ROIEndXPosUm"])
            y2 = float(acquisition.metadata["ROIEndYPosUm"])

            x_min_acq = min(x1, x2)
            x_max_acq = max(x1, x2)
            y_min_acq = min(y1, y2)
            y_max_acq = max(y1, y2)

            if min_x < x_max_acq < max_x and min_y < y_max_acq < max_y:
                pass
            else:
                logger.warning(
                    f"Acquisition {acquisition_idx} is outside of the panorama bounds, will ignore"
                )
                logger.warning(
                    f"Bounding box of acquisition: {x_min_acq}, {x_max_acq}, {y_min_acq}, {y_max_acq}"
                )
                logger.warning(
                    f"Bounding box of panorama: {min_x}, {max_x}, {min_y}, {max_y}"
                )
                continue

            x_pixel, y_pixel = int((x_min_acq - min_x) / x_um_per_pixel), int(
                (max_y - y_max_acq) / y_um_per_pixel
            )
            x_extent, y_extent = int((x_max_acq - x_min_acq) / x_um_per_pixel), int(
                (y_max_acq - y_min_acq) / y_um_per_pixel
            )

            overlay = Image.fromarray(acquisition_arr).convert("L")

            # Resize the overlay image
            overlay = overlay.resize((x_extent, y_extent))

            overlay = greyscale_to_colored_transparency(
                overlay, color=overlay_color
            )

            # Prepare a blank (alpha 0) image with the same size as the background
            temp = Image.new("RGBA", background.size, (0, 0, 0, 0))

            # Paste the resized overlay onto the temporary image
            temp.paste(overlay, (x_pixel, y_pixel))

            temp.save(
                os.path.join(
                    output_dir,
                    f"slide{slide_index}_acquisition{acquisition_idx}_{channel_label}.png",
                )
            )

            # Composite the temporary image onto the background
            combined = Image.alpha_composite(combined, temp)

        combined.save(
            os.path.join(
                output_dir, f"panorama_overlay_{slide_index}_{channel_label}.png"
            )
        )
