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


class AcquisitionOutOfBoundsError(Exception):
    pass


def remove_outliers(arr, percentile=99.5):
    arr = arr.copy()
    max_value = np.percentile(arr, percentile)
    arr[arr > max_value] = max_value
    return arr


def extract_channel(acquisition,
                    acquisition_arr,
                    selected_channel):
    channel_names = [(a if a else b) for (a, b) in
                     zip(acquisition.channel_labels, acquisition.channel_names)]

    if selected_channel not in channel_names:
        raise ValueError(f"Channel {selected_channel} not found in {channel_names}")

    for channel_idx, channel in enumerate(channel_names):
        if channel == selected_channel:
            array = remove_outliers(acquisition_arr[channel_idx, ...])
            return array


def extract_channels(acquisition,
                     acquisition_arr,
                     selected_channels):
    image_arrays = []
    for channel in selected_channels:
        arr = extract_channel(acquisition,
                              acquisition_arr,
                              channel)
        image_arrays.append(arr)

    # Step 3: Stack the N arrays to make an N-channel image
    combined_array = np.stack(image_arrays, axis=2)

    return np.uint8(combined_array)


def extract_maximum_projection_of_channels(
    acquisition,
    acquisition_arr,
    selected_channels):
    arr = extract_channels(acquisition,
                           acquisition_arr,
                           selected_channels)
    return np.max(arr, axis=2)


def normalize_and_scale_to_image_range(arr):
    # scale so max value is 1.0
    arr = arr / np.max(arr)

    # scale so max value is 255
    arr = arr * 255
    return arr.astype(np.uint8)


def extract_channels_to_rgb(acquisition,
                            acquisition_arr,
                            red=None,
                            green=None,
                            blue=None
                            ):
    output_arr = np.zeros((acquisition_arr.shape[1], acquisition_arr.shape[2], 3), dtype=np.uint8)

    for idx, color_channel in enumerate([red, green, blue]):
        if isinstance(color_channel, str):
            arr = extract_channel(
                acquisition,
                acquisition_arr,
                color_channel
            )

        elif isinstance(color_channel, list):
            arr = extract_maximum_projection_of_channels(
                acquisition,
                acquisition_arr,
                color_channel
            )
        else:
            continue

        output_arr[:, :, idx] = normalize_and_scale_to_image_range(arr)

    return output_arr


def extract_channels_to_tiff(mcd_fn,
                             output_file,
                             selected_channels,
                             slide_idx=0,
                             acquisition_idx=0):
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
                if acquisition_idx != acquisition_idx and slide_idx != slide_idx:
                    continue

                acquisition_arr = f.read_acquisition(acquisition)

                combined_array = extract_channels(acquisition,
                                                  acquisition_arr,
                                                  selected_channels)

                # Step 4: Convert the N-channel array back to a Pillow image
                # Note: Standard image formats may not support N-channel images.
                # Here, we're relying on TIFF's flexibility.
                combined_image = Image.fromarray(np.uint8(combined_array))

                # Step 5: Save the new image as a TIFF file
                combined_image.save(output_file)
                return

    raise ValueError(f"Could not find slide {slide_idx} acquisition {acquisition_idx} in {mcd_fn}")


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
                    logger.warning(f"Acquisition {acquisition_idx} is outside of the panorama bounds, will ignore")
                    logger.warning(f"Bounding box of acquisition: {x_min_acq}, {x_max_acq}, {y_min_acq}, {y_max_acq}")
                    logger.warning(f"Bounding box of panorama: {min_x}, {max_x}, {min_y}, {max_y}")
                    continue

                x_pixel, y_pixel = int((x_min_acq - min_x) / x_um_per_pixel), int((max_y - y_max_acq) / y_um_per_pixel)
                x_extent, y_extent = int((x_max_acq - x_min_acq) / x_um_per_pixel), int(
                    (y_max_acq - y_min_acq) / y_um_per_pixel)

                overlay = Image.fromarray(acquisition_arr).convert("L")

                # Resize the overlay image
                overlay = overlay.resize((x_extent, y_extent))

                overlay = greyscale_to_colored_transparency(overlay, color=overlay_color)

                # Prepare a blank (alpha 0) image with the same size as the background
                temp = Image.new('RGBA', background.size, (0, 0, 0, 0))

                # Paste the resized overlay onto the temporary image
                temp.paste(overlay, (x_pixel, y_pixel))

                temp.save(
                    os.path.join(output_dir, f"slide{slide_idx}_acquisition{acquisition_idx}_{channel_label}.png"))

                # Composite the temporary image onto the background
                combined = Image.alpha_composite(combined, temp)

            combined.save(os.path.join(output_dir, f"panorama_overlay_{slide_idx}_{channel_label}.png"))


def greyscale_to_colored_transparency(greyscale_image,
                                      color=(57, 255, 20),
                                      fixed_alpha=None):
    arr = np.array(greyscale_image)
    alpha_channel = arr  # Alpha from grayscale image
    rgba_overlay = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    rgba_overlay[:, :, 0] = color[0]
    rgba_overlay[:, :, 1] = color[1]
    rgba_overlay[:, :, 2] = color[2]
    if fixed_alpha is not None:
        rgba_overlay[:, :, 3] = fixed_alpha
    else:
        rgba_overlay[:, :, 3] = alpha_channel
    return Image.fromarray(rgba_overlay, 'RGBA')


def resize_and_overlay_roi_in_panorama(
    panorama_object,
    panorama_image,
    acquisition_object,
    roi_image
):
    if panorama_image.mode != "RGBA":
        raise ValueError("Panorama image must be RGBA")
    if roi_image.mode != "RGBA":
        raise ValueError("ROI image must be RGBA")

    max_x = max([
        float(panorama_object.metadata["SlideX1PosUm"]),
        float(panorama_object.metadata["SlideX2PosUm"]),
        float(panorama_object.metadata["SlideX3PosUm"]),
        float(panorama_object.metadata["SlideX4PosUm"]),
    ])

    min_x = min([
        float(panorama_object.metadata["SlideX1PosUm"]),
        float(panorama_object.metadata["SlideX2PosUm"]),
        float(panorama_object.metadata["SlideX3PosUm"]),
        float(panorama_object.metadata["SlideX4PosUm"]),
    ])

    max_y = max([
        float(panorama_object.metadata["SlideY1PosUm"]),
        float(panorama_object.metadata["SlideY2PosUm"]),
        float(panorama_object.metadata["SlideY3PosUm"]),
        float(panorama_object.metadata["SlideY4PosUm"]),
    ])

    min_y = min([
        float(panorama_object.metadata["SlideY1PosUm"]),
        float(panorama_object.metadata["SlideY2PosUm"]),
        float(panorama_object.metadata["SlideY3PosUm"]),
        float(panorama_object.metadata["SlideY4PosUm"]),
    ])

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
        logger.warning(f"Bounding box of acquisition: {x_min_acq}, {x_max_acq}, {y_min_acq}, {y_max_acq}")
        logger.warning(f"Bounding box of panorama: {min_x}, {max_x}, {min_y}, {max_y}")
        raise AcquisitionOutOfBoundsError("Acquisition is outside of the panorama bounds")

    x_pixel, y_pixel = int((x_min_acq - min_x) / x_um_per_pixel), int((max_y - y_max_acq) / y_um_per_pixel)
    x_extent, y_extent = int((x_max_acq - x_min_acq) / x_um_per_pixel), int((y_max_acq - y_min_acq) / y_um_per_pixel)

    # Resize the overlay image
    overlay = roi_image.resize((x_extent, y_extent))

    # Prepare a blank (alpha 0) image with the same size as the background
    temp = Image.new('RGBA', panorama_image.size, (0, 0, 0, 0))

    # Paste the resized overlay onto the temporary image
    temp.paste(overlay, (x_pixel, y_pixel))

    return Image.alpha_composite(panorama_image, temp)


def plot_mask_on_to_panorama(
    panorama_object,
    panorama_image,
    mask_arr,
    acquisition,
    mask_color=(57, 255, 20),
    alpha_channel=87
):
    # list of rainbow colors
    colors = [
        (255, 0, 0, alpha_channel), #red
        (255, 127, 0, alpha_channel), #orange
        (255, 255, 0, alpha_channel), #yellow
        (0, 255, 0, alpha_channel), #green
        (0, 0, 255, alpha_channel), #blue
        (75, 0, 130, alpha_channel), #indigo
        (148, 0, 211, alpha_channel), #violet
    ]



    mask = Image.fromarray(mask_arr.astype(np.uint8)).convert("L")
    mask_overlay = greyscale_to_colored_transparency(mask, mask_color, fixed_alpha=100)

    return resize_and_overlay_roi_in_panorama(
        panorama_object=panorama_object,
        panorama_image=panorama_image,
        acquisition_object=acquisition,
        roi_image=mask_overlay
    )


def black_to_alpha(image):
    image_array = np.array(image)

    # Calculate "blackness" based on RGB values (average for this example)
    new_alpha = np.max(image_array[:, :, :3], axis=2).astype(np.uint8)

    # Assign the new alpha channel back to the image array
    image_array[:, :, 3] = new_alpha

    # Convert the NumPy array back to a PIL image and save
    return Image.fromarray(image_array, 'RGBA')


def overlay_rgb_on_panorama(
        mcd_file: str,
                         red=None,
        green=None,
        blue=None,
                         panorama_index=0):

    with MCDFile(mcd_file) as f:
        for slide_idx, slide in tqdm.tqdm(
            enumerate(f.slides),
            total=len(f.slides),
            position=0,
            desc="Slides"
        ):
            panorama = [x for x in slide.panoramas if x.metadata["Type"] == "Instrument"][panorama_index]
            panorama_img = Image.fromarray(f.read_panorama(panorama)).convert("RGBA")

            roi_images = []

            for acquisition_idx, acquisition in tqdm.tqdm(
                enumerate(slide.acquisitions),
                position=1,
                total=len(slide.acquisitions),
                desc="Acquisitions"
            ):
                acquisition_arr = f.read_acquisition(acquisition)
                rbg_img = Image.fromarray(extract_channels_to_rgb(
                    acquisition,
                                               acquisition_arr,
                                                red=red,
                    green=green,
                    blue=blue
                )).convert("RGBA")

                rbg_img = black_to_alpha(rbg_img)

                roi_images.append(rbg_img)

                panorama_img = resize_and_overlay_roi_in_panorama(
                    panorama_object=panorama,
                    panorama_image=panorama_img,
                    acquisition_object=acquisition,
                    roi_image=rbg_img
                )

            return panorama_img, roi_images


def cellpose_segment(mcd_file: str,
                     nuclear_channel: str,
                     panorama_index=0):
    cellpose_model = cellpose.models.Cellpose(model_type='nuclei')

    channels_to_use = [nuclear_channel]
    # channels_to_use = [cytoplasm_channel] if nuclear_channel is None else [cytoplasm_channel, nuclear_channel]

    with MCDFile(mcd_file) as f:
        for slide_idx, slide in tqdm.tqdm(
            enumerate(f.slides),
            total=len(f.slides),
            position=0,
            desc="Slides"
        ):
            acquisition_arrays = []
            acquisition_objects = []


            panorama = [x for x in slide.panoramas if x.metadata["Type"] == "Instrument"][panorama_index]
            panorama_img = Image.fromarray(f.read_panorama(panorama)).convert("RGBA")

            for acquisition_idx, acquisition in tqdm.tqdm(
                enumerate(slide.acquisitions),
                position=1,
                total=len(slide.acquisitions),
                desc="Acquisitions"
            ):
                acquisition_arr = f.read_acquisition(acquisition)
                channel_data = extract_channels(acquisition, acquisition_arr, channels_to_use)
                acquisition_arrays.append(channel_data)
                acquisition_objects.append(acquisition)

            masks, flows, styles, diams = cellpose_model.eval(acquisition_arrays,
                                                              diameter=None,
                                                              channels=[x + 1 for x in range(len(channels_to_use))] + [
                                                                  0]
                                                              )
            mask_arrays= []
            for mask, acquisition in zip(masks, acquisition_objects):
                try:
                    mask_arrays.append(mask.copy())
                    mask[mask > 0] = 255

                    panorama_img = plot_mask_on_to_panorama(
                        panorama_object=panorama,
                        panorama_image=panorama_img,
                        mask_arr=mask,
                        acquisition=acquisition,
                    )
                except AcquisitionOutOfBoundsError:
                    continue

            return panorama_img, mask_arrays


def deepcell_nuclear_segment(
    mcd_file: str,
    channels_to_use=None,
    panorama_index=0
):
    import deepcell.applications
    app = deepcell.applications.NuclearSegmentation()
    with MCDFile(mcd_file) as f:
        for slide_idx, slide in tqdm.tqdm(
            enumerate(f.slides),
            total=len(f.slides),
            position=0,
            desc="Slides"
        ):
            labeled_rois = []

            panorama = [x for x in slide.panoramas if x.metadata["Type"] == "Instrument"][panorama_index]
            panorama_img = Image.fromarray(f.read_panorama(panorama)).convert("RGBA")

            for acquisition_idx, acquisition in tqdm.tqdm(
                enumerate(slide.acquisitions),
                position=1,
                total=len(slide.acquisitions),
                desc="Acquisitions"
            ):
                acquisition_arr = f.read_acquisition(acquisition)

                if len(channels_to_use) == 1:
                    im = remove_outliers(extract_channel(
                        acquisition, acquisition_arr, channels_to_use[0]))[np.newaxis, ..., np.newaxis]
                elif len(channels_to_use) > 1:
                    im = remove_outliers(extract_maximum_projection_of_channels(
                        acquisition, acquisition_arr, channels_to_use))[np.newaxis, ..., np.newaxis]

                labeled_nuclear_arr = app.predict(im, image_mpp=1.0)

                labeled_nuclear_arr = labeled_nuclear_arr[0, :, :, 0]

                labeled_rois.append(labeled_nuclear_arr.copy())

                labeled_nuclear_arr[labeled_nuclear_arr > 0] = 255

                panorama_img = plot_mask_on_to_panorama(
                    panorama_object=panorama,
                    panorama_image=panorama_img,
                    mask_arr=labeled_nuclear_arr,
                    acquisition=acquisition,
                )
            return panorama_img, labeled_rois