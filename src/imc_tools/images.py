import tqdm
from readimc import MCDFile
from glob import glob
import os.path
from pathlib import Path
from PIL import Image
import numpy as np

from matplotlib import pyplot as plt


def remove_outliers(arr, percentile=99.5):
    arr = arr.copy()
    max_value = np.percentile(arr, percentile)
    arr[arr > max_value] = max_value
    return arr


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
                               output_file,
                               panorama_idx=0,
                               channel_label="DNA1"):
    with MCDFile(mcd_file) as f:
        for slide_idx, slide in enumerate(f.slides):
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
            x_um_per_pixel = panorama_dims[0] / (max_x - min_x)
            y_um_per_pixel = panorama_dims[1] / (max_y - min_y)

            print(min_x, max_x, min_y, max_y, panorama_dims, x_um_per_pixel, y_um_per_pixel)
            background = Image.fromarray(panorama_arr).convert("RGBA")
            combined = background
            for acquisition_idx, acquisition in enumerate(slide.acquisitions):
                channel_names = [(a if a else b) for (a, b) in
                                 zip(acquisition.channel_labels, acquisition.channel_names)]

                if channel_label not in channel_names:
                    raise ValueError(f"Channel {channel_label} not found in {channel_names}")

                channel_idx = channel_names.index(channel_label)

                img = f.read_acquisition(acquisition)

                acquisition_arr = remove_outliers(img[channel_idx, ...])

                """
                “ROIEndXPosUm”	End X-coordinates of the region of interest (µm).
                “ROIEndYPosUm”	End Y-coordinates of the region of interest (µm)
                “ROIStartXPosUm”	Start X-coordinates of the region of interest (µm). Note: This value must be divided by 1000 to correct for a bug (missing decimal point) in the Fluidigm software.
                “ROIStartYPosUm”	Start Y-coordinates of the region of interest (µm). Note: This value must be divided by 1000 to correct for a bug (missing decimal point) in the Fluidigm software.
                """

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
                    print("out of bounds")
                    continue

                print(x_min_acq, x_max_acq, y_min_acq, y_max_acq)
                x_pixel, y_pixel = int((x_min_acq-min_x) / x_um_per_pixel), int((max_y-y_max_acq) / y_um_per_pixel)
                x_extent, y_extent = int((x_max_acq - x_min_acq) / x_um_per_pixel), int((y_max_acq - y_min_acq) / y_um_per_pixel)
                print(x_pixel, y_pixel, x_extent, y_extent)

                overlay = Image.fromarray(acquisition_arr).convert("L")

                # Resize the overlay image
                overlay = overlay.resize((x_extent, y_extent))

                overlay = greyscale_to_colored(overlay)

                # Prepare a blank (alpha 0) image with the same size as the background
                temp = Image.new('RGBA', background.size, (0, 0, 0, 0))

                # Paste the resized overlay onto the temporary image
                temp.paste(overlay, (x_pixel, y_pixel))

                temp.save("/tmp/temp.png")

                # Composite the temporary image onto the background
                combined = Image.alpha_composite(combined, temp)

    combined.save(output_file)


def greyscale_to_colored(greyscale_image, color=(57, 255, 20)):

    arr = np.array(greyscale_image)
    alpha_channel = arr  # Alpha from grayscale image
    rgba_overlay = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    rgba_overlay[:, :, 0] = color[0]
    rgba_overlay[:, :, 1] = color[1]
    rgba_overlay[:, :, 2] = color[2]
    rgba_overlay[:, :, 3] = alpha_channel
    return Image.fromarray(rgba_overlay, 'RGBA')

