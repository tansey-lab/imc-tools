import cellpose.models
import numpy as np
import tqdm
from PIL import Image
from readimc import MCDFile
from imc_tools.mcd import get_acquisition, AcquisitionOutOfBoundsError

from imc_tools.images import remove_outliers, apply_clahe, get_image_gradient, get_segmented_image, \
    get_mixture_of_two_gaussians_cutoff_point
from imc_tools.mcd import plot_mask_on_to_panorama, extract_channels_to_rgb, extract_channel, extract_channels, \
    extract_maximum_projection_of_channels


def cellpose_cyto_segment(
    mcd_file: str, nuclear_channels, cyto_channels, panorama_index=0, masks=None
):
    cellpose_model = cellpose.models.Cellpose(model_type="cyto")

    with MCDFile(mcd_file) as f:
        for slide_idx, slide in tqdm.tqdm(
            enumerate(f.slides), total=len(f.slides), position=0, desc="Slides"
        ):
            acquisition_arrays = []
            acquisition_objects = []

            panorama = [
                x for x in slide.panoramas if x.metadata["Type"] == "Instrument"
            ][panorama_index]
            panorama_img = Image.fromarray(f.read_panorama(panorama)).convert("RGBA")

            for acquisition_idx, acquisition in tqdm.tqdm(
                enumerate(slide.acquisitions),
                position=1,
                total=len(slide.acquisitions),
                desc="Acquisitions",
            ):
                acquisition_arr = f.read_acquisition(acquisition)

                rgb_data = extract_channels_to_rgb(
                    acquisition=acquisition,
                    acquisition_arr=acquisition_arr,
                    red=nuclear_channels,
                    green=cyto_channels,
                )

                acquisition_arrays.append(rgb_data)
                acquisition_objects.append(acquisition)

            if masks is None:
                masks, flows, styles, diams = cellpose_model.eval(
                    acquisition_arrays, diameter=None, channels=[2, 1]
                )
            mask_arrays = []
            for mask, acquisition in zip(masks, acquisition_objects):
                try:
                    mask_arrays.append(mask.copy())

                    panorama_img = plot_mask_on_to_panorama(
                        panorama_object=panorama,
                        panorama_image=panorama_img,
                        mask_arr=mask,
                        acquisition=acquisition,
                    )
                except AcquisitionOutOfBoundsError:
                    continue

            return panorama_img, mask_arrays


def deepcell_nuclear_segment(mcd_file: str, channels_to_use, slide_index=0, acquisition_index=0):
    import deepcell.applications

    app = deepcell.applications.NuclearSegmentation()

    with MCDFile(mcd_file) as f:
        slide, acquisition = get_acquisition(mcd_file=f,
                                             slide_index=slide_index,
                                             acquisition_index=acquisition_index)


        acquisition_arr = f.read_acquisition(acquisition)
        channel_data = extract_channels(
            acquisition, acquisition_arr, channels_to_use
        )
        if len(channels_to_use) == 1:
            channel_data = remove_outliers(
                extract_channel(
                    acquisition, acquisition_arr, channels_to_use[0]
                )
            )[np.newaxis, ..., np.newaxis]
        elif len(channels_to_use) > 1:
            channel_data = remove_outliers(
                extract_maximum_projection_of_channels(
                    acquisition, acquisition_arr, channels_to_use
                )
            )[np.newaxis, ..., np.newaxis]
        labeled_nuclear_arr = app.predict(channel_data, image_mpp=1.0)
        labeled_nuclear_arr = labeled_nuclear_arr[0, :, :, 0]

        return labeled_nuclear_arr


def deepcell_nuclear_segment_all_and_plot(mcd_file: str, channels_to_use=None, panorama_index=0):
    import deepcell.applications

    app = deepcell.applications.NuclearSegmentation()
    with MCDFile(mcd_file) as f:
        for slide_idx, slide in tqdm.tqdm(
            enumerate(f.slides), total=len(f.slides), position=0, desc="Slides"
        ):
            labeled_rois = []

            panorama = [
                x for x in slide.panoramas if x.metadata["Type"] == "Instrument"
            ][panorama_index]
            panorama_img = Image.fromarray(f.read_panorama(panorama)).convert("RGBA")

            for acquisition_idx, acquisition in tqdm.tqdm(
                enumerate(slide.acquisitions),
                position=1,
                total=len(slide.acquisitions),
                desc="Acquisitions",
            ):
                acquisition_arr = f.read_acquisition(acquisition)

                if len(channels_to_use) == 1:
                    im = remove_outliers(
                        extract_channel(
                            acquisition, acquisition_arr, channels_to_use[0]
                        )
                    )[np.newaxis, ..., np.newaxis]
                elif len(channels_to_use) > 1:
                    im = remove_outliers(
                        extract_maximum_projection_of_channels(
                            acquisition, acquisition_arr, channels_to_use
                        )
                    )[np.newaxis, ..., np.newaxis]

                labeled_nuclear_arr = app.predict(im, image_mpp=1.0)
                labeled_rois.append(labeled_nuclear_arr.copy())
                labeled_nuclear_arr = labeled_nuclear_arr[0, :, :, 0]

                panorama_img = plot_mask_on_to_panorama(
                    panorama_object=panorama,
                    panorama_image=panorama_img,
                    mask_arr=labeled_nuclear_arr,
                    acquisition=acquisition,
                )
            return panorama_img, labeled_rois


def cellpose_segment(mcd_file: str, channels_to_use, slide_index=0, acquisition_index=0,
                     flow_threshold=0.4,
                     cellprob_threshold=0):
    cellpose_model = cellpose.models.Cellpose(model_type="nuclei")

    with MCDFile(mcd_file) as f:
        slide, acquisition = get_acquisition(mcd_file=f,
                                             slide_index=slide_index,
                                             acquisition_index=acquisition_index)


        acquisition_arr = f.read_acquisition(acquisition)
        channel_data = extract_channels(
            acquisition, acquisition_arr, channels_to_use
        )
        if len(channels_to_use) == 1:
            channel_data = remove_outliers(
                extract_channel(
                    acquisition=acquisition,
                    acquisition_arr=acquisition_arr,
                    selected_channel=channels_to_use[0]
                )
            )
        elif len(channels_to_use) > 1:
            channel_data = remove_outliers(
                extract_maximum_projection_of_channels(
                    acquisition=acquisition,
                    acquisition_arr=acquisition_arr,
                    selected_channels=channels_to_use
                )
            )

        masks, flows, styles, diams = cellpose_model.eval(
            [channel_data], diameter=None, channels=[0, 0],
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold
        )

        return masks, flows, styles, diams


def cellpose_segment_all_and_plot(mcd_file: str,
                     channels_to_use,
                     panorama_index=0,
                     masks=None):
    cellpose_model = cellpose.models.Cellpose(model_type="nuclei")

    with MCDFile(mcd_file) as f:
        for slide_idx, slide in tqdm.tqdm(
            enumerate(f.slides), total=len(f.slides), position=0, desc="Slides"
        ):
            acquisition_arrays = []
            acquisition_objects = []

            panorama = [
                x for x in slide.panoramas if x.metadata["Type"] == "Instrument"
            ][panorama_index]
            panorama_img = Image.fromarray(f.read_panorama(panorama)).convert("RGBA")

            for acquisition_idx, acquisition in tqdm.tqdm(
                enumerate(slide.acquisitions),
                position=1,
                total=len(slide.acquisitions),
                desc="Acquisitions",
            ):
                acquisition_arr = f.read_acquisition(acquisition)
                channel_data = extract_channels(
                    acquisition, acquisition_arr, channels_to_use
                )
                if len(channels_to_use) == 1:
                    channel_data = remove_outliers(
                        extract_channel(
                            acquisition, acquisition_arr, channels_to_use[0]
                        )
                    )
                elif len(channels_to_use) > 1:
                    channel_data = remove_outliers(
                        extract_maximum_projection_of_channels(
                            acquisition, acquisition_arr, channels_to_use
                        )
                    )

                acquisition_arrays.append(channel_data)
                acquisition_objects.append(acquisition)

            if masks is None:
                masks, flows, styles, diams = cellpose_model.eval(
                    acquisition_arrays, diameter=None, channels=[0, 0]
                )
            mask_arrays = []
            for mask, acquisition in zip(masks, acquisition_objects):
                try:
                    mask_arrays.append(mask.copy())

                    panorama_img = plot_mask_on_to_panorama(
                        panorama_object=panorama,
                        panorama_image=panorama_img,
                        mask_arr=mask,
                        acquisition=acquisition,
                    )
                except AcquisitionOutOfBoundsError:
                    continue

            return panorama_img, mask_arrays


def get_total_intensity_per_segment(original_image, segmented_image):
    if original_image.shape != segmented_image.shape:
        raise ValueError("Image and segmentation must have same shape")

    unique_segments = np.unique(segmented_image)

    means = {}
    sums = {}

    for unique_segment in unique_segments:
        means[unique_segment] = original_image[(segmented_image == unique_segment)].mean()
        sums[unique_segment] = original_image[(segmented_image == unique_segment)].sum()

    return means, sums


def get_cell_body_mask(all_channel_sum):
    clahe_image = apply_clahe(all_channel_sum)
    gradient_image = get_image_gradient(clahe_image)
    segmented_image = get_segmented_image(gradient_image, num_regions=500)
    mean_intensity_per_segment, total_intensity_per_segment = get_total_intensity_per_segment(
        original_image=all_channel_sum,
        segmented_image=segmented_image
    )
    cutoff_point = get_mixture_of_two_gaussians_cutoff_point(
        np.array([x for x in mean_intensity_per_segment.values()])
    )

    cell_body_segments = [
     segment_id for segment_id, val in mean_intensity_per_segment.items()
     if val >= cutoff_point]

    return np.isin(segmented_image, cell_body_segments)


def get_cell_body_mask_with_nuclear_segmentation(all_channel_sum, segmentation_mask, num_regions=500):
    clahe_image = apply_clahe(all_channel_sum)
    gradient_image = get_image_gradient(clahe_image)
    segmented_image = get_segmented_image(gradient_image, num_regions=num_regions)

    segmentation_mask = segmentation_mask.copy()
    segmentation_mask[segmentation_mask > 0] = 1

    mean_intensity_per_segment, total_intensity_per_segment = get_total_intensity_per_segment(
        original_image=segmentation_mask,
        segmented_image=segmented_image
    )
    cutoff_point = get_mixture_of_two_gaussians_cutoff_point(
        np.array([x for x in mean_intensity_per_segment.values()])
    )

    cell_body_segments = [
     segment_id for segment_id, val in mean_intensity_per_segment.items()
     if val >= cutoff_point]

    return np.isin(segmented_image, cell_body_segments)
