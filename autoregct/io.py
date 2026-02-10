import os
import SimpleITK as sitk


def load_volume(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")

    if os.path.isdir(path):
        return _load_dicom_series(path)

    ext = _get_extension(path)
    if ext in (".nii", ".gz"):
        return sitk.ReadImage(path)

    raise ValueError(f"Unsupported format: {path}")


def _load_dicom_series(directory):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)

    if not dicom_names:
        raise RuntimeError(f"No DICOM series found in: {directory}")

    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    return reader.Execute()


def _get_extension(path):
    base = path.lower()
    # .nii.gz needs special handling since splitext only gets .gz
    if base.endswith(".nii.gz"):
        return ".gz"
    _, ext = os.path.splitext(base)
    return ext


def save_volume(image, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sitk.WriteImage(image, path)


def resample_to_reference(moving, reference, interpolator=sitk.sitkLinear, default_value=0.0):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetTransform(sitk.Transform())
    return resampler.Execute(moving)


def check_compatible(image_a, image_b):
    return (
        image_a.GetSize() == image_b.GetSize()
        and image_a.GetSpacing() == image_b.GetSpacing()
        and image_a.GetOrigin() == image_b.GetOrigin()
    )
