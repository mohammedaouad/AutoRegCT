import SimpleITK as sitk
import numpy as np


def register_femur(fixed_mask, moving_mask, verbose=False):
    fixed = _prepare(fixed_mask)
    moving = _prepare(moving_mask)

    initial = sitk.CenteredTransformInitializer(
        fixed, moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )

    transform = _run(fixed, moving, initial, verbose)
    registered_mask = _apply(moving_mask, fixed_mask, transform)
    return registered_mask, transform


def _prepare(mask):
    # smooth before registration: helps with gradient computation on binary masks
    mask = sitk.Cast(mask, sitk.sitkFloat32)
    return sitk.SmoothingRecursiveGaussian(mask, sigma=1.0)


def _run(fixed, moving, initial_transform, verbose):
    reg = sitk.ImageRegistrationMethod()

    # Mattes MI works better than mean squares here since intensity ranges
    # can differ between pre/post-op scans
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.1)
    reg.SetInterpolator(sitk.sitkLinear)

    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    # multi-resolution pyramid: to avoid local minima
    reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initial_transform, inPlace=False)

    if verbose:
        reg.AddCommand(sitk.sitkIterationEvent, lambda: _log(reg))

    result = reg.Execute(fixed, moving)

    # sometimes returns CompositeTransform instead of Euler3DTransform
    if isinstance(result, sitk.CompositeTransform):
        t = result.GetNthTransform(0)
        euler = sitk.Euler3DTransform()
        euler.SetMatrix(t.GetMatrix())
        euler.SetTranslation(t.GetTranslation())
        euler.SetCenter(t.GetFixedParameters())
        return euler

    return result


def _apply(moving_mask, reference, transform):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # nearest neighbor to keep mask binary
    resampler.SetTransform(transform)
    resampler.SetDefaultPixelValue(0)
    return sitk.Cast(resampler.Execute(moving_mask), sitk.sitkUInt8)


def apply_transform_to_volume(moving, reference, transform, interpolator=sitk.sitkLinear):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interpolator)
    resampler.SetTransform(transform)
    resampler.SetDefaultPixelValue(-1000)
    return resampler.Execute(moving)


def get_transform_params(transform):
    t = transform if not isinstance(transform, sitk.CompositeTransform) \
        else transform.GetNthTransform(0)
    matrix = np.array(t.GetMatrix()).reshape(3, 3)
    translation = np.array(t.GetTranslation())
    return matrix, translation


def _log(reg):
    print(f"  iter {reg.GetOptimizerIteration():3d}  metric={reg.GetMetricValue():.5f}  lr={reg.GetOptimizerLearningRate():.4f}")
