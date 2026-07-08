"""
  .. versionadded:: 0.7.0
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field as dataclass_field
from typing import Callable

from .._ndarray_backend import xp


def finalize_fourier_field(
    fft_in: xp.ndarray,
    ifft_fn: Callable[[xp.ndarray], xp.ndarray],
    input_shape: tuple[int, int],
    fft_shape: tuple[int, int],
    padding: bool | int,
    scale_to_filter: bool | float,
    crop_radius: int | None = None,
) -> xp.ndarray:
    """Convert a propagated Fourier field back to the spatial domain.

    Applies ``ifftshift`` to move DC from the centre to index 0, then
    calls the inverse FFT.  If padding or Fourier-space cropping was used
    during filtering the result is trimmed and rescaled to match the
    original hologram dimensions.

    Parameters
    ----------
    fft_in : ndarray
        Fourier-domain data in the **fftshifted** convention (DC at centre),
        shaped ``(..., fy, fx)``.
    ifft_fn : callable
        Inverse FFT function that accepts an ndarray and returns an ndarray
        of the same shape.  Must match the ndarray backend (numpy, cupy, …)
        used when the data was created.
    input_shape : tuple of int
        Spatial shape ``(sy, sx)`` of the original (unpadded) hologram.
    fft_shape : tuple of int
        Shape of the FFT array ``(..., fy, fx)`` *before* any Fourier-space
        cropping was applied.
    padding : bool or int
        Whether boundary padding was applied during filtering.  If truthy,
        the spatial field is cropped back to ``input_shape`` after the iFFT.
    scale_to_filter : bool or float
        Whether (or by how much) the Fourier array was cropped to the filter
        support.  If truthy, the crop window and amplitude scale factor are
        recomputed from ``crop_radius`` and ``fft_shape``.
    crop_radius : int or None
        Radius in Fourier pixels of the crop window used when
        ``scale_to_filter`` is active.  Required when ``scale_to_filter`` is
        truthy; ignored otherwise.

    Returns
    -------
    field : ndarray
        Complex spatial field, shaped ``(..., sy, sx)``.
    """
    field = ifft_fn(xp.fft.ifftshift(fft_in, axes=(-2, -1)))

    if padding:
        sx, sy = input_shape
        if scale_to_filter:
            if crop_radius is None:
                raise ValueError(
                    "crop_radius is required when scale_to_filter is set")
            osize = fft_shape[-1]
            sx = int(xp.ceil(sx * 2 * crop_radius / osize))
            sy = int(xp.ceil(sy * 2 * crop_radius / osize))

        field = field[:, :sx, :sy]

        if scale_to_filter:
            osize = fft_shape[-1]
            field *= (2 * crop_radius / osize) ** 2

    return field


@dataclass(slots=True)
class FourierFieldData:
    """Fourier-domain field data returned by ``output_domain="fourier"``.

    Produced by :meth:`.OffAxisHologram.run_pipeline` (and the underlying
    :meth:`.FFTFilter.filter`) when ``output_domain="fourier"`` is requested.
    Instead of performing the inverse FFT, qpretrieve packages the filtered
    Fourier data together with the reconstruction metadata needed to recover
    the spatial field on demand.

    Call :meth:`finalize` to apply the inverse FFT and obtain the spatial
    field.  Downstream libraries such as `nrefocus` can consume this object
    directly via duck-typing (they detect :attr:`fft_used` and skip their own
    forward FFT, avoiding a redundant iFFT + FFT pair at the pipeline
    boundary).

    .. versionadded:: 0.7.0

    Attributes
    ----------
    fft_used : ndarray
        Filtered FFT data in the **fftshifted** convention (DC at the centre
        of the array).  Shape is ``(..., fy, fx)``, where ``fy`` and ``fx``
        are reduced relative to the full FFT when ``scale_to_filter`` is set.
    ifft_fn : callable
        Inverse FFT callable that matches the ndarray backend (numpy, cupy,
        …) used during filtering.
    input_shape : tuple of int
        Spatial shape ``(sy, sx)`` of the original unpadded hologram.
    fft_shape : tuple of int
        Shape of the full FFT array ``(..., fy0, fx0)`` before any
        Fourier-space cropping was applied.
    padding : bool or int
        Whether boundary padding was applied during filtering.
    scale_to_filter : bool or float
        Whether (or by what factor) the Fourier array was cropped to the
        filter support.  Controls the crop and amplitude rescaling in
        :func:`finalize_fourier_field`.
    crop_radius : int or None
        Radius in Fourier pixels of the crop window, set when
        ``scale_to_filter`` is active.
    output_domain : str
        Domain hint used by nrefocus for duck-typing.  Always ``"spatial"``
        — indicates that :meth:`finalize` should be called to convert to the
        spatial domain.
    """

    fft_used: xp.ndarray
    ifft_fn: Callable[[xp.ndarray], xp.ndarray]
    input_shape: tuple[int, int]
    fft_shape: tuple[int, int]
    padding: bool | int
    scale_to_filter: bool | float
    crop_radius: int | None = None
    output_domain: str = "spatial"
    _field: xp.ndarray | None = dataclass_field(
        default=None, init=False, repr=False)

    def finalize(self, propagated_fft: xp.ndarray | None = None) -> xp.ndarray:
        """Return the reconstructed spatial field.

        Applies ``ifftshift`` + inverse FFT to convert from the Fourier
        domain, then crops and rescales to match the original hologram
        dimensions (reversing any padding or Fourier-space cropping applied
        during filtering).

        The result is cached in :attr:`field` after the first call.

        Parameters
        ----------
        propagated_fft : ndarray or None
            Optional Fourier-domain array in the fftshifted convention,
            typically the output of a wave propagation step (e.g. from
            nrefocus).  When provided, this is inverse-transformed instead
            of :attr:`fft_used`, allowing the propagated field to be
            reconstructed with the same cropping/scaling as the original.
            If ``None``, :attr:`fft_used` is used.

        Returns
        -------
        field : ndarray
            Complex spatial field shaped ``(..., sy, sx)``, where ``sy`` and
            ``sx`` are the spatial dimensions of the original hologram.
        """
        fft_in = self.fft_used if propagated_fft is None else propagated_fft
        field = finalize_fourier_field(
            fft_in=fft_in,
            ifft_fn=self.ifft_fn,
            input_shape=self.input_shape,
            fft_shape=self.fft_shape,
            padding=self.padding,
            scale_to_filter=self.scale_to_filter,
            crop_radius=self.crop_radius,
        )
        self._field = field
        return field

    def subtract_background(self, bg: FourierFieldData) -> FourierFieldData:
        """Subtract a background field in Fourier space.

        Subtracts ``bg.fft_used`` from ``self.fft_used`` and returns a new
        :class:`FourierFieldData` with the result, without modifying ``self``.

        .. warning::

            This performs **complex-field subtraction**
            (``field_sample - field_bg``), which is *not* the same as the
            standard holographic background correction of subtracting phases
            and dividing amplitudes.  The standard correction is equivalent
            to **complex division** (``field_sample / field_bg``), which gives
            ``angle(corrected) = φ_sample − φ_bg`` and
            ``abs(corrected) = A_sample / A_bg``.  Complex subtraction gives
            ``angle(field_sample − field_bg) ≠ φ_sample − φ_bg`` in general.

            Use this method when complex-field subtraction is physically
            appropriate for your application.  For the standard phase-only
            background correction, reconstruct both holograms to the spatial
            domain (e.g. via :attr:`~FourierFieldData.field`) and subtract
            the phases directly.

        Because the inverse FFT is linear,
        ``ifft(fft_sample − fft_bg) = field_sample − field_bg``, so the
        subtraction here is equivalent to the spatial-domain complex
        difference.  Performing it in Fourier space avoids a redundant
        iFFT + FFT pair when the corrected data is passed to a downstream
        library such as `nrefocus` for wave propagation.

        Both objects must have been produced by pipelines with identical
        filter settings (same ``filter_name``, ``filter_size``,
        ``freq_pos``, and ``scale_to_filter``), so that ``fft_used`` has
        the same shape and physical meaning in both cases.

        Parameters
        ----------
        bg : FourierFieldData
            Fourier-domain data for the background hologram, obtained via a
            pipeline run with the same filter configuration as ``self``.

        Returns
        -------
        corrected : FourierFieldData
            New instance with ``fft_used = self.fft_used - bg.fft_used``
            and all other reconstruction metadata copied from ``self``.

        Raises
        ------
        ValueError
            If ``bg.fft_used.shape`` does not match ``self.fft_used.shape``.
        """
        if bg.fft_used.shape != self.fft_used.shape:
            raise ValueError(
                f"Shape mismatch: sample fft_used has shape "
                f"{self.fft_used.shape}, but background fft_used has shape "
                f"{bg.fft_used.shape}. Both must come from pipelines with "
                f"identical filter settings.")
        return dataclasses.replace(self, fft_used=self.fft_used - bg.fft_used)

    @property
    def field(self) -> xp.ndarray:
        """Cached spatial field.

        Returns the result of the last :meth:`finalize` call.  If
        :meth:`finalize` has not been called yet, it is invoked with no
        arguments (i.e. using :attr:`fft_used`).
        """
        if self._field is None:
            return self.finalize()
        return self._field
