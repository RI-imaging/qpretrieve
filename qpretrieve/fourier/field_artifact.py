from __future__ import annotations

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
    """Convert a propagated Fourier field back to spatial domain."""
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
class FourierFieldArtifact:
    """Container for qpretrieve Fourier-domain field data.
    Users do not need to worry about how this works, it is for internally
    dealing with direct qpretrieve-to-nrefocus fourier domain pipelines.

    This object carries the filtered Fourier data together with the
    metadata needed to reconstruct the spatial field exactly as
    qpretrieve would have produced it.
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
        """Return the correctly cropped/scaled spatial field.

        Parameters
        ----------
        propagated_fft
            Optional Fourier-domain array after propagation. If omitted,
            the stored ``fft_used`` is inverse transformed.
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

    @property
    def field(self) -> xp.ndarray:
        """Cached finalized field."""
        if self._field is None:
            return self.finalize()
        return self._field
