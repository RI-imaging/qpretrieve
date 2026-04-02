from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Type

from .._ndarray_backend import xp
from ..fourier import get_best_interface, get_available_interfaces
from ..fourier.base import FFTFilter
from ..data_array_layout import (
    convert_data_to_3d_array_layout, convert_3d_data_to_array_layout
)
from ..roi import (
    boxes_from_mask,
    normalize_boxes,
    merge_boxes,
)


class BadFFTFilterError(ValueError):
    pass


class BaseInterferogram(ABC):
    default_pipeline_kws = {
        "filter_name": "disk",
        "filter_size": 1 / 3,
        "filter_size_interpretation": "sideband distance",
        "scale_to_filter": False,
        "sideband_freq": None,
        "invert_phase": False,
    }

    def __init__(self, data: xp.ndarray,
                 fft_interface: str | Type[FFTFilter] | None = "auto",
                 subtract_mean=True, padding: int | bool = 2, copy=True,
                 dtype_conversion=None,
                 **pipeline_kws) -> None:
        """
        Parameters
        ----------
        data
            The experimental input real-valued image. Allowed input shapes are:
              - 2d (y, x)
              - 3d (z, y, x)
              - rgb (y, x, 3) or rgba (y, x, 4)
        fft_interface
            A Fourier transform interface.
            See :func:`qpretrieve.fourier.get_available_interfaces`
            to get a list of implemented interfaces.
            Default is "auto", which will use
            :func:`qpretrieve.fourier.get_best_interface`. This is in line
            with old behaviour. See Notes for more details.
        subtract_mean: bool
            If True, remove the mean of the hologram before performing
            the Fourier transform. This setting is recommended as it
            can reduce artifacts from frequencies around the central
            band.
        padding
            Boundary padding with zeros; This value determines how
            large the padding region should be. If set to zero, then
            no padding is performed. If set to a positive integer, the
            size is computed to the next power of two (square image)::

                2 ** xp.ceil(xp.log(padding * max(data.shape) / xp.log(2)))
        copy: bool
            Whether to create a copy of the input data.
        dtype_conversion
            The dtype that should be used to convert the input data before
            preprocessing occurs. This defaults to ``complex`` if the input
            data is complex, otherwise to ``float`` (64-bit) for all
            other situations. For some use-cases, for example when
            using a GPU, you might want to be more specific
            e.g., ``cp.float32``.
        pipeline_kws:
            Any additional keyword arguments for :func:`run_pipeline`
            as defined in :const:`default_pipeline_kws`.

        Notes
        -----
        For `fft_interface`, if you do not have the relevant package installed,
        then an error will be raised. For example, setting
        `fft_interface=FFTFilterPyFFTW` will fail if you do not have pyfftw
        installed.

        """
        # keep basic statistics from the raw input before preprocessing
        try:
            self._input_mean = float(xp.asarray(data).mean())
        except Exception:  # pragma: no cover - defensive
            self._input_mean = 0.0
        if fft_interface is None:
            raise BadFFTFilterError(
                "`fft_interface` is set to None or is unavailable."
                "This is likely because you "
                "are trying to use `FFTFilterPyFFTW` or `FFTFilterCupy`. "
                "To use `FFTFilterPyFFTW`, install 'pyfftw'. "
                "To use `FFTFilterCupy`, install 'cupy-cuda12x' or "
                "'cupy-cuda11x', depending on your CUDA version. "
                "If you want qpretrieve to find the best FFT interface "
                "for you, set `fft_interface='auto'`.")
        if fft_interface == 'auto':
            self.ff_iface = get_best_interface()
        else:
            if fft_interface in get_available_interfaces():
                self.ff_iface = fft_interface
            else:
                raise BadFFTFilterError(
                    f"User-chosen FFT Interface '{fft_interface}' is not "
                    f"available. The available interfaces are: "
                    f"{get_available_interfaces()}.\n"
                    f"You can use `fft_interface='auto'` to get the best "
                    f"available interface.")

        # figure out what type of data we have, change it to 3d-stack
        data, self.orig_array_layout = convert_data_to_3d_array_layout(data)

        #: qpretrieve Fourier transform interface class
        self.fft = self.ff_iface(data=data,
                                 subtract_mean=subtract_mean,
                                 padding=padding,
                                 copy=copy,
                                 dtype_conversion=dtype_conversion)
        #: originally computed Fourier transform
        self.fft_origin = self.fft.fft_origin
        #: filtered Fourier data from last run of `run_pipeline`
        self.fft_filtered = self.fft.fft_filtered
        #: pipeline keyword arguments
        self.pipeline_kws = pipeline_kws
        # Subclasses con override the properties phase, amplitude, and field
        self._field = None
        self._phase = None
        self._amplitude = None

    def get_data_with_input_layout(self, data: xp.ndarray | str) -> xp.ndarray:
        """Convert `data` to the original input array layout.


        Parameters
        ----------
        data
            Either an array (xp.ndarray) or name (str) of the relevant `data`.

        Returns
        -------
        data_out : xp.ndarray
            array in the original input array layout

        Notes
        -----
        If `data` is the RGBA array layout, then the alpha (A) channel will be
        an array of ones.

        """
        if isinstance(data, str):
            if data == "fft":
                data = "fft_filtered"
                warnings.warn(
                    "You have asked for 'fft' which is a class. "
                    "Returning 'fft_filtered'. "
                    "Alternatively you could use 'fft_origin'.")
            data = getattr(self, data)
        return convert_3d_data_to_array_layout(data, self.orig_array_layout)

    @property
    def phase(self) -> xp.ndarray:
        """Retrieved phase information"""
        if self._phase is None:
            self.run_pipeline()
        return self._phase

    @property
    def amplitude(self) -> xp.ndarray:
        """Retrieved amplitude information"""
        if self._amplitude is None:
            self.run_pipeline()
        return self._amplitude

    @property
    def field(self) -> xp.ndarray:
        """Retrieved amplitude information"""
        if self._field is None:
            self.run_pipeline()
        return self._field

    def compute_filter_size(
            self,
            filter_size: float,
            filter_size_interpretation: str,
            sideband_freq: tuple[float, float] = None) -> float:
        """Compute the actual filter size in Fourier space"""
        if filter_size_interpretation == "frequency":
            # convert frequency to frequency index
            # We always have padded Fourier data with sizes of order 2.
            fsize = filter_size
        elif filter_size_interpretation == "sideband distance":
            if sideband_freq is None:
                raise ValueError("`sideband_freq` must be set!")
            # filter size based on distance b/w central band and sideband
            if filter_size <= 0 or filter_size >= 1:
                raise ValueError("For sideband distance interpretation, "
                                 "`filter_size` must be between 0 and 1; "
                                 f"got '{filter_size}'!")
            fsize = xp.sqrt(xp.sum(xp.array(sideband_freq) ** 2)) * filter_size
        elif filter_size_interpretation == "frequency index":
            # filter size given in Fourier index (number of Fourier pixels)
            # The user probably does not know that we are padding in
            # Fourier space, so we use the unpadded size and translate it.
            if filter_size <= 0 or filter_size >= self.fft.shape[-2] / 2:
                raise ValueError("For frequency index interpretation, "
                                 + "`filter_size` must be between 0 and "
                                 + f"{self.fft.shape[-2] / 2}, got "
                                 + f"'{filter_size}'!")
            # convert to frequencies (compatible with fx and fy)
            fsize = filter_size / self.fft.shape[-2]
        else:
            raise ValueError("Invalid value for `filter_size_interpretation`: "
                             + f"'{filter_size_interpretation}'")
        return fsize

    def process_like(self, other):
        """Process this dataset in the same way as `other` dataset"""
        assert other.__class__.__name__ == self.__class__.__name__
        self.pipeline_kws.clear()
        if not other.pipeline_kws:
            # run default pipeline
            other.run_pipeline()
        self.run_pipeline(**other.pipeline_kws)

    def get_pipeline_kw(self, key):
        """Current pipeline keyword argument with fallback to defaults"""
        return self.pipeline_kws.get(key, self.default_pipeline_kws[key])

    def run_pipeline_rois(
            self,
            roi_boxes: list | tuple | None = None,
            roi_mask: xp.ndarray | None = None,
            background_fill: float | complex | str = "mean_input",
            box_padding: int = 0,
            stitch: bool = True,
            **pipeline_kws):
        """
        Run the pipeline only on regions of interest and stitch
        results back into a full-size field, or return ROI field(s)
        directly when stitching is disabled.

        Parameters
        ----------
        roi_boxes:
            Iterable of bounding boxes ``(y0, y1, x0, x1)`` or slice tuples.
            Coordinates follow NumPy's half-open convention ``[start, end)``.
        roi_mask:
            Boolean mask matching the input image shape. A single bounding
            box is inferred around the non-zero region.
        background_fill:
            Fill value outside all ROIs. Use ``"mean_input"`` (default) to
            fill with the mean of the original input data, ``"zero"`` for
            zeros, or pass a numeric value.
        box_padding:
            Optional padding (in pixels) added around each ROI box.
        stitch:
            If True (default), stitch ROI results back into a full-size
            field. If False, ROI results are returned without stitching.
            When ``stitch=False``, exactly one ROI box is required.

        Returns
        -------
        field : xp.ndarray
            Complex field. If ``stitch=True``, shape matches the input.
            If ``stitch=False``, shape matches the ROI box.
        """
        shape = self.fft.origin.shape[-2:]
        boxes = []
        if roi_mask is not None:
            boxes.extend(boxes_from_mask(roi_mask, padding=box_padding,
                                         shape=shape))
        if roi_boxes:
            boxes.extend(normalize_boxes(roi_boxes, padding=box_padding,
                                         shape=shape))
        boxes = merge_boxes(boxes)
        if not boxes:
            return self.run_pipeline(**pipeline_kws)

        sb_freq = pipeline_kws.get(
            "sideband_freq", self.pipeline_kws.get("sideband_freq"))
        if sb_freq is not None:
            pipeline_kws = dict(pipeline_kws, sideband_freq=sb_freq)

        if not stitch:
            if len(boxes) != 1:
                raise ValueError("stitch=False requires exactly one ROI box.")
            y0, y1, x0, x1 = boxes[0]
            roi_slice = (slice(None), slice(y0, y1), slice(x0, x1))
            roi_data = self.fft.origin[roi_slice]
            roi_obj = self.__class__(
                data=roi_data,
                fft_interface=self.ff_iface,
                subtract_mean=self.fft.subtract_mean,
                padding=self.fft.padding,
                copy=True,
                dtype_conversion=self.fft.dtype_conversion,
            )
            roi_field = roi_obj.run_pipeline(**pipeline_kws)
            self._field = roi_field
            self._phase = xp.angle(roi_field)
            self._amplitude = xp.abs(roi_field)
            self.pipeline_kws.update(pipeline_kws)
            return self._field

        if isinstance(background_fill, str):
            if background_fill == "mean_input":
                bg_val = complex(self._input_mean)
            elif background_fill == "zero":
                bg_val = 0.0
            else:
                raise ValueError(f"Unknown background_fill '{background_fill}'")
        else:
            bg_val = complex(background_fill)

        field_full = xp.full(self.fft.origin.shape, bg_val,
                             dtype=self.fft_origin.dtype)

        for y0, y1, x0, x1 in boxes:
            roi_slice = (slice(None), slice(y0, y1), slice(x0, x1))
            roi_data = self.fft.origin[roi_slice]
            roi_obj = self.__class__(
                data=roi_data,
                fft_interface=self.ff_iface,
                subtract_mean=self.fft.subtract_mean,
                padding=self.fft.padding,
                copy=True,
                dtype_conversion=self.fft.dtype_conversion,
            )
            roi_field = roi_obj.run_pipeline(**pipeline_kws)
            field_full[roi_slice] = roi_field

        self._field = field_full
        self._phase = xp.angle(field_full)
        self._amplitude = xp.abs(field_full)
        self.pipeline_kws.update(pipeline_kws)
        return self._field

    @abstractmethod
    def run_pipeline(self, **pipeline_kws):
        """Perform pipeline analysis, populating `self.field`

        Parameters
        ----------
        filter_name: str
            specifies the filter to use, see
            :func:`qpretrieve.filter.get_filter_array`.
        filter_size: float
            Size of the filter in Fourier space. The interpretation
            of this value depends on `filter_size_interpretation`.
        filter_size_interpretation: str
            If set to "sideband distance", the filter size is interpreted
            as the relative distance between central band and sideband
            (this is the default). If set to "frequency index", the filter
            size is interpreted as a Fourier frequency index ("pixel size")
            and must be between 0 and `max(hologram.shape)/2`.
        scale_to_filter: bool or float
            Crop the image in Fourier space after applying the filter,
            effectively removing surplus (zero-padding) data and
            increasing the pixel size in the output image. If True is
            given, then the cropped area is defined by the filter size,
            if a float is given, the cropped area is defined by the
            filter size multiplied by `scale_to_filter`. You can safely
            set this to True for filters with a binary support. For
            filters such as "smooth square" or "gauss" (filter is not
            a boolean array but a floating-point array), the higher you
            set `scale_to_filter`, the more information will be included
            in the scaled image.
        sideband_freq: tuple of floats
            Frequency coordinates of the sideband to use. By default,
            a heuristic search for the sideband is done.
        invert_phase: bool
            Invert the phase data.
        """
