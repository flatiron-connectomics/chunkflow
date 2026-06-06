

from itertools import product

import numpy as np
import tensorstore as ts
from cloudfiles import CloudFiles
from cloudvolume import CloudVolume
from cloudvolume.exceptions import EmptyVolumeException
from cloudvolume.lib import Vec, yellow

from chunkflow.lib.cartesian_coordinate import BoundingBox, Cartesian
from chunkflow.chunk.validate import validate_by_template_matching
from tinybrain import downsample_with_averaging
from chunkflow.chunk import Chunk
from .base import OperatorBase
from .save_precomputed import _tensorstore_kvstore_from_url


class LoadPrecomputedOperator(OperatorBase):
    def __init__(self,
                 volume_path: str,
                 mip: int = 0,
                 use_tensorstore: bool = False,
                 fill_missing: bool = False,
                 raise_missing: bool = True,
                 validate_mip: int = None,
                 blackout_sections: bool = None,
                 use_https: bool = False,
                 dry_run: bool = False,
                 verbose: bool = False,
                 name: str = 'cutout',
                 green_threads: bool = False):
        super().__init__(name=name)

        if use_tensorstore:
            if validate_mip is not None:
                raise ValueError(
                    "validate_mip is not supported with use_tensorstore=True")
            if use_https:
                raise ValueError(
                    "use_https is not supported with use_tensorstore=True")
            if green_threads:
                raise ValueError(
                    "green_threads=True is not supported with use_tensorstore=True")

        self.mip = mip
        self.fill_missing = fill_missing
        self.raise_missing = raise_missing
        self.validate_mip = validate_mip
        self.blackout_sections = blackout_sections
        self.dry_run = dry_run
        self.green_threads = green_threads
        self._tensorstore = use_tensorstore

        if self._tensorstore:
            url = volume_path
            if url.startswith('precomputed://'):
                url = url[len('precomputed://'):]
            if '://' not in url:
                url = 'file://' + url
            self.volume_path = url
            spec = {
                'driver': 'neuroglancer_precomputed',
                'kvstore': _tensorstore_kvstore_from_url(url),
                'scale_index': mip,
            }
            # NOTE: the neuroglancer_precomputed driver doesn't expose a
            # fill_value slot in its spec/schema, and tensorstore has no
            # `fill_value` kwarg on `open` or `read`. Missing-chunk handling
            # for `fill_missing` happens manually in `_read_tensorstore`.
            self.volume = ts.open(spec, read=True, write=False).result()
            spec_json = self.volume.spec().to_json()
            chunk_layout = self.volume.chunk_layout
            self._dtype = np.dtype(self.volume.dtype.numpy_dtype)
            self._resolution = tuple(spec_json['scale_metadata']['resolution'])
            self._layer_type = spec_json['multiscale_metadata']['type']
            self._chunk_shape_xyz = tuple(
                int(chunk_layout.read_chunk.shape[d]) for d in range(3))
            self._grid_origin_xyz = tuple(
                int(chunk_layout.grid_origin[d]) for d in range(3))
            self.vol = None
        else:
            if '://' not in volume_path:
                volume_path = 'file://' + volume_path
            self.volume_path = volume_path
            self.vol = CloudVolume(
                self.volume_path,
                bounded=False,
                fill_missing=self.fill_missing,
                progress=verbose,
                mip=self.mip,
                cache=False,
                use_https=use_https,
                green_threads=self.green_threads)
            self.volume = self.vol
            self._dtype = self.vol.dtype
            self._resolution = self.vol.resolution
            self._layer_type = self.vol.layer_type

        if blackout_sections:
            stor = CloudFiles(self.volume_path)
            self.blackout_section_ids = stor.get_json(
                'blackout_section_ids.json')['section_ids']

    def mip_bounds(self, mip: int):
        """Return (minpt, maxpt) Vec triples in xyz order at the given mip level."""
        if self._tensorstore:
            if mip != self.mip:
                raise NotImplementedError(
                    "mip_bounds at a different mip is not yet supported "
                    "with use_tensorstore=True")
            d = self.volume.domain
            minpt = Vec(int(d[0].inclusive_min), int(d[1].inclusive_min), int(d[2].inclusive_min))
            maxpt = Vec(int(d[0].exclusive_max), int(d[1].exclusive_max), int(d[2].exclusive_max))
        else:
            bbox = self.vol.mip_bounds(mip)
            minpt = Vec(*bbox.minpt)
            maxpt = Vec(*bbox.maxpt)
        return minpt, maxpt

    def __call__(self, bbox: BoundingBox):
        # if we do not clone this bounding box,
        # the bounding box in task will be modified!
        assert isinstance(bbox, BoundingBox)
        chunk_slices = bbox.slices

        if self.dry_run:
            # input_bbox = BoundingBox.from_slices(chunk_slices)
            # we can not use pattern=zero since it might got skipped by
            # the operator of skip-all-zero
            return Chunk.from_bbox(
                bbox,
                pattern='random',
                dtype=self._dtype,
                voxel_size=Cartesian.from_collection(self._resolution[::-1]),
            )

        print(f'cutout ZYX_{chunk_slices} from {self.volume_path}')

        if self._tensorstore:
            chunk = self._read_tensorstore(bbox)
            if chunk is None:
                return None
        else:
            # always reverse the indexes since cloudvolume use x,y,z indexing
            try:
                chunk = self.vol[chunk_slices[::-1]]
            except EmptyVolumeException:
                if not self.raise_missing:
                    print(yellow(f"Empty chunk in {self.volume_path} for slices {chunk_slices}, returning None"))
                    return None
                else:
                    raise
            chunk = np.asarray(chunk)
            # the cutout is fortran ordered, so need to transpose and make it C order
            chunk = chunk.transpose()

        # if the channel number is 1, squeeze it as 3d array
        # this should not be neccessary
        # TODO: remove this step and use 4D array all over this package.
        # always use 4D array will simplify some operations
        if chunk.shape[0] == 1:
            chunk = np.squeeze(chunk, axis=0)

        chunk = Chunk(
            chunk,
            voxel_offset=bbox.start,
            voxel_size=Cartesian.from_collection(self._resolution[::-1]),
            layer_type=self._layer_type)

        if self.blackout_sections:
            chunk = self._blackout_sections(chunk)

        if self.validate_mip:
            self._validate_chunk(chunk)
        return chunk

    def _read_tensorstore(self, bbox: BoundingBox):
        """Read a chunk via tensorstore, returning a czyx array (matching the
        layout of `CloudVolume[...].transpose()`), or None to signal an empty
        read with raise_missing=False.

        Mirrors CloudVolume's ``bounded=False`` semantics: if the requested
        region extends past the volume's domain, we crop and (when
        fill_missing=True) zero-pad back to the requested shape.
        """
        xyz_slices = bbox.slices[::-1]
        domain = self.volume.domain

        crop_slices = []
        pad_before_xyz = []
        pad_after_xyz = []
        for dim, sl in enumerate(xyz_slices):
            lo = max(sl.start, int(domain[dim].inclusive_min))
            hi = min(sl.stop, int(domain[dim].exclusive_max))
            crop_slices.append(slice(lo, hi))
            pad_before_xyz.append(lo - sl.start)
            pad_after_xyz.append(sl.stop - hi)

        overlap_empty = any(s.stop <= s.start for s in crop_slices)
        needs_padding = overlap_empty or any(
            pb or pa for pb, pa in zip(pad_before_xyz, pad_after_xyz))

        if needs_padding and not self.fill_missing:
            if self.raise_missing:
                raise EmptyVolumeException(
                    f"requested bbox {bbox} extends past volume domain in "
                    f"{self.volume_path} and fill_missing=False")
            print(yellow(
                f"Out-of-bounds chunk in {self.volume_path} for slices "
                f"{bbox.slices}, returning None"))
            return None

        num_channels = int(domain[-1].exclusive_max - domain[-1].inclusive_min)
        if overlap_empty:
            xyz_size = [s.stop - s.start for s in xyz_slices]
            arr = np.zeros((*xyz_size, num_channels), dtype=self._dtype)
        else:
            try:
                # TensorStore dataset.read() reads in C order by default, despite
                # precomputed format encoding in Fortran order. Make it consistent
                # by specifying Fortran-order read.
                arr = self.volume[tuple(crop_slices)].read(order='F').result()
            except Exception as e:
                # neuroglancer_precomputed doesn't support fill_value, so a
                # missing chunk within the domain raises here. Mirror
                # CloudVolume's fill_missing / raise_missing semantics.
                if self.fill_missing:
                    # Fall back to per-chunk reads so present chunks aren't
                    # masked by zeros for the missing ones.
                    arr = self._read_per_chunk(tuple(crop_slices), num_channels)
                elif self.raise_missing:
                    raise EmptyVolumeException(
                        f"missing chunk in {self.volume_path} for slices "
                        f"{bbox.slices}") from e
                else:
                    print(yellow(
                        f"Missing chunk in {self.volume_path} for slices "
                        f"{bbox.slices}, returning None: {e}"))
                    return None
            if needs_padding:
                pad_width = list(zip(pad_before_xyz, pad_after_xyz)) + [(0, 0)]
                arr = np.pad(arr, pad_width, mode='constant', constant_values=0)

        # xyzc → czyx to match CloudVolume's transposed output.
        return np.asarray(arr).transpose()

    def _read_per_chunk(self, crop_slices, num_channels):
        """Read the cropped xyz region one storage chunk at a time, leaving
        zeros in place of any chunk whose read fails. Used as a fallback when
        a bulk read raises and fill_missing=True, so that chunks already
        written on disk are preserved rather than wiped to zeros along with
        the missing ones.
        """
        cs_xyz = self._chunk_shape_xyz
        org_xyz = self._grid_origin_xyz

        crop_size_xyz = [s.stop - s.start for s in crop_slices]
        arr = np.zeros((*crop_size_xyz, num_channels), dtype=self._dtype)

        chunk_index_ranges = []
        for d, sl in enumerate(crop_slices):
            first = (sl.start - org_xyz[d]) // cs_xyz[d]
            last = (sl.stop - 1 - org_xyz[d]) // cs_xyz[d]
            chunk_index_ranges.append(range(first, last + 1))

        for idx in product(*chunk_index_ranges):
            read_slices = []
            out_slices = []
            for d in range(3):
                sl = crop_slices[d]
                chunk_start = org_xyz[d] + idx[d] * cs_xyz[d]
                chunk_stop = chunk_start + cs_xyz[d]
                read_lo = max(chunk_start, sl.start)
                read_hi = min(chunk_stop, sl.stop)
                read_slices.append(slice(read_lo, read_hi))
                out_slices.append(slice(read_lo - sl.start, read_hi - sl.start))
            try:
                chunk_arr = self.volume[tuple(read_slices)].read(order='F').result()
            except Exception:
                # leave the corresponding region of arr as zeros
                continue
            arr[tuple(out_slices)] = chunk_arr

        return arr

    def _blackout_sections(self, chunk):
        """
        make some sections black.
        this was normally used for the section with bad alignment.
        The ConvNet was supposed to handle them better with black image.

        TODO: make this function as a separate operator
        """
        # current code only works with 3d image
        assert chunk.ndim == 3, "current code assumes that the chunk is 3D image."
        for z in self.blackout_section_ids:
            z0 = z - chunk.voxel_offset[0]
            if z0 >= 0 and z0 < chunk.shape[0]:
                chunk[z0, :, :] = 0
        return chunk

    def _validate_chunk(self, chunk):
        """
        check that all the input voxels was downloaded without black region
        We have found some black regions in previous inference run,
        so hopefully this will solve the problem.
        """
        if chunk.ndim == 4 and chunk.shape[0] > 1:
            chunk = chunk[0, :, :, :]

        validate_vol = CloudVolume(self.volume_path,
                                   bounded=False,
                                   fill_missing=self.fill_missing,
                                   progress=False,
                                   mip=self.validate_mip,
                                   cache=False,
                                   green_threads=self.green_threads)


        chunk_mip = self.mip
        print('validate chunk in mip {}'.format(self.validate_mip))
        assert self.validate_mip >= chunk_mip
        # only use the region corresponds to higher mip level
        # clamp the surrounding regions in XY plane
        # this assumes that the input dataset was downsampled starting from the
        # beginning offset in the info file
        voxel_offset = chunk.voxel_offset

        # factor3 follows xyz order in CloudVolume
        factor3 = np.array([
            2**(self.validate_mip - chunk_mip), 2
            **(self.validate_mip - chunk_mip), 1
        ],
                           dtype=np.int32)
        clamped_offset = tuple(go + f - (go - vo) % f for go, vo, f in zip(
            voxel_offset[::-1], self.vol.voxel_offset, factor3))
        clamped_stop = tuple(
            go + s - (go + s - vo) % f
            for go, s, vo, f in zip(voxel_offset[::-1], chunk.shape[::-1],
                                    self.vol.voxel_offset, factor3))
        clamped_slices = tuple(
            slice(o, s) for o, s in zip(clamped_offset, clamped_stop))
        clamped_bbox = BoundingBox.from_slices(clamped_slices)
        clamped_input = chunk.cutout(clamped_slices[::-1])
        # transform to xyz order
        clamped_input = np.transpose(clamped_input)
        # get the corresponding bounding box for validation
        validate_bbox = self.vol.bbox_to_mip(clamped_bbox,
                                             mip=chunk_mip,
                                             to_mip=self.validate_mip)
        #validate_bbox = clamped_bbox // factor3

        # downsample the input using avaraging
        # keep the z as it is since the mip only applies to xy plane
        # recursivly downsample the input
        # if we do it directly, the downsampled input will not be the same with the recursive one
        # because of the rounding error of integer division
        for _ in range(self.validate_mip - chunk_mip):
            clamped_input = downsample_with_averaging(clamped_input, (2, 2, 1))

        # validation by template matching
        assert validate_by_template_matching(clamped_input)

        validate_input = validate_vol[validate_bbox.slices]
        if validate_input.shape[3] == 1:
            validate_input = np.squeeze(validate_input, axis=3)

        # use the validate input to check the downloaded input
        assert np.all(validate_input == clamped_input)
