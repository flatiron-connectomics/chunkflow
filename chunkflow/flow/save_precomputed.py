import json
import time
import os

import numpy as np
import tensorstore as ts

from cloudvolume import CloudVolume
from cloudvolume.lib import Vec, yellow
from cloudfiles import CloudFiles

from chunkflow.lib.cartesian_coordinate import BoundingBox
from chunkflow.lib.igneous.tasks import downsample_and_upload
from chunkflow.chunk import Chunk

from .base import OperatorBase
#from .downsample_upload import DownsampleUploadOperator


_TENSORSTORE_KVSTORE_SCHEMES = ('file', 's3')


def _tensorstore_kvstore_from_url(url: str) -> dict:
    """Translate a CloudVolume-style URL into a tensorstore kvstore spec.

    The optional `precomputed://` prefix should already be stripped.
    """
    scheme, sep, rest = url.partition('://')
    if not sep:
        raise ValueError(f"expected '<scheme>://...', got {url!r}")
    if scheme == 'file':
        kvstore = {'driver': 'file', 'path': rest}
    elif scheme == 's3':
        bucket, _, path = rest.partition('/')
        kvstore = {'driver': 's3', 'bucket': bucket, 'path': path}
    else:
        raise ValueError(
            f"unsupported tensorstore kvstore scheme {scheme!r}; "
            f"supported: {list(_TENSORSTORE_KVSTORE_SCHEMES)}")
    return kvstore


def _tensorstore_write_chunk(volume, xyz_slices, arr, autocrop,
                             skip_all_zero=True):
    """Write `arr` (in xyz[c] order) to a tensorstore volume at `xyz_slices`.

    When ``autocrop=True``, the requested slices are first intersected with
    the volume's domain and the array sliced to match (raises if there is no
    overlap). When False, the write is passed through and tensorstore raises
    on out-of-bounds indexing.

    If `arr` has one fewer dim than the volume (e.g. 3D xyz against a 4D
    xyzc volume), a trailing axis is added; this is only valid for
    single-channel volumes.

    When ``skip_all_zero=True`` (the default, matching CloudVolume's
    ``delete_black_uploads=True``), an array that is entirely zero after
    cropping is not written at all. NOTE: this is coarser than CloudVolume,
    which makes the decision per storage chunk — here we skip only when the
    *whole* post-crop array is zero.
    """
    if autocrop:
        domain = volume.domain
        cropped_slices = []
        arr_indexers = []
        for dim, sl in enumerate(xyz_slices):
            lo = max(sl.start, int(domain[dim].inclusive_min))
            hi = min(sl.stop, int(domain[dim].exclusive_max))
            if hi <= lo:
                raise ValueError(
                    f"chunk slice {sl} on dim {dim} does not overlap "
                    f"volume domain [{int(domain[dim].inclusive_min)}, "
                    f"{int(domain[dim].exclusive_max)})")
            cropped_slices.append(slice(lo, hi))
            arr_indexers.append(slice(lo - sl.start, hi - sl.start))
        while len(arr_indexers) < arr.ndim:
            arr_indexers.append(slice(None))
        xyz_slices = tuple(cropped_slices)
        arr = arr[tuple(arr_indexers)]
    if arr.ndim == volume.ndim - 1:
        channel_dim = volume.domain[-1]
        channel_size = int(channel_dim.exclusive_max - channel_dim.inclusive_min)
        if channel_size != 1:
            raise ValueError(
                f"chunk has {arr.ndim} dims but volume has {volume.ndim} "
                f"dims with channel size {channel_size}; cannot broadcast")
        arr = arr[..., np.newaxis]
    if skip_all_zero and not arr.any():
        return
    volume[xyz_slices].write(arr).result()


class SavePrecomputedOperator(OperatorBase):
    def __init__(self,
                 volume_path: str,
                 mip: int,
                 use_tensorstore: bool = False,
                 upload_log: bool = True,
                 create_thumbnail: bool = False,
                 fill_missing: bool = False,
                 autocrop: bool = True,
                 invert: bool = False,
                 green_threads: bool = False,
                 parallel: int = 1,
                 name: str = 'save-precomputed',
                 non_aligned_writes=False,
    ):
        super().__init__(name=name)

        if use_tensorstore:
            if create_thumbnail:
                raise ValueError(
                    "create_thumbnail=True is not supported with use_tensorstore=True")
            if non_aligned_writes:
                raise ValueError(
                    "non_aligned_writes=True is not supported with use_tensorstore=True")
            if parallel > 1:
                raise ValueError(
                    "parallel>1 is not supported with use_tensorstore=True")
            if green_threads:
                raise ValueError(
                    "green_threads=True is not supported with use_tensorstore=True")

        self.upload_log = upload_log
        self.create_thumbnail = create_thumbnail
        self.mip = mip
        self.invert = invert
        self.autocrop = autocrop
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
            # fill_value slot in its spec/schema, so `fill_missing` has no
            # effect on the save path. CloudVolume's `fill_missing` only ever
            # mattered for reads anyway.
            self.volume = ts.open(spec, read=True, write=True).result()
            self._volume_dtype = np.dtype(self.volume.dtype.numpy_dtype)
        else:
            if '://' not in volume_path:
                volume_path = 'file://' + volume_path
            self.volume_path = volume_path
            self.volume = CloudVolume(
                self.volume_path,
                fill_missing=fill_missing,
                bounded=False,
                autocrop=autocrop,
                mip=self.mip,
                cache=False,
                green_threads=green_threads,
                delete_black_uploads=True,
                parallel=parallel,
                progress=True,
                non_aligned_writes=non_aligned_writes,
            )
            self._volume_dtype = self.volume.dtype

        if upload_log:
            log_path = os.path.join(self.volume_path, 'log')
            self.log_storage = CloudFiles(log_path)

    def create_chunk_with_zeros(self, bbox, num_channels, dtype):
        """Create a fake all zero chunk.
        this is used in skip some operation based on mask."""
        shape = (num_channels, *bbox.size3())
        arr = np.zeros(shape, dtype=dtype)
        chunk = Chunk(arr, voxel_offset=(0, *bbox.minpt))
        return chunk

    def __call__(self, chunk: Chunk, log=None):
        assert isinstance(chunk, Chunk)
        print(f'save chunk {chunk.bbox.string} to {self.volume_path}')

        start = time.time()
        arr = np.transpose(chunk.array)
        arr = self._auto_convert_dtype(arr, self._volume_dtype)

        if self.invert:
            max_val = np.iinfo(arr.dtype).max
            print(yellow(f'inverting chunk data using max value {max_val}'))
            arr = max_val - arr

        if self._tensorstore:
            _tensorstore_write_chunk(
                self.volume, chunk.slices[::-1], arr, autocrop=self.autocrop)
        else:
            # transpose czyx to xyzc order
            self.volume[chunk.slices[::-1]] = arr

        if self.create_thumbnail:
            self._create_thumbnail(chunk)

        if log:
            # log save operation time
            log['timer'][self.name] = time.time() - start

        if self.upload_log:
            self._upload_log(log, chunk.bbox)

    def _auto_convert_dtype(self, chunk, vol_dtype):
        """convert the data type to fit volume datatype"""
        if np.issubdtype(vol_dtype, np.floating) and np.issubdtype(chunk.dtype, np.uint8):
            chunk = chunk.astype(vol_dtype)
            chunk /= 255.
        elif np.issubdtype(vol_dtype, np.uint8) and np.issubdtype(chunk.dtype, np.floating):
            assert chunk.max() <= 1.
            chunk *= 255

        if vol_dtype != chunk.dtype:
            print(yellow(f'converting chunk data type {chunk.dtype} ' +
                         f'to volume data type: {vol_dtype}'))
            float_chunk = chunk.astype(np.float64)
            chunk = float_chunk / np.iinfo(chunk.dtype).max * np.iinfo(vol_dtype).max
            chunk = chunk.astype(vol_dtype)
        return chunk

    def _create_thumbnail(self, chunk):
        print('creating thumbnail...')

        thumbnail_volume_path = os.path.join(self.volume_path, 'thumbnail')
        thumbnail_volume = CloudVolume(
            thumbnail_volume_path,
            compress='gzip',
            fill_missing=True,
            bounded=False,
            autocrop=True,
            mip=self.mip,
            cache=False,
            green_threads=True,
            delete_black_uploads=True,
            progress=False)

        # only use the last channel, it is the Z affinity
        # if this is affinitymap
        image = chunk[-1, :, :, :]
        if np.issubdtype(image.dtype, np.floating):
            image = (image * 255).astype(np.uint8)

        #self.thumbnail_operator(image)
        # transpose to xyzc
        image = np.transpose(image)
        image_bbox = BoundingBox.from_slices(chunk.slices[::-1][:3])

        downsample_and_upload(image,
                              image_bbox,
                              thumbnail_volume,
                              Vec(*(image.shape)),
                              mip=self.mip,
                              max_mip=6,
                              axis='z',
                              skip_first=True,
                              only_last_mip=True)

    def _upload_log(self, log, output_bbox):
        assert log
        assert isinstance(output_bbox, BoundingBox)

        # write to cloud storage
        self.log_storage.put_json(output_bbox.string + '.json', content=json.dumps(log))
        print(f'uploaded log: {log}')
