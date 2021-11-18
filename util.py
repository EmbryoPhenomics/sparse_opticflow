#! /usr/bin/env python3
import contextlib
from tqdm import tqdm
from natsort import natsorted, ns
import h5py
import os

import vuba


class EmptyPGbar:
    """Dummy progress bar for progress bar context."""

    @staticmethod
    def update(n):
        """Dummy update for progress bar context."""
        pass


@contextlib.contextmanager
def pgbar(total, show):
    """
    Create a progress bar to be used in a 'with' context.

    Keyword arguments:
        total    Integer.    Length of progress bar (Required).
        show     Bool.       Whether to return a progress bar (Required).

    Returns:
        tqdm progress bar.

    """

    if show:
        pgbar = tqdm(total=total)
        try:
            yield pgbar
        finally:
            pgbar.close()
    else:
        yield EmptyPGbar


def slices(shape):
    """
    Create a tuple of slices from a given shape.

    Keyword arguments:
        shape    Tuple.   Dimensions to creates slices (Required).

    Returns:
        Tuple.   Slices for the supplied dimension.

    """
    return tuple(map(slice, shape))


class Dataset:
    """
    Convenience wrapper around h5py.

    Note all files hdf5 files are opened in 'a' mode via h5py.

    """

    def __init__(self, filename):
        """
        Keyword arguments:
            filename    String.    Filename to create HDF5 (Required).

        """
        self.file = h5py.File(filename, mode="a")

    def close(self):
        """ Closes the declared file. """
        self.file.close()

    def keys_shapes(self):
        """
        Convenience function for retrieving subdatasets and their corresponding
        dimensions from a HDF5 file.

        Keyword arguments:
            file    String.    File to grab metadata from (Required).

        Returns:
            Dict.    Dict of subdataset names and their corresponding dimensions.

        """

        dims = {}
        for name in list(self.file.keys()):
            dims[name] = self.file[name].shape
        return dims

    def empty(self, name, shape):
        """
        Create an empty subdataset.

        Keyword arguments:
            name    String.   Name for subdataset (Required).
            shape  Tuple.     Dimensions of new subdataset (Required).

        """
        if name in list(self.file.keys()):
            del self.file[name]

        self.file.create_dataset(name, shape=shape)
        self.file.flush()

    def __getitem__(self, name):
        """Getter for retrieving data from the declared file."""
        return self.file[name]

    def __setitem__(self, name, data, shape=None):
        """Setter that dumps data out-of-core to a given subdataset. """
        if name not in list(self.file.keys()):
            self.file.create_dataset(name, data=data, shape=data.shape)
        else:
            if shape is None:
                shape = slices(data.shape)
            self.file[name][shape] = data
        self.file.flush()


@contextlib.contextmanager
def dataset(filename):
    """
    Create a Dataset object for use in a 'with' context.

    Keyword arguments:
        filename    String.   Filename to create (Required).

    """
    ds = Dataset(filename)
    try:
        yield ds
    finally:
        ds.close()


def export_results(results, out_file):
    """
    Export a results dict as a HDF5 file out-of-core.

    Keyword arguments:
        results    Dict.    Dict of results obtained from a data integration method (Required.
        out_file   String.  Filename to export results (Required).

    """
    with dataset(out_file) as ds:
        for key in results:
            ds[key] = results[key]


def trim_datasets(in_files, keys_shapes, verbose=True):
    """
    Trim a set of hdf5 files to a given set of subdataset shapes.

    Keyword arguments:
        in_files     List.   List of filenames to files to trim (Required).
        keys_shapes  Dict.   Dict of subdataset names with corresponding shapes
                             to trim to (Required).
        verbose      Bool.   Whether to print progress info, default is True.

    """
    if verbose:
        print("Trimming datasets to uniform length...")
    with pgbar(total=len(in_files), show=verbose) as pg:
        for file in in_files:
            with dataset(file) as ds:
                for key in keys_shapes:
                    ds[key] = ds[key][slices(keys_shapes[key])]
            pg.update(1)


def concat_datasets(in_files, out_file, cleanup=True, verbose=True):
    """
    Combine a set of hdf5 files out-of-core.

    Note that in_files is sorted prior to concatenation.

    Keyword arguments:
        in_files     List.    List of filenames to files to combine (Required).
        out_file     String.  Filename to fill with the combined data (Required).
        cleanup      Bool.    Whether to remove the individual files, default is True.
        verbose      Bool.    Whether to print progress info, default is True.

    """

    sortedFiles = natsorted(in_files, alg=ns.IGNORECASE)
    with dataset(vuba.take_first(sortedFiles)) as ds:
        dims = ds.keys_shapes()

    if verbose:
        print("Combining datasets into a single file...")
    with pgbar(total=len(in_files), show=verbose) as pg:
        with dataset(out_file) as DS:
            for key in dims:
                DS.empty(key, shape=(len(sortedFiles), *dims[key]))

            for i, file in enumerate(sortedFiles):
                with dataset(file) as ds:
                    for key in ds.keys_shapes():
                        DS[key][i, :] = ds[key][:]

                if cleanup:
                    os.remove(file)

                pg.update(1)
