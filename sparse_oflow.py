#! /usr/bin/env python3
import numpy as np
import cv2
import math
from more_itertools import pairwise
from tqdm import tqdm
import re
import multiprocessing as mp

import util
import vuba


def pointwise_diff(diff, axis):
    """
    Compute the pointwise distances between two sets of points or individual
    coordinates.

    """
    return np.sqrt(np.sum(diff ** 2, axis=axis))


def base(frames, mask, feature_params, lk_params):
    """
    Wrapper around OpenCV sparse optical flow implementation.

    This wrapper essentially performs the typical setup required to run sparse optical
    flow, and produces a generator for iterating through not only the frames produced
    but also the points detected.

    Keyword arguments:
        frames            Vuba.Frames.        Frames to compute sparse optical flow on (Required).

        mask              Numpy.ndarray.      Mask to filter frames, if specified as None, then
                                              the entirety of the individual frames will be used.

        feature_params    Dict.               Keyword arguments to pass to feature detection in
                                              OpenCV (Required).

        lk_params         Dict.               Keyword arguments to pass to sparse optical flow in
                                              OpenCV (Required).

    Returns:
        Generator.    Generator object that will yield both frames and blank masks, but also the
                      points detected.

    """
    if mask is not None:
        filter_mask = vuba.Mask(mask)
    else:
        filter_mask = None

    frame_it = iter(frames)
    first = next(frame_it)

    if filter_mask:
        first = filter_mask(first)

    p0 = cv2.goodFeaturesToTrack(first, mask=None, **feature_params)
    for frame_gray in frame_it:
        if filter_mask:
            frame_gray = filter_mask(frame_gray)

        p1, st, err = cv2.calcOpticalFlowPyrLK(first, frame_gray, p0, None, **lk_params)

        if p1 is None:
            good_new = None
            good_old = None
        else:    
            good_new = p1[st == 1]
            good_old = p0[st == 1]

        yield (frame_gray, (good_new, good_old))

        first = frame_gray.copy()

        if p1 is None:
            p0 = p0.reshape(-1, 1, 2)


def motion_tracks(
    frames, mask, feature_params, lk_params, writer, display=True, ms=30, verbose=True
):
    """
    Motion analysis method adapted from Tills et al. 2013.

    This method computes angles of rotation, centers of rotation and center of mass.
    Angles of rotation are computed using the arc tangent of y/x where y and x are
    the differences between consecutive points (atan2). Centers of rotation are computed
    via computing the means of positive and negative coordinates. Centers of mass
    are computed as the center between both positive and negative centers of rotation.

    Note that there are some differences in the metrics returned in comparison to the
    implementation in Tills et al. 2013. Namely, distances between consecutive points
    are used over recording the raw coordinates to get a direct measure of magnitude
    of movement.

    Keyword arguments:
        frames            Vuba.Frames.        Frames to compute sparse optical flow on (Required).

        mask              Numpy.ndarray.      Mask to filter frames, if specified as None, then
                                              the entirety of the individual frames will be used.

        feature_params    Dict.               Keyword arguments to pass to feature detection in
                                              OpenCV (Required).

        lk_params         Dict.               Keyword arguments to pass to sparse optical flow in
                                              OpenCV (Required).

        writer            Vuba.Writer.        Writer object for exporting visual output to (Required).

        display           Boolean.            Whether to display a live view of the method. Default
                                              is True.

        ms                Integer.            Wait time (microseconds) between frames for live view.
                                              Default is 30 ms.

        verbose           Bool.               Whether to display progress, default is True.

    Returns:
        Tuple.     Tuple containing lists for positive and negative angles of rotation metrics and
                   centers of rotation/mass metrics. Angles of rotation metrics contain sum totals
                   of the distances between consecutive points. Centers of rotation/mass
                   metrics contain distances between consecutive points as well.

    References:
        Tills O, Bitterli TS, Culverhouse P, Spicer JI, Rundle SD(2013) A novel application of motion
        analysis for detecting stress responses in embryos at different stages of development.
        BMC Bioinformatics 14; 37

    """
    # Movement vectors
    pos_m = []
    neg_m = []
    ce_m = []
    com_m = []

    # From original impl
    pos_angle = []
    pos_x = []
    pos_y = []
    neg_angle = []
    neg_x = []
    neg_y = []

    # Visual output canvases
    first = vuba.take_first(frames)
    ce_mask = cv2.cvtColor(np.zeros_like(first), cv2.COLOR_GRAY2BGR)
    rot_mask = cv2.cvtColor(np.zeros_like(first), cv2.COLOR_GRAY2BGR)

    with util.pgbar(len(frames), verbose) as pg:
        for (frame, (good_new, good_old)) in base(
            frames, mask, feature_params, lk_params
        ):
            if good_new is None and good_old is None:
                pos_m.append(np.nan)
                neg_m.append(np.nan)
                pos_angle.append(np.nan)
                neg_angle.append(np.nan)
                pos_x.append(np.nan)
                pos_y.append(np.nan)
                neg_x.append(np.nan)
                neg_y.append(np.nan)
                ce_m.append(np.nan)
            else:
                # Magnitude of rotation (positive and negative)
                ind_diff = good_new - good_old
                ind_dist = pointwise_diff(ind_diff, axis=(1,))
                rot = np.asarray(
                    list(map(lambda vals: math.atan2(*np.flip(vals)), ind_diff))
                ) # compute angles of rotation

                pos_m.append(np.sum(ind_dist[rot > 0]))
                neg_m.append(np.sum(ind_dist[rot <= 0]))

                # From original impl (raw values)
                pos_angle.append(np.sum(rot[rot > 0]))
                neg_angle.append(np.sum(rot[rot <= 0]))

                pos_x.append(np.sum([coord[0] for coord in good_new[rot > 0]]))
                pos_y.append(np.sum([coord[1] for coord in good_new[rot > 0]]))
                neg_x.append(np.sum([coord[0] for coord in good_new[rot <= 0]]))
                neg_y.append(np.sum([coord[1] for coord in good_new[rot <= 0]]))

                # Center of mass movement
                com_old = good_old.mean(axis=(0,))
                com_new = good_new.mean(axis=(0,))
                com_m.append(pointwise_diff((com_new - com_old), axis=(0,)))
                com_old, com_new = map(tuple, (com_old, com_new))  # for live view

                # Movement relative to center of mass
                old_com_dist = pointwise_diff((good_old - com_old), axis=(1,))
                new_com_dist = pointwise_diff((good_new - com_new), axis=(1,))
                ce_dist = np.sum(new_com_dist - old_com_dist)
                ce_m.append(ce_dist)

                if writer or display:
                    ce_line_color = (0, 0, 255)
                    if ce_dist > 0:
                        ce_line_color = (0, 255, 255)
                    for old, new, r in zip(good_old, good_new, rot):
                        old, new = map(tuple, (old, new))

                        rot_line_color = (0, 0, 255)
                        if r > 0:
                            rot_line_color = (0, 255, 255)

                        new = tuple(map(int, new))
                        old = tuple(map(int, old))
                        cv2.line(rot_mask, new, old, rot_line_color, 1)
                        cv2.line(ce_mask, new, old, ce_line_color, 1)

                    com_new = tuple(map(int, com_new))
                    com_old = tuple(map(int, com_old))
                    cv2.line(rot_mask, com_new, com_old, (255, 0, 0), 2)
                    cv2.line(ce_mask, com_new, com_old, (255, 0, 0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            img = np.hstack((frame, rot_mask, ce_mask))

            if writer:
                writer.write(img)

            if display:
                cv2.imshow("Motion tracking live view", img)
                k = cv2.waitKey(ms)
                if k == 27:
                    break

            pg.update(1)

    return (pos_m, neg_m, ce_m, com_m, pos_angle, pos_x, pos_y, neg_angle, neg_x, neg_y)


class Par_MotionAnalysis:
    """
    Class for computing the above motion analysis implementation in parallel
    for a compiled multi-timepoint video.

    """

    def __init__(self, file, output, split_by, mask, feature_params, lk_params):
        self.file = file
        self.split_by = split_by
        self.output = output
        self.mask = mask
        self.feature_params = feature_params
        self.lk_params = lk_params

    def _compute(self, args):
        timepoint, outpath = args
        video = vuba.Video(self.file)
        frames = video.read(
            start=timepoint * self.split_by,
            stop=((timepoint + 1) * self.split_by),
            low_memory=self.low_memory,
            grayscale=True,
        )

        out = motion_tracks(
            frames,
            self.mask,
            self.feature_params,
            self.lk_params,
            None,
            False,
            verbose=False,
        )
        pm, nm, ce, cm, pa, px, py, na, nx, ny = map(np.asarray, out)
        ret = dict(
            pos_mag=pm_out, 
            neg_mag=nm_out, 
            cofM=cm_out, 
            cofE=ce_out,
            pos_angle=pa,
            neg_angle=na, 
            pos_x=px,
            pos_y=py,
            neg_x=nx,
            neg_y=ny)

        util.export_results(ret, outpath)
        video.close()

    def compute(self, cores, low_memory=True, num_timepoint=None):
        self.low_memory = low_memory

        if not num_timepoint:
            video = vuba.Video(file)
            num_timepoint = int(len(video) / split_by)
            video.close()

        timepoints = list(range(num_timepoint))
        out_files = [re.sub(".avi", f"_{i}.h5", self.file) for i in timepoints]
        args = zip(timepoints, out_files)

        print("Computing motion analysis...")
        pool = mp.Pool(processes=cores)
        list(tqdm(pool.imap(self._compute, args), total=num_timepoint))
        pool.close()

        util.concat_datasets(out_files, self.output)
