"""Perform human pose estimation using MediaPipe Holistic."""
import argparse
import os

import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic


def _rescale_joints(keypoints: np.ndarray, inverse_aspect_ratio: float) -> np.ndarray:
    """Because MediaPipe returns keypoint coordinates in the [0, 1] range, where 0 corresponds to the left or top,
    and 1 to the right or bottom edge of the image, we need to account for aspect ratio in the scaling of keypoints.
    Otherwise, datasets with different aspect ratios will result in different proportions.

    :param keypoints: An array of keypoints of arbitrary shape.
    :param inverse_aspect_ratio: The inverse aspect ratio, or: height / width.
    :return: The keypoints scaled by the inverse aspect ratio."""
    # y coordinate is the second coordinate, and coordinates are the last axis.
    keypoints[..., 1] *= inverse_aspect_ratio
    return keypoints


def _temporal_imputation(pose_array: np.ndarray) -> np.ndarray:
    """Perform temporal data imputation. If possible, poses will be interpolated from temporal neighbors.
    If not possible, then constant extrapolation will be used (i.e., edge poses will be copied).
    In some cases, this function may fail to impute all missing values, for example, if there was no value for a given
    landmark in the entire sequence.

    :param pose_array: NumPy array of shape (L, N, 3), possibly containing `np.nan` values.
    :returns: pose_array with no or some `np.nan` values.
    :raises AssertionError: If the `pose_array` did not have three axes."""

    # Input assertions.
    assert pose_array.ndim == 3, 'Expected array with three axes.'

    # Processing per body part.
    BODY_PART_OFFSETS = [(0, 33), (33, 21), (54, 21)]
    for offset, length in BODY_PART_OFFSETS:
        missing = []
        for i, pose in enumerate(pose_array):
            if np.any(np.isnan(pose[offset:offset + length])):
                missing.append(i)
        body_part_array = pose_array[:, offset:offset + length]

        # Now, for every body part, we collect the indices between which we will interpolate.
        # For every "missing index" `mi` we will interpolate between `pi` and `ni` (previous and next respectively).
        interpolation_indices = _find_interpolation_indices(missing, len(pose_array) - 1)

        # Finally, we interpolate (or extrapolate if pi or ni is -1).
        pose_array[:, offset:offset + length] = _impute(body_part_array, missing, interpolation_indices)

    return pose_array


def _impute(pose_array: np.ndarray, missing_indices: [int], interpolation_indices: dict) -> np.ndarray:
    """Impute, using interpolation or extrapolation.

    :param pose_array: NumPy array of shape (L, N, 3), possibly containing `np.nan` values.
    :param missing_indices: The missing indices to impute at.
    :param interpolation_indices: The interpolation indices to impute with.
    :returns: The pose_array after imputation."""
    for mi in missing_indices:
        pi, ni = interpolation_indices[mi]
        if pi == -1 or ni == -1:
            pose_array[mi] = _extrapolate(pose_array, mi, pi, ni)
        else:
            pose_array[mi] = _interpolate(pose_array, mi, pi, ni)
    return pose_array


def _interpolate(pose_array: np.ndarray, missing_index: int, previous_index: int, next_index: int) -> np.ndarray:
    """Linearly interpolate the pose for the given missing index between the previous and the next index.
    Neither of these should be -1.
    The interpolated pose will be set in the list of poses.

31October_2009_Saturday_tagesschau-7138.mp4     :param pose_array: NumPy array of shape (L, N, 3), possibly containing `np.nan` values.
    :param missing_index: The missing index: this pose will be interpolated from other data.
    :param previous_index: The previous non-missing index.
    :param next_index: The next non-missing index.
    :returns: An interpolated pose array.
    :raises AssertionError: if the previous and/or next index is -1."""
    assert previous_index > -1 and next_index > -1, 'Neither `previous_index` nor `next_index` should be -1.'

    interpolation_factor = _get_interpolation_factor(missing_index, previous_index, next_index)

    previous_pose = pose_array[previous_index]
    next_pose = pose_array[next_index]

    interpolated = (1 - interpolation_factor) * previous_pose + interpolation_factor * next_pose

    return interpolated


def _get_interpolation_factor(missing_index: int, previous_index: int, next_index: int) -> float:
    """Get the interpolation factor to linearly interpolate missing_index from previous_index and next_index.

    :param missing_index: Index at which we interpolate.
    :param previous_index: Lowest index of interpolation.
    :param next_index: Highest index of interpolation.
    :returns: The interpolation factor, indicating how far the missing_index is between previous and next.
    """
    return (missing_index - previous_index) / (next_index - previous_index)


# TODO: Other methods than constant extrapolation?
def _extrapolate(pose_array: np.ndarray, missing_index: int, previous_index: int, next_index: int):
    """Constantly extrapolate the pose for the given missing index between the previous and the next index.
    One of these will be -1: we will extrapolate from the other.

    :param pose_array: NumPy array of shape (L, N, 3), possibly containing `np.nan` values.
    :param missing_index: The missing index: this pose will be interpolated from other data.
    :param previous_index: The previous non-missing index.
    :param next_index: The next non-missing index..
    :returns: An extrapolated pose array.
    :raises AssertionError: If both `previous_index` and `next_index` are -1.
    """
    if previous_index == -1:
        # Extrapolate backwards from next_index.
        next_pose = pose_array[next_index]
        pose_array[missing_index] = next_pose
        return pose_array[missing_index]
    else:
        assert next_index == -1, 'One of `previous_index` and `next_index` should be 0 or greater.'
        # Extrapolate forwards from previous_index.
        next_pose = pose_array[previous_index]
        pose_array[missing_index] = next_pose
        return pose_array[missing_index]


def _find_interpolation_indices(missing_indices: [int], maximum: int) -> dict:
    """For every missing index, find the indices between which we will need to interpolate.

    :param missing_indices: The missing indices.
    :param maximum: The maximum index.
    :returns: A dictionary with key a missing index, and value a tuple with the previous and next non-missing index.
    """
    result = dict()
    for mi in missing_indices:
        pi = _find_previous_index(missing_indices, mi)
        ni = _find_next_index(missing_indices, mi, maximum)
        result[mi] = (pi, ni)
    return result


# TODO: Profile and optionally optimize this brute force implementation.
def _find_previous_index(missing_indices: [int], missing_index: int) -> int:
    """For a given list of missing indices and a missing index,
    find the last index that was not missing prior to the missing index.

    For example, if missing_indices is [3, 4] and missing_index is 4, then
    the result of this function is 2.

    If no prior missing index could be found, the function returns -1.

    :param missing_indices: List of all missing indices.
    :param missing_index: The missing index.
    :returns: The last index prior to the missing index that was not missing or -1."""
    for i in range(missing_index, -1, -1):
        if i not in missing_indices:
            return i
    return -1


# TODO: Profile and optionally optimize this brute force implementation.
def _find_next_index(missing_indices: [int], missing_index: int, maximum: int) -> int:
    """For a given list of missing indices and a missing index,
    find the first index that was not missing after the missing index.

    For example, if missing_indices is [3, 4] and missing_index is 3, then
    the result of this function is 5.
    If however the maximum parameter was 4, then the result would be -1.

    If no later missing index could be found, the function returns -1.

    :param missing_indices: List of all missing indices.
    :param missing_index: The missing index.
    :return: The first index after the missing index that was not missing or -1."""
    for i in range(missing_index, maximum + 1, 1):
        if i not in missing_indices:
            return i
    return -1


def _normalize_pose(pose: np.ndarray) -> np.ndarray:
    """Normalize the given pose. Returns the modified pose."""
    if np.count_nonzero(pose) == 0.0:  # We used constant imputation and there was nothing to impute from...
        return pose
    neck = 0.5 * (pose[12] + pose[11])
    pelvis = 0.5 * (pose[24] + pose[23])
    chest = 0.5 * (neck + pelvis)
    lshoulder = pose[11]
    rshoulder = pose[12]
    dist = np.linalg.norm(lshoulder - rshoulder)
    pose -= chest
    if dist != 0.0:
        pose /= dist
    return pose


def _normalize_hands_in_pose(pose: np.ndarray) -> np.ndarray:
    """Normalize both hands in the given pose. Returns the modified pose."""
    BODY_PARTS = [(33, 21), (54, 21)]
    for offset, length in BODY_PARTS:
        part = pose[offset:offset + length]
        if np.isclose(np.sum(np.var(part, axis=0)), 0.0):  # Body part is entirely zeros. Can't normalize.
            continue
        wrist = pose[offset]
        middle_mcp = pose[offset + 9]
        dist = np.linalg.norm(middle_mcp - wrist)
        pose[offset:offset + length] -= wrist
        if dist != 0.0:
            pose[offset:offset + length] /= dist
    return pose


def _normalize(keypoints: np.ndarray) -> np.ndarray:
    for i in range(keypoints.shape[0]):
        pose = keypoints[i]
        normalized = _normalize_pose(pose)
        normalized = _normalize_hands_in_pose(normalized)
        keypoints[i] = normalized
    return keypoints


def postprocess(keypoints: np.ndarray, inverse_aspect_ratio: float) -> np.ndarray:
    keypoints = _rescale_joints(keypoints, inverse_aspect_ratio)  # Adapt to aspect ratio.
    keypoints = _temporal_imputation(keypoints)  # Perform temporal imputation (interpolation/extrapolation).
    keypoints = np.nan_to_num(keypoints)  # Replace anything that couldn't be temporally imputed with zeros.
    keypoints = _normalize(keypoints)
    return keypoints


def extract(frames: np.ndarray, sharpen_fn=None, sharpen_sigma=None) -> np.ndarray:
    out = []
    with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True) as holistic:
        for frame in frames:
            if sharpen_fn:
                frame = sharpen(frame, sigma=sharpen_sigma)
            frame_landmarks = holistic.process(frame)
            frame_output = []

            if frame_landmarks.pose_landmarks:
                landmarks = np.stack([np.array([l.x, l.y, l.z]) for l in frame_landmarks.pose_landmarks.landmark])
                frame_output.append(landmarks)
            else:
                frame_output.append(np.full((33, 3), np.nan))
            if frame_landmarks.left_hand_landmarks:
                landmarks = np.stack(
                    [np.array([l.x, l.y, l.z]) for l in frame_landmarks.left_hand_landmarks.landmark])
                frame_output.append(landmarks)
            else:
                frame_output.append(np.full((21, 3), np.nan))
            if frame_landmarks.right_hand_landmarks:
                landmarks = np.stack(
                    [np.array([l.x, l.y, l.z]) for l in frame_landmarks.right_hand_landmarks.landmark])
                frame_output.append(landmarks)
            else:
                frame_output.append(np.full((21, 3), np.nan))

            out.append(np.concatenate(frame_output, axis=0))

        return np.stack(out)


def run_mediapipe(video_path: str) -> np.ndarray:
    """Perform human pose estimation using MediaPipe Holistic for a given video.
    The video will be processed in its entirety, and a NumPy array will be returned containing the pose keypoints.

    The shape of the NumPy array is (L, 75, 3), where L is the number of video frames,
    75 is the number of extracted keypoints, and 3 is the coordinate dimensionality (x, y, z).
    If a keypoint was not detected by MediaPipe, it will be set to `np.nan`.

    The order of the keypoints is always:
        - body pose (33)
        - left hand (21)
        - right hand (21)

    :param video_path: Path to the video file.
    :returns: A NumPy array of shape (L, 75, 3) containing the keypoints.
    :raises FileNotFoundError: If the video file was not found."""

    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frames = np.stack(frames)
    return extract(frames)


def _standard_sharpen(frame):
    # Standard sharpening kernel:
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]]
    )
    sharpened = cv2.filter2D(frame, -1, kernel)
    return sharpened


def _unsharpen_mask(frame, sigma=2):
    # Syntax borrowed from: https://stackoverflow.com/questions/32454613/python-unsharp-mask
    gaussian = cv2.GaussianBlur(frame, (0, 0), sigma)
    # From docs:  src1*alpha + src2*beta + gamma, 
    # When beta is -1, src1*alpha -src2, so we minus the gausian blur. 
    # Alpha is 1 so the magnitude of the pixel values are not effected by t
    # the negative weighting. 
    return cv2.addWeighted(frame, 2.0, gaussian, -1.0, 0)


def sharpen(frame, sigma=2):
    if sigma:
        return _unsharpen_mask(frame, sigma=sigma)
    return _standard_sharpen(frame)


def main(args):
    video_name, _ = os.path.splitext(os.path.basename(args.clip))
    output_path = os.path.join(args.out_dir, video_name + '.npy')
    if not os.path.isfile(output_path):
        keypoints = run_mediapipe(args.clip)
        np.save(output_path, keypoints)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('clip', type=str,
                        help='Path to the video from which we will extract MediaPipe features.')
    parser.add_argument('out_dir', type=str, help='Output directory to which MediaPipe features will be saved.')

    args = parser.parse_args()

    main(args)
