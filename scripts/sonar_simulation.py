#!/usr/bin/env python3

import yaml
import logging
import matplotlib.pyplot as plt
import numpy as np
import itertools as it
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from tqdm import tqdm

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def parse_args(argv=None):
    """
    Argument parsing routine.

    :param list argv: A list of argument strings.
    :return: A parsed and verified arguments namespace.
    :rtype: :py:class:`argparse.Namespace`
    """

    parser = ArgumentParser(
        description=(
            "Python simulation of optical and sonar sensors for 3D reconstruction."
        )
    )
    parser.add_argument(
        "-l",
        "--log_level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="logging level",
    )
    args = parser.parse_args(argv)
    return args


def box3d(n=16):
    """Generate 3D points inside a cube with n-points along each edge"""
    points = []
    N = tuple(np.linspace(-1, 1, n))
    for i, j in [(-1, -1), (-1, 1), (1, 1), (0, 0)]:
        points.extend(set(it.permutations([(i,) * n, (j,) * n, N])))
    return np.hstack(points) / 2


def Pi(ph: np.ndarray):
    """
    Converts coordinates from homogeneous to inhomogeneous.

    Args:
        ph (np.array): shape (n+1,m)

    Returns:
        p (np.array): shape (n,m)
    """
    p = ph[:-1] / ph[-1]  # divide by and remove last coordinate
    return p


def PiInv(p: np.ndarray):
    """
    Converts coordinates from inhomogeneous to homogeneous.

    Args:
        p (np.array): shape (n,m)

    Returns:
        ph (np.array): shape (n+1,m)
    """
    ph = np.vstack((p, np.ones(p.shape[1])))
    return ph


def camera_intrinsic(f: float, c: tuple[float, float], alpha=1.0, beta=0.0):
    """
    Create a camera intrinsic matrix

    Args:
        f (float): focal length
        c (tuple): 2D principal point (x,y)
        alpha (float): skew
        beta (float): aspect ratio

    Returns:
        K (np.array): intrinsic camera matrix, shape (3, 3)
    """
    K = np.array(
        [
            [f, beta * f, c[0]],
            [0, alpha * f, c[1]],
            [0, 0, 1],
        ],
    )
    return K


def project_optical(K, R, t, Q):
    """
    Project 3D points to 2D using the pinhole model,
    without distortion.

    Args:
        K: Camera intrinsics matrix
        R: Rotation matrix
        t: Translation vector
        Q: 3D points, shape (3, n)
    """
    pose = K @ np.hstack((R, t))
    Qh = PiInv(Q)  # (4, n)
    Ph = pose @ Qh  # (3, n)
    P = Pi(Ph)  # (2, n)
    return P


def project_sonar(Qs):
    """
    Project 3D points to 2D in the sonar frame.

    Args:
        Qs: 3D points in {sonar}, shape (3, n)

    Returns:
        x: 2D points in {sonar} as (r, theta), shape (2, n)
    """
    # r = np.linalg.norm(Qs, axis=0)  # np.sqrt(x^2 + y^2 + z^2)
    x = Qs[0, :]
    y = Qs[1, :]
    z = Qs[2, :]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(x, y)  # arctan2(x, y)
    x = np.array([r, theta])  # shape (2, n)
    return x


def range_solution(s, p, K, R, t):
    """
    Closed-form range solution for range measurements.

    Args:
        s: 2D points in {sonar} as (r, theta), shape (2, n)
        p: 2D points in {camera}, shape (2, n)
        K: Camera intrinsics matrix
        R: Rotation matrix from {camera} to {sonar}
        t: Translation vector from {camera} to {sonar}

    Returns:
        z: range measurements, shape (n,)
    """
    f = K[0, 0]
    range_ = s[0, :]
    p = np.vstack((p, f * np.ones(p.shape[1])))
    z = f * (
        (range_ * np.linalg.norm(p[:2]) - t.T @ R @ p)
        / np.linalg.norm(p[:2]) ** 2
    )
    z = z.reshape(-1)
    return z


def azimuth_solution(s, p, K, R, t):
    """
    Closed-form azimuth solution for range measurements.

    Args:
        s: 2D points in {sonar} as (r, theta), shape (2, n)
        p: 2D points in {camera}, shape (2, n)
        K: Camera intrinsics matrix
        R: Rotation matrix from {camera} to {sonar}
        t: Translation vector from {camera} to {sonar}

    Returns:
        z: range measurements, shape (n,)
    """
    f = K[0, 0]
    r1 = R[0, :].reshape(1, 3)
    r2 = R[1, :].reshape(1, 3)
    theta = s[1, :].reshape(-1, 1)
    tx = t[0, 0]
    ty = t[1, 0]
    p = np.vstack((p, f * np.ones(p.shape[1])))
    z = np.zeros(s.shape[1])
    for i in range(s.shape[1]):
        num = f * (np.tan(theta[i]) * ty - tx)
        den = (r1 - np.tan(theta[i]) * r2) @ p[:, i]
        z[i] = (num / den).item()
    return z


def form_alpha(p, R):
    """
    Returns:
        ax: shape (n, 3)
        ay: shape (n, 3)
    """
    x, y = p[0, :], p[1, :]
    r1, r2, r3 = R[:, 0], R[:, 1], R[:, 2]  # column vectors
    ax = [xi * r3 - r1 for xi in x]
    ay = [yi * r3 - r2 for yi in y]
    ax = np.array(ax)
    ay = np.array(ay)
    return ax, ay


def m_solution(s, p, R, t):
    """
    3rd closed-form solution for range measurements.
    """
    ax, ay = form_alpha(p, R)  # shape (n, 3), (n ,3)
    theta = s[1, :]  # shape (n,)
    z = np.zeros(s.shape[1])
    for i in range(s.shape[1]):
        ax1, ax2, ax3 = ax[i]
        ay1, ay2, ay3 = ay[i]
        th = theta[i]
        num = (
            (ay1 * np.tan(th) + ay2) * ax[i] - (ax1 * np.tan(th) + ax2) * ay[i]
        ) @ t
        den = (ay1 * np.tan(th) + ay2) * ax3 - (ax1 * np.tan(th) + ax2) * ay3
        z[i] = (num / den).item()
    return z


def add_noise(s, p, s_noise_sd, p_noise_sd):
    """
    Add Gaussian noise to sonar and optical measurements.

    Args:
        s: 2D points in {sonar} as (r, theta), shape (2, n)
        p: 2D points in {camera}, shape (2, n)
        s_noise_sd: (sigma_r, sigma_theta), standard deviation of sonar noise
        p_noise_sd: (sigma_x, sigma_y), standard deviation of camera noise

    Returns:
        s_n: Noisy sonar measurements, shape (2, n)
        p_n: Noisy optical measurements, shape (2, n)
    """
    s_noise_sd = np.array(s_noise_sd).reshape(2, 1)
    p_noise_sd = np.array(p_noise_sd).reshape(2, 1)
    s_n = s + np.random.normal(0, s_noise_sd, s.shape)
    p_n = p + np.random.normal(0, p_noise_sd, p.shape)
    return s_n, p_n


def epipolar_value(s, p, K, R, t):
    """
    Compute the epipolar value for a pair of optical and sonar measurements.

    Args:
        s: 2D points in {sonar} as (r, theta), shape (2, n)
        p: 2D points in {camera} as (x, y), shape (2, n)
        K: Camera intrinsics matrix
        R: Rotation matrix from {camera} to {sonar}
        t: Translation vector from {camera} to {sonar}

    Returns:
        ep: Epipolar value, shape (n,)
    """
    if s.shape == (2,):
        s = s.reshape(-1, 1)
    if p.shape == (2,):
        p = p.reshape(-1, 1)
    f = K[0, 0]
    r = s[0].item()
    th = s[1].item()
    r1 = R[0, :].reshape(1, 3)
    r2 = R[1, :].reshape(1, 3)
    tx = t[0, 0]
    ty = t[1, 0]
    p = np.vstack((p, f * np.ones(p.shape[1])))
    term1 = (
        (np.linalg.norm(t) ** 2 - r**2)
        * (r1 - np.tan(th) * r2).T
        @ (r1 - np.tan(th) * r2)
    )
    term2 = (np.tan(th) * ty - tx) ** 2 * np.eye(3)
    term3 = 2 * (np.tan(th) * ty - tx) * (r1 - np.tan(th) * r2).T @ t.T @ R
    ep = p.T @ (term1 + term2 + term3) @ p
    return ep.item()


def epipolar_value_parallel(s, p, K, R, t):
    """
    Compute the epipolar value for a pair of optical and sonar measurements
    for parallel camera configuration (tx != 0).

    Args:
        p: 2D points in {camera}, shape (2, 1)
        s: 2D points in {sonar} as (r, theta), shape (2, 1)
        K: Camera intrinsics matrix
        R: Rotation matrix from {camera} to {sonar}
        t: Translation vector from {camera} to {sonar}

    Returns:
        ep: Epipolar value, shape (n,)
    """
    if s.shape == (2,):
        s = s.reshape(-1, 1)
    if p.shape == (2,):
        p = p.reshape(-1, 1)
    f = K[0, 0]
    r = s[0].item()
    th = s[1].item()
    tx = t[0, 0]
    p = np.vstack((p, f * np.ones(p.shape[1])))
    beta = tx / r
    k2 = 1 + np.tan(th) ** 2
    U = r**2 * np.array(
        [
            [-1, 0, np.tan(th)],
            [0, beta**2, 0],
            [np.tan(th), 0, k2 * beta**2 - (np.tan(th)) ** 2],
        ]
    )
    ep = p.T @ U @ p
    return ep.item()


def maximum_likelihood_estimate(s, p, K, R, t, s_sd, p_sd, lambda_, method):
    """
    Maximum likelihood estimate of the optical and sonar measurements.

    Args:
        s: Sonar measurements as (r, theta), shape (2, n)
        p: Optical measurements as (x, y), shape (2, n)
        K: Camera intrinsics matrix
        R: Rotation matrix from {camera} to {sonar}
        t: Translation vector from {camera} to {sonar}
        s_sd: (sigma_r, sigma_theta), standard deviation of sonar noise
        p_sd: (sigma_x, sigma_y), standard deviation of camera noise
        lambda_: Regularization parameter
        method: Optimization method

    Returns:
        s_mle: 2D points in {sonar} as (r, theta), shape (2, n)
        p_mle: 2D points in {camera}, shape (2, n)
    """

    def objective_func(X):
        """
        Args:
            X: (xi, yi, ri, thi), vector to optimize, shape (4,)

        Returns:
            residual: scalar value
        """
        nonlocal i, j
        xi, yi, ri, thi = X
        si = np.vstack((ri, thi))
        pi = np.vstack((xi, yi))
        ep = epipolar_value(si, pi, K, R, t)
        residual = (
            (xi - p[0, i]) ** 2 / (p_sd[0] ** 2)
            + (yi - p[1, i]) ** 2 / (p_sd[1] ** 2)
            + (ri - s[0, i]) ** 2 / (s_sd[0] ** 2)
            + (thi - s[1, i]) ** 2 / (s_sd[1] ** 2)
            + lambda_ * abs(ep)
        )
        return residual

    p_mle = np.zeros(p.shape)
    s_mle = np.zeros(s.shape)
    n_failed = 0
    bounds = np.array(
        [
            [-K[0, 0], K[0, 0]],
            [-K[0, 0], K[0, 0]],
            [0.0, 10.0],
            [-0.5, 0.5],
        ]
    )
    for i in tqdm(range(s.shape[1])):  # for each point
        j = 0  # for residuals

        # Optimize point
        x0 = np.vstack((p[0, i], p[1, i], s[0, i], s[1, i])).reshape(-1)
        result = minimize(objective_func, x0, method=method, bounds=bounds)
        p_mle[:, i] = result.x[:2]
        s_mle[:, i] = result.x[2:]

        # Review result
        if i % 10 == 0:
            logging.debug(
                f"\n\tpoint i: {i}\n\tx0: {x0}\n\tx: {result.x}\n\tcost: {result.fun}"
            )
        if result.success is not True:
            n_failed += 1
            logging.debug(f"Optimization failed for point {i}")
            logging.debug(result.message)
    logging.info(f"Failed optimization: {n_failed}")
    return p_mle, s_mle


def reconstruct(p, K, Zo):
    """
    Reconstruction of 3D points in {camera} from Z-values using the optical
    projection model.
    """
    f = K[0, 0]
    deltax, deltay = K[0, 2], K[1, 2]
    Xo = (p[0, :] - deltax) * Zo / f
    Yo = (p[1, :] - deltay) * Zo / f
    Qo_rec = np.vstack((Xo, Yo, Zo))
    return Qo_rec


def plot_epipolar_values(s, p, s_n, p_n, K, R, t):
    """
    Plot the epipolar values for a pair of optical and sonar measurements.
    """
    epi = [epipolar_value(s[:, i], p[:, i], K, R, t) for i in range(s.shape[1])]
    epi_noise = [
        epipolar_value(s_n[:, i], p_n[:, i], K, R, t) for i in range(s.shape[1])
    ]
    plt.figure()
    plt.title("Epipolar values of all points")
    plt.plot(epi, label="True points")
    plt.plot(epi_noise, label="Noisy points")
    plt.xlabel("Point index")
    plt.ylabel("Epipolar value")
    plt.legend()


def plot_3d(Q, title, limit=None, **kwargs):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(Q[0, :], Q[1, :], Q[2, :], **kwargs)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_aspect("equal")
    if limit:
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
    ax.set_title(title)


def plot_comparison(Q_true, Q1, Q2, title1, title2, scale):
    fig = plt.figure(figsize=(12, 6))
    # First plot
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(
        Q_true[0, :],
        Q_true[1, :],
        Q_true[2, :],
        color=((0, 0, 1, 0.5)),
        label="True",
    )
    ax.scatter(
        Q1[0, :],
        Q1[1, :],
        Q1[2, :],
        c="r",
        label="Reconstructed",
        marker="X",
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_zlim(-scale, scale)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(title1)
    # Second plot
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.scatter(
        Q_true[0, :],
        Q_true[1, :],
        Q_true[2, :],
        color=((0, 0, 1, 0.5)),
        label="True",
    )
    ax.scatter(
        Q2[0, :],
        Q2[1, :],
        Q2[2, :],
        c="r",
        label="Reconstructed",
        marker="X",
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_zlim(-scale, scale)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(title2)


def plot_3d_reconstruction(Q_true, Q_rec, title, scale=2.0):
    """
    Plot the 3D points of the true and reconstructed scene.

    Args:
        Q_true: True 3D points, shape (3, n)
        Q_rec: Reconstructed 3D points, shape (3, n)
        title: Plot title
        scale: +ve float, plot limits
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(
        Q_true[0, :],
        Q_true[1, :],
        Q_true[2, :],
        color=((0, 0, 1, 0.5)),
        label="True",
    )
    ax.scatter(
        Q_rec[0, :],
        Q_rec[1, :],
        Q_rec[2, :],
        c="r",
        label="Reconstructed",
        marker="X",
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_xlim(-scale / 2, scale / 2)
    ax.set_ylim(-scale / 2, scale / 2)
    ax.set_zlim(-scale / 2, scale / 2)
    ax.set_aspect("equal")
    # ax.legend(loc='upper right')
    ax.set_title(title)


def plot_dual_projection(s, p, K, title):
    """
    Plot the results of the optical and sonar projection models.

    Args:
        p: 2D points in {camera}, shape (2, n)
        s: 2D points in {sonar} as (r, theta), shape (2, n)
        K: Camera intrinsics matrix
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(title)
    ax[0].scatter(p[0, :], p[1, :], s=5)
    ax[0].set_title("Camera frame")
    ax[0].set_xlabel("x (pixels)")
    ax[0].set_ylabel("y (pixels)")
    ax[0].set_aspect("equal")
    # ax[0].set_xlim(-K[0, 0] / 2 + K[0, 2], K[0, 0] / 2 + K[0, 2])
    # ax[0].set_ylim(-K[0, 0] / 2 + K[1, 2], K[0, 0] / 2 + K[1, 2])
    ax[0].set_aspect("equal")
    ax[1].scatter(s[1, :], s[0, :], s=5)
    ax[1].set_title("Sonar frame")
    ax[1].set_xlabel("theta (rad)")
    ax[1].set_ylabel("range (m)")


def main():
    # Logging
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="[%(levelname)s]: %(message)s",
    )

    # Load config parameters
    config_path = "config/sim_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    d_baseline = config["transforms"]["baseline_distance"]
    d_plane = config["transforms"]["plane_distance"]
    scale = config["transforms"]["scale"]
    t_wo_x = config["transforms"]["camera_x_offset"]
    t_wo_y = config["transforms"]["camera_y_offset"]
    f = config["camera_intrinsics"]["focal_length"]
    delta_x = config["camera_intrinsics"]["delta_x"]
    delta_y = config["camera_intrinsics"]["delta_y"]
    s_noise_sd = (
        config["sensor_noise_std"]["r"],
        config["sensor_noise_std"]["theta"],
    )
    p_noise_sd = (
        config["sensor_noise_std"]["x"],
        config["sensor_noise_std"]["y"],
    )
    method = config["maximum_likelihood"]["method"]
    lambda_list = config["maximum_likelihood"]["lambda_values"]
    logging.info(f"Configuration loaded: {config}")

    # Create object
    Qw = scale * box3d(n=16)  # 3D points centered at origin of {world}
    # tw = np.array([[0.5, 0., 0.]]).T
    # Qw = Qw + tw      # offset box

    # Parallel camera configuration (R and t from the paper)
    t_os = np.array([[d_baseline, 0, 0]]).T
    Ros = np.array(
        [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    )  # 90 degree rotation around x-axis

    # Camera extrinsics, {world} to {camera}
    t_wo = np.array([[t_wo_x, t_wo_y, d_plane]]).T
    Rwo = np.eye(3)
    Qo = Rwo @ Qw + t_wo  # points in {camera}

    K = camera_intrinsic(f, (delta_x, delta_y))

    # Sonar extrinsics, {camera} to {sonar}
    Qs = Ros @ Qo + t_os

    # Projection
    p = project_optical(K, Rwo, t_wo, Qw)
    s = project_sonar(Qs)
    s_n, p_n = add_noise(s, p, s_noise_sd, p_noise_sd)

    # Check epipolar values should be zero without noise
    # plot_epipolar_values(s, p, s_n, p_n, K, Ros, t_os)

    # Optimize
    # epi_noise = np.array(
    #     [
    #         epipolar_value(s_n[:, i], p_n[:, i], K, Ros, t_os)
    #         for i in range(s.shape[1])
    #     ]
    # )
    # mean_ep = np.median(abs(epi_noise))
    # logging.debug(f"Mean epipolar value: {mean_ep}")

    # Without MLE
    Zo_azi = azimuth_solution(s_n, p_n, K, Ros, t_os)
    Qo_azi = reconstruct(p, K, Zo_azi)
    Qw_azi = np.linalg.inv(Rwo) @ (Qo_azi - t_wo)  # to {world}
    plot_3d_reconstruction(Qw, Qw_azi, "Without MLE", scale=scale)

    # Optimize
    for i, lambda_ in enumerate(lambda_list):
        logging.info(f"Running MLE for lambda={lambda_:.1E}")
        p_mle, s_mle = maximum_likelihood_estimate(
            s_n, p_n, K, Ros, t_os, s_noise_sd, p_noise_sd, lambda_, method
        )
        # Azimuth solution
        Zo_azi_mle = azimuth_solution(s_mle, p_mle, K, Ros, t_os)
        Qo_azi_mle = reconstruct(p, K, Zo_azi_mle)
        Qw_azi_mle = np.linalg.inv(Rwo) @ (Qo_azi_mle - t_wo)  # to {world}
        plot_3d_reconstruction(
            Qw, Qw_azi_mle, r"$\lambda=$" + f"{lambda_:.1E}", scale=scale
        )

    # Rangle solution
    # Zo_range = range_solution(s, p, K, Ros, t_os)
    # Qo_range = reconstruct(p, K, Zo_range)
    # Qo_range_origin = np.mean(Qo_range, axis=1).reshape(-1, 1)
    # Qo_range_align = Qw + Qo_range_origin
    # Qw_range = np.linalg.inv(Rwo) @ (Qo_range - t_wo)  # to {world}

    # m solution
    # Zo_m = m_solution(s, p, Ros, t_os)
    # Qo_m = reconstruct(p, K, Zo_m)
    # Qo_m_origin = np.mean(Qo_m, axis=1).reshape(-1, 1)
    # Qo_m_align = Qw + Qo_m_origin
    # Qw_m = np.linalg.inv(Rwo) @ (Qo_m - t_wo) # to {world}

    # Plot
    plot_dual_projection(s, p, K, "Projection of true measurements")
    plot_dual_projection(s_n, p_n, K, "Projection of noisy measurements")

    # plot_3d_reconstruction(Qw, Qw_range, "Range solution in {world}")
    # plot_3d_reconstruction(Qw, Qw_azi, "Azimuth solution in {world}", scale)
    # plot_3d_reconstruction(
    #     Qw, Qw_azi_mle, "MLE Azimuth solution in {world}", scale
    # )
    # plot_3d_reconstruction(Qw, Qw_m, "m solution in {world}")

    # plot_3d_reconstruction(Qo_range_align, Qo_range, "Range solution by aligning box origins")
    # plot_3d_reconstruction(Qo_azi_align, Qo_azi, "Azimuth solution by aligning box origins")
    # plot_3d_reconstruction(Qo_m_align, Qo_m, "M solution by aligning box origins")

    # plot_3d(Qw, "3D points in {world}")
    # plot_3d(Qs, "3D points in {sonar}")
    plot_comparison(
        Qw,
        Qw_azi,
        Qw_azi_mle,
        "Azimuth solution\n",
        r"MLE Azimuth solution $\lambda=$" + str(lambda_),
        scale,
    )
    plt.show()


if __name__ == "__main__":
    main()
