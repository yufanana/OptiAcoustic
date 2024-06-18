#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from tqdm import tqdm


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
    z = f * ((range_ * np.linalg.norm(p[:2]) - t.T @ R @ p) / np.linalg.norm(p[:2]) ** 2)
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
        z[i] = (num/den).item()
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
    ax, ay = form_alpha(p, R)   # shape (n, 3), (n ,3)
    theta = s[1, :]  # shape (n,)
    z = np.zeros(s.shape[1])
    for i in range(s.shape[1]):
        ax1, ax2, ax3 = ax[i]
        ay1, ay2, ay3 = ay[i]
        th = theta[i]
        num = ((ay1 * np.tan(th) + ay2) * ax[i] - (ax1 * np.tan(th) + ax2) * ay[i]) @ t
        den = (ay1 * np.tan(th) + ay2) * ax3 - (ax1 * np.tan(th) + ax2) * ay3
        z[i] = (num/den).item()
    return z


def add_noise(s, p, s_noise_sd, p_noise_sd):
    """
    Add Gaussian noise to sonar and optical measurements.

    Args:
        s: 2D points in {sonar} as (r, theta), shape (2, n)
        p: 2D points in {camera}, shape (2, n)
        s_noise_sd: (sigma_r, sigma_theta), Standard deviation of noise in {sonar}
        p_noise_sd: (sigma_x, sigma_y), Standard deviation of noise in {camera}

    Returns:
        s_n: Noisy sonar measurements, shape (2, n)
        p_n: Noisy optical measurements, shape (2, n)
    """
    s_noise_sd = np.array(s_noise_sd).reshape(2, 1)
    p_noise_sd = np.array(p_noise_sd).reshape(2, 1)
    s_noise = np.random.normal(0, s_noise_sd, s.shape)
    p_noise = np.random.normal(0, p_noise_sd, p.shape)
    s_n = s + s_noise
    p_n = p + p_noise

    # plt.figure()
    # plt.plot(s_n[0, :], label="r noise")
    # plt.plot(s_n[1, :], label="theta noise")
    # plt.plot(s[0, :, ], label="r")
    # plt.plot(s[1, :], label="theta")
    # plt.legend()

    # plt.figure()
    # plt.plot(p_n[0, :], label="x noise")
    # plt.plot(p_n[1, :], label="y noise")
    # plt.plot(p[0, :], label="x")
    # plt.plot(p[1, :], label="y")
    # plt.legend()

    return s_n, p_n


def epipolar_value(s, p, i, K, R, t):
    """
    Compute the epipolar value for a pair of optical and sonar measurements.

    Args:
        p: 2D points in {camera}, shape (2, n)
        s: 2D points in {sonar} as (r, theta), shape (2, n)
        K: Camera intrinsics matrix
        R: Rotation matrix from {camera} to {sonar}
        t: Translation vector from {camera} to {sonar}

    Returns:
        ep: Epipolar value, shape (n,)
    """
    f = K[0,0]
    r = s[0, i]
    th = s[1, i]
    r1 = R[0, :].reshape(1, 3)
    r2 = R[1, :].reshape(1, 3)
    tx = t[0, 0]
    ty = t[1, 0]
    p = np.vstack((p, f*np.ones(p.shape[1])))
    term1 = (np.linalg.norm(t)**2-r**2)*(r1-np.tan(th)*r2).T @ (r1-np.tan(th)*r2)
    term2 = (np.tan(th) * ty - tx)**2 * np.eye(3)
    term3 = 2 * (np.tan(th) * ty - tx) * (r1 - np.tan(th) * r2).T @ t.T @ R
    ep = p[:,i].T @ (term1 + term2 + term3) @ p[:,i]
    return ep


def epipolar_value_parallel(s, p, i, K, R, t):
    """
    Compute the epipolar value for a pair of optical and sonar measurements
    for parallel camera configuration.

    Args:
        p: 2D points in {camera}, shape (2, n)
        s: 2D points in {sonar} as (r, theta), shape (2, n)
        K: Camera intrinsics matrix
        R: Rotation matrix from {camera} to {sonar}
        t: Translation vector from {camera} to {sonar}

    Returns:
        ep: Epipolar value, shape (n,)
    """
    f = K[0,0]
    r = s[0, i]
    th = s[1, i]
    tx = t[0, 0]
    p = np.vstack((p, f*np.ones(p.shape[1])))
    beta = tx/r
    k2 = 1 + np.tan(th)**2
    U = r**2 * np.array([[-1, 0, np.tan(th)], 
                         [0, beta**2, 0], 
                         [np.tan(th), 0, k2 * beta**2 - (np.tan(th))**2]])
    ep = p[:,i].T @ U @ p[:,i]
    return ep
    
    

def maximum_likelihood_estimate(s, p, K, R, t, s_sd, p_sd, lambda_):
    """
    Maximum likelihood estimate of the optical and sonar measurements.

    Args:
        s: Sonar measurements as (r, theta), shape (2, n)
        p: Optical measurements as (x, y), shape (2, n)
        K: Camera intrinsics matrix
        R: Rotation matrix from {camera} to {sonar}
        t: Translation vector from {camera} to {sonar}

    Returns:
        s_mle: 2D points in {sonar} as (r, theta), shape (2, n)
        p_mle: 2D points in {camera}, shape (2, n)
    """

    def compute_residuals(X):
        """
        Args:
            X: (xi, yi, ri, thi), vector to optimize, shape (4,)

        Returns:
            residual: scalar value
        """
        nonlocal i
        xi, yi, ri, thi = X
        ep = epipolar_value_parallel(s, p, i, K, R, t)
        residual = (xi - p[0,i])**2/p_sd[0] + (yi - p[1,i])**2/p_sd[1] \
            + (ri - s[0,i])**2/s_sd[0] + (thi - s[1,i])**2/s_sd[1] \
            + lambda_ * ep
        return residual
    
    p_mle = np.zeros(p.shape)
    s_mle = np.zeros(s.shape)
    print("Running MLE...")
    for i in tqdm(range(s.shape[1])):
        x0 = np.vstack((p[0, i], p[1, i], s[0, i], s[1, i])).reshape(-1) 
        result = least_squares(compute_residuals, x0, method="trf")
        p_mle[:, i] = result.x[:2]
        s_mle[:, i] = result.x[2:]
    return p_mle, s_mle


def reconstruct(p, K, Zo):
    """
    Reconstruction of 3D points from Z-values
    using the optical projection model.
    """
    f = K[0, 0]
    deltax, deltay = K[0, 2], K[1, 2]
    Xo = (p[0, :] - deltax) * Zo / f
    Yo = (p[1, :] - deltay) * Zo / f
    Qo_rec = np.vstack((Xo, Yo, Zo))
    return Qo_rec


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


def plot_3d_reconstruction(Q_true, Q_rec, title, scale=10.):
    """
    Plot the 3D points of the true and reconstructed scene.

    Args:
        Q_true: True 3D points, shape (3, n)
        Q_rec: Reconstructed 3D points, shape (3, n)
        title: Plot title
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(Q_true[0, :], Q_true[1, :], Q_true[2, :], color=((0,0,1,0.5)), label="True")
    ax.scatter(Q_rec[0, :], Q_rec[1, :], Q_rec[2, :], c="r", label="Reconstructed", marker="X")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_zlim(-scale, scale)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(title)


def plot_dual_projection(s, p, K):
    """
    Plot the results of the optical and sonar projection models.

    Args:
        p: 2D points in {camera}, shape (2, n)
        s: 2D points in {sonar} as (r, theta), shape (2, n)
        K: Camera intrinsics matrix
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Projection of camera and sonar")
    ax[0].scatter(p[0, :], p[1, :], s=5)
    ax[0].set_title("Camera frame")
    ax[0].set_xlabel("x (pixels)")
    ax[0].set_ylabel("y (pixels)")
    ax[0].set_aspect("equal")
    ax[0].set_xlim(-K[0, 0]/2 + K[0, 2], K[0, 0]/2 + K[0, 2])
    ax[0].set_ylim(-K[0, 0]/2 + K[1, 2], K[0, 0]/2 + K[1, 2])
    ax[0].set_aspect("equal")
    ax[1].scatter(s[1, :], s[0, :], s=5)
    ax[1].set_title("Sonar frame")
    ax[1].set_xlabel("theta (rad)")
    ax[1].set_ylabel("range (m)")


def main():
    # World config
    d_baseline = 0.1  # baseline dist between {camera} and {sonar}
    d_plane = 10.0 # plane dist
    scale = 100.

    Qw = scale * box3d(n=16)  # 3D points centered around origin of {world}
    # tw = np.array([[0.5, 0., 0.]]).T
    # Qw = Qw + tw      # offset box
    
    # Parallel camera configuration, R and t from the paper
    t_os = np.array([[d_baseline, 0, 0]]).T
    Ros = np.array([[1, 0, 0], 
                  [0, 0, 1], 
                  [0, -1, 0]])  # 90 degree rotation around x-axis

    # Camera extrinsics, {world} to {camera}
    t_wo_x = 0.0
    t_wo_y = -0.0
    t_wo = np.array([[t_wo_x, t_wo_y, d_plane]]).T
    Rwo = np.eye(3)
    Qo = Rwo @ Qw + t_wo  # points in {camera}

    # Camera intrinsics
    f = 800
    delta_x, delta_y = (0,0)
    K = camera_intrinsic(f, (delta_x, delta_y))

    # Sonar extrinsics, {camera} to {sonar}
    Qs = Ros @ Qo + t_os

    # Projection
    p = project_optical(K, Rwo, t_wo, Qw)
    s = project_sonar(Qs)

    # # Check epipolar values should be zero without noise
    # epi = [epipolar_value(s, p, i, K, Ros, t_os) for i in range(s.shape[1])]
    # epi = [epipolar_value_parallel(s, p, i, K, Ros, t_os) for i in range(s.shape[1])]
    # plt.figure()
    # plt.plot(epi)

    # Add noise
    s_noise_sd = (0.01, 0.00175) # (r, theta)
    p_noise_sd = (1, 1) # (x, y)
    s_n, p_n = add_noise(s, p, s_noise_sd, p_noise_sd)

    # Optimize
    lambda_ = 100.  # regularization
    p_mle, s_mle = maximum_likelihood_estimate(s_n, p_n, K, Ros, t_os, s_noise_sd, p_noise_sd, lambda_)

    # Reconstruct
    Zo_azi_mle = azimuth_solution(s_mle, p_mle, K, Ros, t_os)
    Zo_azi = azimuth_solution(s, p, K, Ros, t_os)   # no noise
    Qo_azi_mle = reconstruct(p, K, Zo_azi_mle)
    Qo_azi = reconstruct(p, K, Zo_azi)

    # Zo_range = range_solution(s, p, K, Ros, t_os)
    # Zo_m = m_solution(s, p, Ros, t_os)
    # Qo_range = reconstruct(p, K, Zo_range)
    # Qo_m = reconstruct(p, K, Zo_m)

    # Qo_range_origin = np.mean(Qo_range, axis=1).reshape(-1, 1)
    # Qo_azi_origin = np.mean(Qo_azi, axis=1).reshape(-1, 1)
    # Qo_m_origin = np.mean(Qo_m, axis=1).reshape(-1, 1)

    # Qo_range_align = Qw + Qo_range_origin
    # Qo_azi_align = Qw + Qo_azi_origin
    # Qo_m_align = Qw + Qo_m_origin

    # Transform to {world}
    # Qw_range = np.linalg.inv(Rwo) @ (Qo_range - t_wo)
    Qw_azi = np.linalg.inv(Rwo) @ (Qo_azi - t_wo)
    Qw_azi_mle = np.linalg.inv(Rwo) @ (Qo_azi_mle - t_wo)
    # Qw_m = np.linalg.inv(Rwo) @ (Qo_m - t_wo)

    # Plot
    # plot_3d_reconstruction(Qw, Qw_range, "Range solution in {world}")
    plot_3d_reconstruction(Qw, Qw_azi, "Azimuth solution in {world}", scale)
    plot_3d_reconstruction(Qw, Qw_azi_mle, "MLE Azimuth solution in {world}", scale)
    # plot_3d_reconstruction(Qw, Qw_m, "m solution in {world}")

    # plot_3d_reconstruction(Qo_range_align, Qo_range, "Range solution by aligning box origins")
    # plot_3d_reconstruction(Qo_azi_align, Qo_azi, "Azimuth solution by aligning box origins")
    # plot_3d_reconstruction(Qo_m_align, Qo_m, "M solution by aligning box origins")

    # plot_3d(Qw, "3D points in {world}")
    # plot_3d(Qs, "3D points in {sonar}")
    # plot_dual_projection(s, p, K)
    plt.show()


if __name__ == "__main__":
    main()
