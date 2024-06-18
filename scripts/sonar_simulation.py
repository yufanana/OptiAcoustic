#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from scipy.spatial.transform import Rotation


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


def reconstruct(p, K, Zo):
    """
    Reconstruction of 3D points from Z-values.
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


def plot_3d_reconstruction(Q_true, Q_rec, title):
    """
    Plot the 3D points of the true and reconstructed scene.

    Args:
        Q_true: True 3D points, shape (3, n)
        Q_rec: Reconstructed 3D points, shape (3, n)
        title: Plot title
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(Q_rec[0, :], Q_rec[1, :], Q_rec[2, :], c="r", label="Reconstructed", marker="x")
    ax.scatter(Q_true[0, :], Q_true[1, :], Q_true[2, :], c="b", label="True")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(title)


def plot_2d_sonar(x, title):
    """
    Plot 2D sonar measurements with range and azimuth.

    Args:
        x: 2D points in {sonar} as (r, theta), shape (2, n)
        title: Plot title
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(x[1, :], x[0, :], s=5)
    plt.title(title)
    plt.xlabel("theta (rad)")
    plt.ylabel("range (m)")


def plot_dual_projection(p, s, K):
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

    Qw = box3d(n=16)  # 3D points centered around origin of {world}
    # tw = np.array([[0.5, 0., 0.]]).T
    # Qw = Qw + tw      # offset box
    
    # Parallel camera configuration, R and t from the paper
    t_os = np.array([[d_baseline, 0, 0]]).T
    Ros = np.array([[1, 0, 0], 
                  [0, 0, 1], 
                  [0, -1, 0]])  # 90 degree rotation around x-axis

    # Camera extrinsics, {world} to {camera}
    t_wo_x = 0.0
    t_wo_y = -0.5
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

    # Reconstruct
    Zo_range = range_solution(s, p, K, Ros, t_os)
    Zo_azi = azimuth_solution(s, p, K, Ros, t_os)
    Zo_m = m_solution(s, p, Ros, t_os)
    Qo_range = reconstruct(p, K, Zo_range)
    Qo_azi = reconstruct(p, K, Zo_azi)
    Qo_m = reconstruct(p, K, Zo_m)

    # noise_level = 0.01  # adjust the noise level as desired
    # noise = np.random.normal(0, noise_level, Qo_azi.shape)
    # Qo_azi_noisy = Qo_azi + noise
    
    Qo_range_origin = np.mean(Qo_range, axis=1).reshape(-1, 1)
    Qo_azi_origin = np.mean(Qo_azi, axis=1).reshape(-1, 1)
    Qo_m_origin = np.mean(Qo_m, axis=1).reshape(-1, 1)
    Qo_range_align = Qw + Qo_range_origin
    Qo_azi_align = Qw + Qo_azi_origin
    Qo_m_align = Qw + Qo_m_origin

    # Transform to {world}
    Qw_range = np.linalg.inv(Rwo) @ (Qo_range - t_wo)
    Qw_azi = np.linalg.inv(Rwo) @ (Qo_azi - t_wo)
    Qw_m = np.linalg.inv(Rwo) @ (Qo_m - t_wo)

    plot_3d_reconstruction(Qw, Qw_range, "Range solution in {world}")
    plot_3d_reconstruction(Qw, Qw_azi, "Azimuth solution in {world}")
    plot_3d_reconstruction(Qw, Qw_m, "m solution in {world}")
    plot_3d_reconstruction(Qo_range_align, Qo_range, "Range solution by aligning box origins")
    plot_3d_reconstruction(Qo_azi_align, Qo_azi, "Azimuth solution by aligning box origins")
    # plot_3d_reconstruction(Qo_m_align, Qo_m, "Reconstruction of z_range and z_azi")

    # Plot
    # plot_3d(Qw, "3D points in {world}")
    # plot_3d(Qs, "3D points in {sonar}")
    plot_dual_projection(p, s, K)
    plt.show()


if __name__ == "__main__":
    main()
