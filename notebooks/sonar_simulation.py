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
    r = np.linalg.norm(Qs, axis=0)  # np.sqrt(x^2 + y^2 + z^2)
    theta = np.arctan2(Qs[0, :], Qs[1, :])  # arctan2(x, y)
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
    z = f * ((range_ * np.linalg.norm(p) - t.T @ R @ p) / np.linalg.norm(p) ** 2)
    return z


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


def plot_2d_sonar(x, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(x[1, :], x[0, :], s=5)
    plt.title(title)
    plt.xlabel("theta (rad)")
    plt.ylabel("range (m)")


def plot_dual_pov(p, Qs_proj, f):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].scatter(p[0, :], p[1, :], s=5)
    ax[0].set_title("Camera frame")
    ax[0].set_xlabel("X (m)")
    ax[0].set_ylabel("Y (m)")
    ax[0].set_xlim(0, f)
    ax[0].set_ylim(0, f)
    ax[0].set_aspect("equal")
    ax[1].scatter(Qs_proj[0, :], Qs_proj[1, :], s=5)
    ax[1].set_title("Optical view from sonar's position")
    ax[1].set_xlabel("X (m)")
    ax[1].set_ylabel("Y (m)")
    ax[1].set_xlim(0, f)
    ax[1].set_ylim(0, f)
    ax[1].set_aspect("equal")


def plot_dual_projection(p, s, f):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].scatter(p[0, :], p[1, :], s=5)
    ax[0].set_title("Camera frame")
    ax[0].set_xlabel("X (m)")
    ax[0].set_ylabel("Y (m)")
    ax[0].set_aspect("equal")
    ax[0].set_xlim(0, f)
    ax[0].set_ylim(0, f)
    ax[0].set_aspect("equal")
    ax[1].scatter(s[1, :], s[0, :], s=5)
    ax[1].set_title("Sonar frame")
    ax[1].set_xlabel("theta (rad)")
    ax[1].set_ylabel("range (m)")


def main():
    Qw = box3d(n=16)  # shape (3, 240)
    # 3D points centered around origin of {world}
    # +x is right, +y is down, +z is forward

    # Camera extrinsics, {camera} to {world}
    distance = 10.0
    tc_y = -0.5
    # theta_x = -15.0  # degrees (pitch down)
    # Rc = Rotation.from_euler("xyz", [theta_x / 180.0, 0, 0]).as_matrix()
    Rc = np.eye(3)
    tc = np.array([[0, tc_y, distance]]).T

    # Camera intrinsics
    f = 800
    deltax = 400
    deltay = 400
    K = camera_intrinsic(f, (deltax, deltay))
    # resolution = 800x800
    # assuming principal point is the middle of the img

    # Parallel camera configuration
    tx = 0.1  # between {camera} and {sonar}
    t = np.array([[tx, 0, 0]]).T
    R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])  # 90 degree rotation around x-axis
    # R = np.eye(3)

    # Define sonar frame ??{sonar} to {world}??
    # Rs = np.eye(3)
    ts = np.array([[0, 0.1, distance]]).T
    Qs = Qw + ts

    Qc = Rc @ Qw + tc  # points in {camera}

    # Projection
    p = project_optical(K, Rc, tc, Qw)
    s = project_sonar(Qs)

    # Filter points with theta more than 150 degrees, plot them in 3D
    # mask = (s[1, :]) > 3
    # inv_mask = np.logical_not(mask)
    # fig = plt.figure(figsize=(6, 6))
    # ax = fig.add_subplot(projection="3d")
    # ax.scatter(Qs[0, mask], Qs[1, mask], Qs[2, mask], c="r")
    # ax.scatter(Qs[0, inv_mask], Qs[1, inv_mask], Qs[2, inv_mask], c="b")
    # ax.set_xlabel("X (m)")
    # ax.set_ylabel("Y (m)")
    # ax.set_zlabel("Z (m)")
    # ax.set_aspect("equal")
    # ax.view_init(elev=-60, azim=-85, roll=0)
    # ax.set_title("3D points in {sonar} with masked points in red")

    # Optical projection of sonar points
    Qs_proj = project_optical(K, np.eye(3), np.array([[0, -0.4, 0]]).T, Qs)

    z_range = range_solution(s, p, K, R, t)

    print(np.min(z_range), np.max(z_range))
    Xo = (p[0, :] - deltax) * z_range / f
    Yo = (p[1, :] - deltay) * z_range / f
    reconstructed = np.vstack((Xo, Yo, z_range))

    # Plot
    plot_3d(Qw, "3D points in {world}")
    # plot_3d(Qs, "3D points in {sonar}")
    plot_dual_pov(p, Qs_proj, f)
    plot_dual_projection(p, s, f)
    plot_3d(reconstructed, "Reconstruction with z_range in {optical}", c="r")
    plt.show()


if __name__ == "__main__":
    main()
