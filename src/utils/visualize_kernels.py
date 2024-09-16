#%%
import cv2 as cv
from typing import List
from matplotlib import cm
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

__all__ = [
    "plot_kernels_inv",
    "plot_kernels_chol",
    "plot_kernel_centers",
    "plot_kernel_centers_inv",
    "shade_kernel_areas",
    "plot_block_with_kernels"
]

# Funtion that upsamples image by factor of n

def _upsample(img, n):
    img = np.repeat(img, n, axis=0)
    img = np.repeat(img, n, axis=1)
    return img

# Function that plots the one sigma contour of a multivariate gaussian distribution
    
def _plot_gaussian_contour(mean: np.ndarray, cov: np.ndarray, ax: plt.Axes, color='r', linewidth: int = 10, alpha: float = 1, res: int = 1000, calls: int = 0) -> None:
    if calls > 10:
        return
    m_x, m_y = mean
    (a,_), (b,c) = cov
    l1 = ((a+c)/2) + (((a-c)/2)**2 + b**2)**0.5
    l2 = ((a+c)/2) - (((a-c)/2)**2 + b**2)**0.5

    if l2 < 0:
        print("Invalid Covariance matrix.")
        print(f"{b=}")
        m = min(np.abs(a), np.abs(c))
        b = np.clip(b, -m, m)
        b *= 0.99
        cov = np.array([
            [a,0],
            [b,c],
        ])
        _plot_gaussian_contour(mean, rectify_covariance_matrix(cov), ax, color=color, linewidth=linewidth, alpha=alpha, res=res, calls=calls+1)

    if b == 0 and a >= c:
        phi = 0
    elif b == 0 and a < c:
        phi = np.pi/2
    else:
        phi = np.arctan2(l1 - a, b)
    phi += np.pi/2
    t = np.linspace(0, 2*np.pi, res)
    x = np.sqrt(l1) * np.cos(phi) * np.cos(t) - np.sqrt(l2) * np.sin(phi) * np.sin(t)
    y = np.sqrt(l1) * np.sin(phi) * np.cos(t) + np.sqrt(l2) * np.cos(phi) * np.sin(t)

    x += m_x
    y += m_y

    ax.plot(x,y, color=color, linewidth=linewidth, alpha=np.clip(alpha, 0, 1))


def plot_kernels_inv(smoe_vector: torch.Tensor, ax: plt.Axes, block_size: int, padding: List[int] = None, n_kernels: int = 4, special_kernel_ind: int = None, colors: list = None) -> None:
    """
    Plots the kernels of a smoe vector in ax.
    """
    if padding is None:
        padding = [0, 0, 0, 0]
    elif type(padding) is int:
        padding = [padding]*4
    smoe_vector = smoe_vector.detach().cpu().numpy()
    block_size -= 1
    means_y = (block_size * smoe_vector[:n_kernels]) + padding[0]
    means_x = (block_size * smoe_vector[n_kernels:2*n_kernels]) + padding[2]
    nus = smoe_vector[2*n_kernels:3*n_kernels]
    covs = smoe_vector[3*n_kernels:].reshape(-1, 2, 2)
    covs = np.tril(covs)
    covs = np.array([np.flip(np.flip(c @ c.T, axis=0).T, axis=0) for c in covs])
    for i, (mx, my, nu, cov) in enumerate(zip(means_x, means_y, nus, covs)):
        if special_kernel_ind is not None and i == special_kernel_ind:
            c = "g"
        elif colors is not None:
            c = colors[i]
        else:
            c = "r"
        _plot_gaussian_contour(np.array([mx, my]), cov, ax, color=c, alpha=np.abs(nu))

def plot_kernels_chol(smoe_vector: torch.Tensor, ax: plt.Axes, block_size: int, padding: List[int] = None, n_kernels: int = 4, special_kernel_ind: int = None, colors: list = None) -> None:
    """
    Plots the kernels of a smoe vector in ax.
    """
    if padding is None:
        padding = [0, 0, 0, 0]
    elif type(padding) is int:
        padding = [padding]*4
    smoe_vector = smoe_vector.detach().cpu().numpy()
    block_size -= 1
    means_y = (block_size * smoe_vector[:n_kernels]) + padding[0]
    means_x = (block_size * smoe_vector[n_kernels:2*n_kernels]) + padding[2]
    nus = smoe_vector[2*n_kernels:3*n_kernels]
    covs = smoe_vector[3*n_kernels:].reshape(-1, 2, 2)
    covs = np.tril(covs)
    covs = np.array([c @ c.T for c in covs])
    for i, (mx, my, nu, cov) in enumerate(zip(means_x, means_y, nus, covs)):
        if special_kernel_ind is not None and i == special_kernel_ind:
            c = "g"
        elif colors is not None:
            c = colors[i]
        else:
            if nu >= 0:
                c = "r"
            else:
                c = "b"
        _plot_gaussian_contour(np.array([mx, my]), cov, ax, color=c, alpha=np.abs(nu))

def interpolate(p_from, p_to, num):
    direction = (p_to - p_from) / np.linalg.norm(p_to - p_from)
    distance = np.linalg.norm(p_to - p_from) / (num - 1)

    ret_vec = []

    for i in range(0, num):
        ret_vec.append(p_from + direction * distance * i)

    return np.array(ret_vec)

def plotImage(ax, img, R, t, size=np.array((1, 1)), img_scale=8, cmap='gray'):
    """
        plot image (plane) in 3D with given Pose (R|t) of corner point

        ax      : matplotlib axes to plot on
        R       : Rotation as roation matrix
        t       : translation as np.array (1, 3), left down corner of image in real world coord
        size    : Size as np.array (1, 2), size of image plane in real world
        img_scale: Scale to bring down image, since this solution needs 1 face for every pixel it will become very slow on big images 
        cmap    : Color map for image
    """
    import cv2 as cv
    img_size = (np.array((img.shape[0], img.shape[1])) / img_scale).astype('int32')
    img = cv.resize(img, ((img_size[1], img_size[0])))

    corners = np.array(([0., 0, 0], [0, size[0], 0],
                        [size[1], 0, 0], [size[1], size[0], 0]))

    corners += t
    corners = corners @ R
    xx = np.zeros((img_size[0], img_size[1]))
    yy = np.zeros((img_size[0], img_size[1]))
    zz = np.zeros((img_size[0], img_size[1]))
    l1 = interpolate(corners[0], corners[2], img_size[0])
    xx[:, 0] = l1[:, 0]
    yy[:, 0] = l1[:, 1]
    zz[:, 0] = l1[:, 2]
    l1 = interpolate(corners[1], corners[3], img_size[0])
    xx[:, img_size[1] - 1] = l1[:, 0]
    yy[:, img_size[1] - 1] = l1[:, 1]
    zz[:, img_size[1] - 1] = l1[:, 2]

    for idx in range(0, img_size[0]):
        p_from = np.array((xx[idx, 0], yy[idx, 0], zz[idx, 0]))
        p_to = np.array((xx[idx, img_size[1] - 1], yy[idx, img_size[1] - 1], zz[idx, img_size[1] - 1]))
        l1 = interpolate(p_from, p_to, img_size[1])
        xx[idx, :] = l1[:, 0]
        yy[idx, :] = l1[:, 1]
        zz[idx, :] = l1[:, 2]

    if img.max() > 1:
        img = img / 255


    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, facecolors=np.stack(3*[img]).transpose(1, 2, 0), shade=False, cmap=cm.get_cmap(cmap), alpha=0.8, zorder=0)
    return None

def shade_kernel_areas(smoe_vector: torch.Tensor, ax: plt.Axes, block_size: int, image, padding: List[int] = None, n_kernels: int = 4, special_kernel_ind: int = None, colors: list = None) -> None:
    if padding is None:
        padding = [0, 0, 0, 0]
    smoe_vector = smoe_vector.detach().cpu().numpy()
    block_size -= 1
    means_x = (block_size * smoe_vector[:n_kernels]) + padding[0]
    means_y = (block_size * smoe_vector[n_kernels:2*n_kernels]) + padding[2]
    nus = smoe_vector[2*n_kernels:3*n_kernels]
    covs = smoe_vector[3*n_kernels:].reshape(-1, 2, 2)
    covs = np.tril(covs)
    covs = np.array([c @ c.T for c in covs])
    
    from scipy.stats import multivariate_normal
    _vars = [multivariate_normal(mean=mean, cov=cov) for mean, cov in zip(zip(means_x, means_y), covs)]
    x = np.linspace(0, block_size, 100)
    y = np.linspace(0, block_size, 100)
    X, Y = np.meshgrid(x, y)

    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    Z = [v.pdf(pos) for v in _vars]
    print(means_x, means_y, nus)
    # make plot interactive
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.view_init(elev=75, azim=75, roll=180)
    R_rad = np.array((0.0, 0.0, 0.0)) * np.pi / 180
    R = cv.Rodrigues(R_rad)[0]
    plotImage(ax, image.cpu().flip(0).numpy(), R, np.array([0., 0., 0.05]), size=np.array([block_size, block_size]), img_scale=1, cmap='gray')
    for z in Z:
        ax.plot_surface(X, Y, z, cmap='viridis', edgecolor='none', alpha=0.8, zorder=100)

    plt.figure()
    plt.imshow(image.cpu(), cmap='gray')

def plot_kernel_centers_inv(smoe_vector: torch.Tensor, ax: plt.Axes, block_size: int, padding: List[int] = None, n_kernels: int = 4, special_kernel_ind: int = None, colors: list = None) -> None:
    """
    Plots the kernels of a smoe vector in ax.
    """
    if padding is None:
        padding = [0, 0, 0, 0]
    elif type(padding) is int:
        padding = [padding]*4
    smoe_vector = smoe_vector.detach().cpu().numpy()
    block_size -= 1
    means_y = (block_size * smoe_vector[:n_kernels]) + padding[0]
    means_x = (block_size * smoe_vector[n_kernels:2*n_kernels]) + padding[2]
    for i, (mx, my) in enumerate(zip(means_x, means_y)):
        if special_kernel_ind is not None and i == special_kernel_ind:
            c = "g"
        elif colors is not None:
            c = colors[i]
        else:
            c = "r"
        ax.scatter(mx, my, color=c, marker="x", s=50)

def plot_kernel_centers(smoe_vector: torch.Tensor, ax: plt.Axes, block_size: int, padding: List[int] = None, n_kernels: int = 4, special_kernel_ind: int = None, colors: list = None) -> None:
    """
    Plots the kernels of a smoe vector in ax.
    """
    if padding is None:
        padding = [0, 0, 0, 0]
    elif type(padding) is int:
        padding = [padding]*4
    smoe_vector = smoe_vector.detach().cpu().numpy()
    block_size -= 1
    means_x = (block_size * smoe_vector[:n_kernels]) + padding[0]
    means_y = (block_size * smoe_vector[n_kernels:2*n_kernels]) + padding[2]
    for i, (mx, my) in enumerate(zip(means_x, means_y)):
        if special_kernel_ind is not None and i == special_kernel_ind:
            c = "g"
        elif colors is not None:
            c = colors[i]
        else:
            c = "r"
        ax.scatter(mx, my, color=c, marker="x", s=50)


def rectify_covariance_matrix(cov):
    """
    Rectify a covariance matrix to ensure it is symmetric and positive semidefinite.

    Parameters:
    cov (np.array): A covariance matrix that might be invalid.

    Returns:
    np.array: A rectified, valid covariance matrix.
    """
    if (cov[0,0] < 0) and (cov[1,1]) < 0 and (cov[1,0] > 0):
        cov = -cov

    # Ensure the matrix is symmetric
    if cov[0,1] == 0:
        cov[0,1] += cov[1,0]
        cov_sym = cov
    else:
        cov_sym = (cov + cov.T) / 2

    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_sym)

    # Set any negative eigenvalues to a small positive value (e.g., 1e-10)
    eigenvalues_rectified = np.clip(eigenvalues, 1e-9, None)

    # Reconstruct the matrix
    cov_rectified = eigenvectors @ np.diag(eigenvalues_rectified) @ eigenvectors.T

    return cov_rectified

def plot_block_with_kernels(smoe_vector, block_img: torch.Tensor, n_kernels: int = 4, block_size: int = 8):
    plt.imshow(block_img.transpose(0,1).detach().cpu(), cmap='gray', vmin=0, vmax=1)
    plot_kernel_centers(smoe_vector, plt.gca(), block_size=block_size, n_kernels=n_kernels)
    plot_kernels_chol(smoe_vector, plt.gca(), block_size=block_size, n_kernels=n_kernels)

if __name__ == "__main__":
    # fig, ax = plt.subplots()
    # for b in np.arange(-1, 1.1, .02):
    #     plot_gaussian_contour(np.array([0,0]), np.array([[1,b], [b,1]]), ax)

    # x_lim, y_lim = plt.xlim(), plt.ylim()
    # xy_lim = max(*x_lim, *y_lim)
    # plt.xlim(-xy_lim, xy_lim)
    # plt.ylim(-xy_lim, xy_lim)
    img = np.random.uniform(0, 1, (16,16))
    n = 100
    # img = _upsample(img, n)
    mean = np.array([7.5,7.5])
    b = 1
    cov = np.array([
        [-4,0],
        [b,-2],
        ]
    )
    fig, ax = plt.subplots()
    ax: plt.Axes
    ax.imshow(img)
    _plot_gaussian_contour(mean, rectify_covariance_matrix(cov), ax, alpha=0.4)
# %%
