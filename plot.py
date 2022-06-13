"""
File: plot.py

Description: A Python file for plotting figures
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator
from src.library.paramCells.basis import *

# xCoords = np.linspace(-1, 1, num=10)
# yCoords = np.linspace(-1, 1, num=10)
# x2d, y2d = np.meshgrid(xCoords, yCoords)
# xyCoords = np.column_stack((y2d.ravel(), x2d.ravel()))
#
# value2d = GetLegendre2dGrad(xyCoords, 1, 1)
# uCoeffs = np.array([0, 0, 0, 1])
# plotBase = np.matmul(value2d[:, :, 1], uCoeffs)
# plotBase = np.reshape(plotBase.transpose(), (y2d.shape[0], x2d.shape[0]))
# # tensorialBase = np.reshape(value2d.transpose()[i], (y2d.shape[0], x2d.shape[0]))
#
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(7.2, 4.8), dpi=100)
#
# # Plot the surface.
# surf = ax.plot_surface(x2d, y2d, plotBase, cmap='coolwarm', linewidth=0, antialiased=False)
# color_tuple = (1.0, 1.0, 1.0, 0.0)
#
# # Customize the axes.
# ax.set_xlim(1, -1)
# ax.set_ylim(-1, 1)
# ax.xaxis.set_major_locator(LinearLocator(5))
# ax.yaxis.set_major_locator(LinearLocator(5))
# ax.zaxis.set_major_locator(LinearLocator(10))
# # ax.w_xaxis.set_pane_color(color_tuple)
# # ax.w_xaxis.line.set_color(color_tuple)
# # ax.w_yaxis.set_pane_color(color_tuple)
# # ax.w_yaxis.line.set_color(color_tuple)
# # ax.w_zaxis.set_pane_color(color_tuple)
# # ax.w_zaxis.line.set_color(color_tuple)
#
# # A StrMethodFormatter is used automatically
# ax.xaxis.set_major_formatter('{x:.01f}')
# ax.yaxis.set_major_formatter('{x:.01f}')
# ax.zaxis.set_major_formatter('{x:.02f}')
# # ax.set_xticks([])
# # ax.set_yticks([])
# # ax.set_zticks([])
# # ax.view_init(45, -45)
#
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()

# %%
xCoords = np.linspace(-1, 1, num=10)
yCoords = np.linspace(-1, 1, num=10)
x2d, y2d = np.meshgrid(xCoords, yCoords)
xyCoords = np.column_stack((y2d.ravel(), x2d.ravel()))

value2d = GetLegendre2d(xyCoords, 1, 1)
uCoeffs = np.array([0, 0, 1, 1])
plotBase = np.matmul(value2d, uCoeffs)
plotBase = np.reshape(plotBase.transpose(), (y2d.shape[0], x2d.shape[0]))
# tensorialBase = np.reshape(value2d.transpose()[i], (y2d.shape[0], x2d.shape[0]))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6.4, 4.8), dpi=100)

# Plot the surface.
surf = ax.plot_surface(x2d, y2d, plotBase, cmap='coolwarm', linewidth=0, antialiased=False)
color_tuple = (1.0, 1.0, 1.0, 0.0)

# Customize the axes.
ax.set_xlim(1, -1)
ax.set_ylim(-1, 1)
ax.xaxis.set_major_locator(LinearLocator(5))
ax.yaxis.set_major_locator(LinearLocator(5))
ax.zaxis.set_major_locator(LinearLocator(10))
# ax.w_xaxis.set_pane_color(color_tuple)
# ax.w_xaxis.line.set_color(color_tuple)
# ax.w_yaxis.set_pane_color(color_tuple)
# ax.w_yaxis.line.set_color(color_tuple)
# ax.w_zaxis.set_pane_color(color_tuple)
# ax.w_zaxis.line.set_color(color_tuple)

# A StrMethodFormatter is used automatically
ax.xaxis.set_major_formatter('{x:.01f}')
ax.yaxis.set_major_formatter('{x:.01f}')
ax.zaxis.set_major_formatter('{x:.02f}')
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
# ax.view_init(45, -45)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
name = 'example0011.eps'
plt.savefig(name, dpi=1000)
plt.show()

# %%
# for i in range(9):
#     xCoords = np.linspace(-1, 1, num=10)
#     yCoords = np.linspace(-1, 1, num=10)
#     x2d, y2d = np.meshgrid(xCoords, yCoords)
#     xyCoords = np.column_stack((y2d.ravel(), x2d.ravel()))
#
#     value2d = GetLegendre2d(xyCoords, 2, 2)
#     # nodeCoords = np.linspace(-1, 1, num=5)
#     # value2d = GetLagrange2d(xyCoords, nodeCoords, nodeCoords)
#     tensorialBase = np.reshape(value2d.transpose()[i], (y2d.shape[0], x2d.shape[0]))
#
#     fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(4.8, 4.8), dpi=100)
#
# # Plot the surface. surf = ax.plot_surface(x2d, y2d, tensorialBase, cmap='coolwarm', linewidth=0,
# antialiased=False, vmin=-1.0, vmax=1.0) color_tuple = (1.0, 1.0, 1.0, 0.0)
#
#     # Customize the axes.
#     ax.set_xlim(1, -1)
#     ax.set_ylim(-1, 1)
#     # ax.xaxis.set_major_locator(LinearLocator(5))
#     # ax.yaxis.set_major_locator(LinearLocator(5))
#     # ax.zaxis.set_major_locator(LinearLocator(10))
#     ax.w_xaxis.set_pane_color(color_tuple)
#     ax.w_xaxis.line.set_color(color_tuple)
#     ax.w_yaxis.set_pane_color(color_tuple)
#     ax.w_yaxis.line.set_color(color_tuple)
#     ax.w_zaxis.set_pane_color(color_tuple)
#     ax.w_zaxis.line.set_color(color_tuple)
#
#     # A StrMethodFormatter is used automatically
#     # ax.xaxis.set_major_formatter('{x:.01f}')
#     # ax.yaxis.set_major_formatter('{x:.01f}')
#     # ax.zaxis.set_major_formatter('{x:.02f}')
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_zticks([])
#     ax.view_init(45, -45)
#
#     # Add a color bar which maps values to colors.
#     # fig.colorbar(surf, shrink=0.5, aspect=5)
#
#     # plt.contour(x2d, y2d, test, 100, cmap='plasma')
#     # plt.pcolormesh(x2d, y2d, test, cmap='plasma')
#     # plt.colorbar()
#     name = 'legendre{0}.eps'.format(i)
#     # plt.savefig(name, dpi=1000)
#     plt.show()

# %%
"""
Monomials in 1D
"""
p = 5
value = GetMonomial1d(xCoords, p)

for i in range(p+1):
    plt.plot(xCoords, value.transpose()[i])

plt.rc('axes', linewidth=1.25)
# plt.legend(['$\Phi_0$', '$\Phi_1$', '$\Phi_2$', '$\Phi_3$', '$\Phi_4$', '$\Phi_5$'], loc="upper left")
# plt.xticks([-1, 0, 1])
# plt.tick_params(axis='x', direction='in', pad=5)
# plt.yticks([-1, 0, 1])
# plt.tick_params(axis='y', direction='in', pad=5)
# plt.grid()
plt.show()

# %%
"""
Legendre polynomials in 1D
"""
p = 3
value = GetLegendre1d(xCoords, p)

for i in range(p+1):
    plt.plot(xCoords, value.transpose()[i])

plt.show()

# %%
"""
Lagrange polynomials in 1D
"""
nodeCoords = np.linspace(-1, 1, num=5)
nodeCoords.shape = -1, 1
# value = np.zeros((xCoords.shape[0], nodeCoords.shape[0]))
value = GetLagrange1d(xCoords, nodeCoords)

for i in range(len(nodeCoords)):
    plt.plot(xCoords, value.transpose()[i])

plt.show()
