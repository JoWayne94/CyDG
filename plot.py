"""
File: plot.py

Description: A Python file for plotting figures
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator
from src.library.paramCells.basis import *
from src.library.geometries.quadrilateral import Quadrilateral as QuadGeom


# %%
""" Read dat files and plot """
N = [25, 50, 100, 200, 400, 800]
line_style = ["o", "x", "^", "s", "*", "+"]
fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)

for i in range(6):
    # Read in file
    coords = open("/Users/jwtan/PycharmProjects/PyDG/data/ADR/pure_advection_sine_wave_P1_T1.0_N{0}_coords.dat".format(
        str(N[i]))).read()
    values = open("/Users/jwtan/PycharmProjects/PyDG/data/ADR/pure_advection_sine_wave_P1_T1.0_N{0}_values.dat".format(
        str(N[i]))).read()

    # Split the file into tokens
    coords = coords.split()
    values = values.split()
    coords = [float(x) for x in coords]
    values = [float(x) for x in values]

    ax.plot(coords, values, linestyle="", marker=line_style[i], label="N = {0}".format(str(N[i])), markersize=1.75,
            color="black")

exact_x = np.linspace(0., 1., 100)
exact_y = np.sin(2 * np.pi * exact_x)
ax.plot(exact_x, exact_y, linestyle="-", linewidth=1.5, label="Exact solution", color="blue")
ax.xaxis.set_major_formatter('{x:.01f}')
ax.yaxis.set_major_formatter('{x:.01f}')
ax.title.set_text("Sine wave, final time = 1.0 s")
plt.grid()
plt.legend()
plt.savefig('/Users/jwtan/Desktop/dgProject/Plots/pure_advection_sine_wave_P1_T1.eps', dpi=1000)
plt.show()


# %%
""" Plot velocity field """
x = np.linspace(-1, 1, 15)
y = np.linspace(-1, 1, 15)
X, Y = np.meshgrid(x, y)
U = -Y
V = X

fig, ax = plt.subplots(figsize=(4.8, 4.8))
q = ax.quiver(X, Y, U, V)
ax.quiverkey(q, X=0.4, Y=1.05, U=1, label='Velocity field', labelpos='E')
ax.xaxis.set_major_formatter('{x:.01f}')
ax.yaxis.set_major_formatter('{x:.01f}')
plt.savefig("velocity_field", dpi=1000)
plt.show()

# %%
""" Define x and y-coordinates in parametric space """
xCoords = np.linspace(-1, 1, num=10)
yCoords = np.linspace(-1, 1, num=10)
x2d, y2d = np.meshgrid(xCoords, yCoords)
xyCoords = np.column_stack((y2d.ravel(), x2d.ravel()))

""" Define a quadrilateral geometry with integration points equal to x, y-coordinates above """
quad = QuadGeom(np.array([0, 1, 2, 3]), np.array([[0, 0], [2, 0], [2, 1], [0, 1]]),
                np.array([i[0] for i in xyCoords]), np.array([i[1] for i in xyCoords]))

""" Get 2d basis values in parametric space """
# Basis values
value2d = Legendre2d(xyCoords, 1, 1)
# Basis gradients
# value2d = Legendre2dGrad(xyCoords, 1, 1)

""" Define solution coefficients """
uCoeffs = np.array([0, 0, 0, 1])

""" Approximate solution, u_h = B uCoeffs """
# Approximate solution in real space
plotBase = np.matmul(value2d, uCoeffs)
# Approximate solution in parametric space
# plotBase = np.matmul(value2d[:, :, 1], uCoeffs)
plotBase = np.reshape(plotBase.transpose(), (y2d.shape[0], x2d.shape[0]))

""" Define new coordinates in real space """
# x2dnew, y2dnew = np.meshgrid(quad.parametricMapping()[0], quad.parametricMapping()[1])
x2dnew = np.reshape(quad.parametricMapping()[0], (x2d.shape[0], y2d.shape[0])).transpose()
y2dnew = np.reshape(quad.parametricMapping()[1], (x2d.shape[0], y2d.shape[0])).transpose()

""" Figure parameters """
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6.4, 4.8), dpi=100)

# Plot the surface
# surf = ax.plot_surface(x2d, y2d, plotBase, cmap='coolwarm', linewidth=0, antialiased=False)
surf = ax.plot_surface(x2dnew, y2dnew, plotBase, cmap='coolwarm', linewidth=0, antialiased=False)
color_tuple = (1.0, 1.0, 1.0, 0.0)

# Customize the axes
ax.set_xlim(1, 0)
ax.set_ylim(0, 1)
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

# Add a color bar which maps values to colours
fig.colorbar(surf, shrink=0.5, aspect=5)
name = 'example.eps'
plt.savefig(name, dpi=1000)
plt.show()

# %%
# for i in range(9):
#     xCoords = np.linspace(-1, 1, num=10)
#     yCoords = np.linspace(-1, 1, num=10)
#     x2d, y2d = np.meshgrid(xCoords, yCoords)
#     xyCoords = np.column_stack((y2d.ravel(), x2d.ravel()))
#
#     value2d = Legendre2d(xyCoords, 2, 2)
#     # nodeCoords = np.linspace(-1, 1, num=5)
#     # value2d = Lagrange2d(xyCoords, nodeCoords, nodeCoords)
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
# p = 5
# value = GetMonomial1d(xCoords, p)
#
# for i in range(p + 1):
#     plt.plot(xCoords, value.transpose()[i])
#
# plt.rc('axes', linewidth=1.25)
# plt.legend(['$\Phi_0$', '$\Phi_1$', '$\Phi_2$', '$\Phi_3$', '$\Phi_4$', '$\Phi_5$'], loc="upper left")
# plt.xticks([-1, 0, 1])
# plt.tick_params(axis='x', direction='in', pad=5)
# plt.yticks([-1, 0, 1])
# plt.tick_params(axis='y', direction='in', pad=5)
# plt.grid()
# plt.show()

# %%
"""
Legendre polynomials in 1D
"""
# p = 3
# value = GetLegendre1d(xCoords, p)
#
# for i in range(p + 1):
#     plt.plot(xCoords, value.transpose()[i])
#
# plt.show()

# %%
"""
Lagrange polynomials in 1D
"""
# nodeCoords = np.linspace(-1, 1, num=5)
# nodeCoords.shape = -1, 1
# # value = np.zeros((xCoords.shape[0], nodeCoords.shape[0]))
# value = GetLagrange1d(xCoords, nodeCoords)
#
# for i in range(len(nodeCoords)):
#     plt.plot(xCoords, value.transpose()[i])
#
# plt.show()
