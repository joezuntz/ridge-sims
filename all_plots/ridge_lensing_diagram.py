import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Ellipse
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
})



def draw_bar_with_cap(x0, y0, x1, y1, capsize=0.05):
    # line from (x0, y0) to (x1, y1)
    plt.plot([x0, x1], [y0, y1], 'k-', lw=2)

    # line perpendicular to that line through (x0, y0)
    dx = x1 - x0
    dy = y1 - y0
    length = np.sqrt(dx**2 + dy**2)
    dx /= length
    dy /= length
    plt.plot([x0 - dy*capsize, x0 + dy*capsize], [y0 + dx*capsize, y0 - dx*capsize], 'k-', lw=2)
    plt.plot([x1 - dy*capsize, x1 + dy*capsize], [y1 + dx*capsize, y1 - dx*capsize], 'k-', lw=2)

# plt.figure(figsize=(4, 4))
fig, axes = plt.subplots(2, 1, figsize=(4, 8))

ax = axes[0]
x_ridge = 5*np.array([0.2, 0.25, 0.3, 0.32, 0.4, 0.45, 0.50, 0.52, 0.54, 0.56, 0.6, 0.62, 0.65, 0.7])
x1 = np.linspace(x_ridge[0], x_ridge[-1], 1000)
y1 = x1/5 + np.sin(x1*2)
ax.plot(x1, y1, '-', color="gray", label="True ridge")
y_ridge = x_ridge/5 + np.sin(x_ridge*2)
ax.plot(x_ridge, y_ridge, '.', markersize=10, color="blue", label="Located rige points")

x_bg = np.array([1, 2.0, 2.8,  3.8])
y_bg = np.array([-0.5, -1.0, 0.75,  1.6])

# find the nearest ridge point for each bg point
for i, (xb, yb) in enumerate(zip(x_bg, y_bg)):
    distances = np.sqrt((x_ridge - xb)**2 + (y_ridge - yb)**2)
    min_index = np.argmin(distances)
    xr_near = x_ridge[min_index]
    yr_near = y_ridge[min_index]

    # Line from background point to nearest ridge point
    ax.plot([xr_near, xb], [yr_near, yb], 'k--', lw=1)

    if i == 1:
        angle = 0
    else:
        angle = np.random.uniform(0, 360)

    bg = Ellipse((xb, yb), width=0.2, height=0.1, angle=angle, facecolor="orange", zorder=5)
    ax.add_artist(bg)



# ax.plot(x_bg, y_bg, 'o', markersize=10, color="orange", label="Background galaxies")
ax.legend(loc="upper center", frameon=False)
ax.axis('equal')
ax.set_xticks([])
ax.set_yticks([])
# plt.savefig("ridge_bg_example.pdf", bbox_inches="tight")
# plt.show()

# draw a box shpwing the region of the lower plot
# rect = plt.Rectangle((1.75, -1.25), 0.77, 1.07, edgecolor='red', facecolor='none', lw=2)
# ax.add_patch(rect)
# (np.float64(1.7650000000000001), np.float64(2.535)) (np.float64(-1.248407105452864), np.float64(-0.18345078548985486))


ax = axes[1]

# second diagram - a zoom in on one of the background points
# x1 = np.linspace(1.9, 2.5, 1000)
# y1 = x1/5 + np.sin(x1*2)
ax.plot(x1, y1, '-', color="gray", label="True ridge")
xr = 0.45*5
yr = xr/5 + np.sin(xr*2)
# ax.plot([xr], [yr], 'o', markersize=5, color="blue", label="Located rige point")
ax.plot(x_ridge, y_ridge, 'o', markersize=5, color="blue", label="Located rige point")

x_bg = np.array([2.0, ])
y_bg = np.array([-1.0])

bg = Ellipse((x_bg, y_bg), width=0.1, height=0.05, facecolor="orange", zorder=5)
ax.add_artist(bg)


# line from bg point to ridge point

dx1 = x_bg[0] - xr
dy1 = y_bg[0] - yr
length1 = np.sqrt(dx1**2 + dy1**2)
dx1 /= length1
dy1 /= length1
ax.plot([xr, x_bg[0] + dx1*0.2], [yr,  y_bg[0]+dy1*0.2], 'k--', lw=1)

# line perpendicular to that line through background point
dx = xr - x_bg[0]
dy = yr - y_bg[0]
length = np.sqrt(dx**2 + dy**2)
dx /= length
dy /= length
ax.plot([x_bg[0] - dy*0.2, x_bg[0] + dy*0.2], [y_bg[0] + dx*0.2, y_bg[0] - dx*0.2], 'k--', lw=1)


# vertical line through background point
ax.plot([x_bg[0], x_bg[0]], [y_bg[0] - 0.2, y_bg[0] + 0.4], 'k-', lw=1)
# horizontal line through background point
ax.plot([x_bg[0] - 0.2, x_bg[0] + 0.2], [y_bg[0], y_bg[0]], 'k-', lw=1)

theta1 = np.arctan2(yr - y_bg[0], xr - x_bg[0]) * 180 / np.pi

arc = Arc((x_bg[0], y_bg[0]), width=0.7, height=0.7, theta1=theta1, theta2=90, color="k", lw=3)
ax.add_artist(arc)

# add "theta" text in the middle of the arc
angle = (theta1 + 90) / 2
x_text = x_bg[0] + 0.2 * np.cos(np.radians(angle))
y_text = y_bg[0] + 0.2 * np.sin(np.radians(angle))
ax.text(x_text, y_text, r'$\phi$', fontsize=16, color="k", ha='center', va='center')


# draw a measurement bar along the line from the bg point to the ridge point
xm = x_bg[0] + 0.075
ym = y_bg[0] + 0.025
# ax.plot([xm, xm+dx*length*0.85], [ym, ym+dy*length*0.85], 'k-', lw=2)
draw_bar_with_cap(xm, ym, xm+dx*length*0.85, ym+dy*length*0.85, capsize=0.025)
# draw caps on the measurement bar, perpendicular to the bar
# ax.plot([xm - dy*0.025, xm + dy*0.025], [ym + dx*0.05, ym - dx*0.025], 'k-', lw=2)
# ax.plot([xm + dx*length*0.85 - dy*0.025, xm + dx*length*0.85 + dy*0.025], [ym + dy*length*0.85 + dx*0.025, ym + dy*length*0.85 - dx*0.05], 'k-', lw=2)


ax.text(xm+dx*length*0.45+0.05, ym+dy*length*0.45-0.05, r'$\theta$', fontsize=16, color="k", ha='center', va='center')

ax.axis('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(1.765, 2.535)
ax.set_ylim(-1.25, -0.18)


plt.subplots_adjust(hspace=0.05)
plt.savefig("ridge_lensing_diagram.pdf", bbox_inches="tight")