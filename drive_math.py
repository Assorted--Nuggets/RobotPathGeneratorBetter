import matplotlib.pyplot as plt
import numpy as np
import bezier
from matplotlib.widgets import Slider


def vel(s):
    return np.ones(s.shape)*10


def delta(vals):
    return vals[1:] - vals[:-1]


def integrate(f, domain, is_delta=True):
    output = f(domain)
    if is_delta:
        return np.dot(np.transpose(output)[:-1], delta(domain))
    return np.sum(output)


def arc_length_domain(x, y):
    points = np.array([delta(x), delta(y)])
    ds = metric(points)
    return np.cumsum(ds)


def angle_domain(x, y):
    return np.arctan(delta(y)/delta(x))


def derivative(domain, codomain):
    res = delta(codomain)/delta(domain)
    for i in range(0, len(res)):
        if abs(res[i]) > 100:
            res[i] = (res[i-1] + res[i+1])/2

    return res


def gen_curvature_profile(path_x, path_y, velocity_profile, res=1000):
    angles = np.reshape(angle_domain(path_x, path_y), (res - 1, 1))
    distance = np.reshape(arc_length_domain(path_x, path_y), (res - 1, 1))

    ad = np.array([distance.T, np.degrees(angles.T)])[:, 0, :]
    np.savetxt("angles.csv", ad.T)
    velocities = np.resize(velocity_profile(distance), (res - 2, 1))

    result = np.array([np.resize(distance, (res - 2, 1)),
                       velocities,
                       velocities*(derivative(distance, angles).reshape((res - 2, 1)))])
    result = np.reshape(np.transpose(result), (res - 2, 3))

    return result


def gen_wheel_vels(curve_profile, wheel_width):
    differential = curve_profile[:, 2]/(wheel_width*2)
    return np.array([curve_profile[:, 0],
                     curve_profile[:, 1] - differential,
                     curve_profile[:, 1] + differential]).T


def metric(v):
    return (v[0]**2 + v[1]**2)**0.5


def setmarkerpos(marker, pos):
    marker[0].set_xdata(pos[0])
    marker[0].set_ydata(pos[1])


def update(val):
    n = np.floor(val * 1000)
    pos = bez(val)
    cpos = [curvature_profile[int(n), 0], curvature_profile[int(n), 2]]
    lpos = [wheelvels[int(n), 0], wheelvels[int(n), 1]]
    rpos = [wheelvels[int(n), 0], wheelvels[int(n), 2]]
    setmarkerpos(pos_marker, pos)
    setmarkerpos(curve_marker, cpos)
    setmarkerpos(left_marker, lpos)
    setmarkerpos(right_marker, rpos)
    fig.canvas.draw_idle()


points = np.array([[0, 0],
                       [0, 60],
                       [60, 0],
                       [30, 60],
                        [60, -30],
                        [90, 60]])

curve, bez = bezier.build_bezier(points)

curvature_profile = gen_curvature_profile(curve[0], curve[1], vel)
wheelvels = gen_wheel_vels(curvature_profile, 24)
np.savetxt("testpath.csv", wheelvels)
np.savetxt("cp.csv", curvature_profile)
gs_kw = dict(width_ratios=[1,1], height_ratios=[4,4,1])
fig, axs = plt.subplots(3, 2, gridspec_kw=gs_kw, figsize=(5, 5))

axs[0][0].plot(curvature_profile[:, 0], curvature_profile[:, 2])
axs[0][0].set_xlabel('distance')
axs[1][0].plot(wheelvels[:, 0], wheelvels[:, 1])
axs[1][1].plot(wheelvels[:, 0], wheelvels[:, 2])
axs[0][1].plot(curve[0], curve[1])


t = 0.5
n = np.floor(t * 1000)

pos_marker = axs[0][1].plot(bez(t)[0], bez(t)[1], marker='o', markersize=3, color="red")
curve_marker = axs[0][0].plot(curvature_profile[int(n), 0], curvature_profile[int(n), 2], marker='o', markersize=3, color="red")
left_marker = axs[1][0].plot(wheelvels[int(n), 0], wheelvels[int(n), 1], marker='o', markersize=3, color="red")
right_marker = axs[1][1].plot(wheelvels[int(n), 0], wheelvels[int(n), 2], marker='o', markersize=3, color="red")

stime = Slider(axs[2][0], 'time', 0, 0.997, valinit=0)

stime.on_changed(update)
plt.show()

