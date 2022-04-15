import matplotlib.pyplot as plt
import vplot
import vplanet

orbits = vplanet.get_output()
rot = vplanet.get_output("Rotation")

fig = plt.figure(figsize=(8.5, 6))
plt.subplot(2, 2, 1)
plt.plot(orbits.b.Time / 1e6, orbits.b.Eccentricity,color=vplot.colors.red)
plt.plot(orbits.c.Time / 1e6, orbits.c.Eccentricity,color=vplot.colors.orange)
plt.ylabel("Eccentricity")
plt.xlabel("Time (Myr)")
plt.xlim(0,100)

plt.subplot(2, 2, 2)
plt.plot(orbits.b.Time / 1e6, orbits.b.LongP,color=vplot.colors.red)
plt.plot(orbits.c.Time / 1e6, orbits.c.LongP,color=vplot.colors.orange)
plt.ylabel("Inclination ($^\circ$)")
plt.xlabel("Time (Myr)")
plt.xlim(0,100)

plt.subplot(2, 2, 3)
plt.plot(rot.b.Time / 1e3, rot.b.Obliquity,color=vplot.colors.red)
plt.plot(rot.c.Time / 1e3, rot.c.Obliquity,color=vplot.colors.orange)
plt.xlabel("Time (kyr)")
plt.ylabel("Obliquity ($^{\circ}$)")
plt.ylim(1e-3,10)
plt.xlim(0,100)
plt.yscale('log')

plt.subplot(2, 2, 4)
plt.plot(rot.b.Time / 1e3, rot.b.SurfEnFluxTotal,color=vplot.colors.red)
plt.plot(rot.c.Time / 1e3, rot.c.SurfEnFluxTotal,color=vplot.colors.orange)
plt.xlabel("Time (kyr)")
plt.ylabel("Tidal Energy Flux (W/m$^2$)")
plt.xlim(0,100)
plt.ylim(0,1e4)


# Save the figure
fig.savefig("Orbits.pdf")
