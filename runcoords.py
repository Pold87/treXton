import coordtransform
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("/home/pold/Documents/aligner/imagelocations.csv")

ECEF_ORIGIN = (392433243.22061324, 30035730.384397615, 500219810.70945865) # from point 4185, 2728

gpsx = df.gpsx.values
gpsy = df.gpsy.values
gpsz = df.gpsz.values

xs = []
ys = []

for x, y, z in zip(gpsx, gpsy, gpsz):

    lat, lon, alt = coordtransform.ecef2latlon(x, y, z)

    datumLatLon = coordtransform.ecef2latlon(*ECEF_ORIGIN)
    x, y = coordtransform.latlon2m(datumLatLon, (lat, lon))
    xs.append(x)
    ys.append(y)

plt.plot(xs, ys)
plt.show()

preds = []

for i, (x, y) in enumerate(zip(xs, ys)):
    preds.append([x, y])

np.save("optitrack_coords", preds)
