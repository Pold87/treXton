import pylab as pl
data = pl.random((25,25)) # 25x25 matrix of values
pl.pcolor(data)
pl.colorbar()
pl.show()
