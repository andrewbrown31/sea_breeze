import numpy as np
import metpy.calc as mpcalc

def metpy_grid_area(lon,lat):
    """
    From a grid of latitudes and longitudes, calculate the grid spacing in x and y, and the area of each grid cell in km^2
    """
    xx,yy=np.meshgrid(lon,lat)
    dx,dy=mpcalc.lat_lon_grid_deltas(xx, yy)
    dx=np.pad(dx,((0,0),(0,1)),mode="edge")
    dy=np.pad(dy,((0,1),(0,0)),mode="edge")
    return dx.to("km"),dy.to("km"),(dx*dy).to("km^2")