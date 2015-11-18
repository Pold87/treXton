from __future__ import division

import pyproj
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# Cyberzoo reference datum: 
# datumLatLon = (51.9906254106, 4.37679299373)

rotation_angle = 54

# ECEF_ORIGIN = (392433199, 30035725, 500219845)
ECEF_ORIGIN = (392433243.22061324, 30035730.384397615, 500219810.70945865) # from point 4185, 2728
ECEF_ORIGIN = (392433290.4869664, 30035742.316323183, 500219772.9140699)
ECEF_ORIGIN_CALIBRATE = (392433249, 30036183, 500219779)

# Convert ECEF to latitude/longitude
def ecef2latlon(x_ecef,y_ecef,z_ecef):

    # Compute NED location (use first coordinate as datum)
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    lon, lat, alt = pyproj.transform(ecef, lla, 
                                     x_ecef, y_ecef, z_ecef, 
                                     radians=False)
    return lat, lon, alt

def latlon2ecef(lat, lon, alt):

    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    
    x_ecef, y_ecef, z_ecef = pyproj.transform(lla, ecef, 
                                              lon, lat, alt,
                                              radians=False)
    return x_ecef, y_ecef, z_ecef

def latlon2m(datumLatLon,latLon):

    """
    Convert latitude, longitude coordinates to meter offset
    """

    datumLat = float(datumLatLon[0])
    datumLon = float(datumLatLon[1])
    lat = float(latLon[0])
    lon = float(latLon[1])

    Xn = (lat - datumLat) * 6076.11549 * 60.0 * 0.3048
    Ye = (lon - datumLon) * 6076.11549 * 60.0 * \
         math.cos(math.radians(datumLat)) * 0.3048

    return Xn, Ye

def m2latlon(datumLatLon,Xn,Ye):

    """
    Convert meter offset to latitude, longitude coordinates
    """
    
    # Reference datum
    datumLat = float(datumLatLon[0])
    datumLon = float(datumLatLon[1])
    
    Xn = float(Xn) # North
    Ye = float(Ye) # East

    earth_radius = 6076.11549 
    lat = Xn / (earth_radius * 60.0 * 0.3048) + datumLat
    lon = Ye / (earth_radius * 60.0 * 0.3048 * \
                math.cos(math.radians(datumLat))) + datumLon

    return lat, lon


def pixel_diff_to_meter_diff(x_diff, y_diff, pixel_image_shape=(2891, 4347)):

    "Convert pixel difference to meter difference"
    real_image_shape = (0.297 * 9, 0.420 * 9) # In meters

    y_factor = real_image_shape[0] / pixel_image_shape[0] 
    x_factor = real_image_shape[1] / pixel_image_shape[1] 

    map_path = "newMaze17_full.jpg"
    location_img = cv2.imread(map_path, 0) # trainImage, the zero indicates that we load it in black and white
    
    pixel_image_shape = location_img.shape # In pixels

    real_image_shape = (0.297 * 9, 0.420 * 9) # In meters

    x_factor = real_image_shape[0] / pixel_image_shape[0]

    y_factor = real_image_shape[1] / pixel_image_shape[1] 

    x_m = x_factor * x_diff
    y_m = y_factor * y_diff

    return x_m, y_m

def meter_diff_to_pixel_diff(x_diff, y_diff, pixel_image_shape=(2891, 4347)):

    "Convert meter difference to pixel difference"

    real_image_shape = (0.297 * 9, 0.420 * 9) # In meters

    y_factor = pixel_image_shape[0] / real_image_shape[0]
    x_factor = pixel_image_shape[1] / real_image_shape[1]

    x_m = x_factor * x_diff
    y_m = y_factor * y_diff

    return x_m, y_m


def pixel_pos_to_ecef_pos(x, y, pixel_image_shape = (2891, 4347)):

    """
    Takes a x, y coordinate from CV and returns the ecef coordinates
    """
    
    datumLatLon = ecef2latlon(*ECEF_ORIGIN)

    # Datum is the (0, 0) point of the real image on the floor
    lat_p_o, lon_p_o, alt = datumLatLon

    X, Y = pixel_diff_to_meter_diff(x, y, pixel_image_shape)

    print "m_diff", x, y

    # Rotate Xn, and Ye
    theta = np.radians(rotation_angle)
    Xn = np.cos(theta) * X - np.sin(theta) * Y
    Ye = np.sin(theta) * X + np.cos(theta) * Y

    print "m_diff_rot", Xn, Ye

    lat, lon = m2latlon(datumLatLon, Xn, Ye)

    print "lla", lat, lon

    ecef = latlon2ecef(lat, lon, alt)

    print "ecef", ecef

    return ecef


def ecef_pos_to_pixel_pos(ecef_x, ecef_y, ecef_z, pixel_image_shape = (2891, 4347)):

    lat, lon, alt = ecef2latlon(ecef_x, ecef_y, ecef_z) 

    datumLatLon = ecef2latlon(*ECEF_ORIGIN)


    Xn, Ye = latlon2m(datumLatLon, (lat, lon))

    # Rotate Xn, and Ye
    theta = np.radians(rotation_angle)

    ecef = latlon2ecef(lat, lon, alt)
    
    X = np.cos(theta) * Xn + np.sin(theta) * Ye
    Y = - np.sin(theta) * Xn + np.cos(theta) * Ye    

    x_m, y_m = meter_diff_to_pixel_diff(X, Y, pixel_image_shape=(2891, 4347))

    return x_m, y_m


def pixel_pos_to_ecef_pos_calibrate(x, y, pixel_image_shape = (2891, 4347)):

    """
    Takes a x, y coordinate from CV and returns the ecef coordinates
    """
    
    datumLatLon = ecef2latlon(*ECEF_ORIGIN_CALIBRATE)

    # Datum is the (0, 0) point of the real image on the floor
    lat_p_o, lon_p_o, alt = datumLatLon

    X, Y = pixel_diff_to_meter_diff(x, y, pixel_image_shape)

    # Rotate Xn, and Ye
    theta = np.radians(rotation_angle)
    Xn = np.cos(theta) * X - np.sin(theta) * Y
    Ye = np.sin(theta) * X + np.cos(theta) * Y

    lat, lon = m2latlon(datumLatLon, Xn, Ye)

    ecef = latlon2ecef(lat, lon, alt)

    return ecef

def plot():

    df = pd.read_csv('drone_info.csv')
    
    df['pixel_x'] = np.zeros(len(df.gps_x))
    df['pixel_y'] = np.zeros(len(df.gps_y))

    for i in range(len(df.gps_x)):
        
        x, y = ecef_pos_to_pixel_pos(df.ix[i, 'gps_x'],
                                     df.ix[i, 'gps_y'],
                                     df.ix[i, 'gps_z'],
                                     angle)

        df.ix[i, 'pixel_x'] = x
        df.ix[i, 'pixel_y'] = y

    df.to_csv('drone_info_pixels.csv', index=False)

    
def plot_real(angle=0):

    plot()

    df = pd.read_csv('drone_info_pixels.csv')
    plt.plot(df.pixel_x, df.pixel_y)
    plt.show()
