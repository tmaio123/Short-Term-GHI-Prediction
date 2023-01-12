import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
from datetime import datetime
import xarray as xr
import sys
import netCDF4 as nc
from rasterio.transform import Affine
import rasterio as rio
import datetime as dt
from netCDF4 import date2num,num2date


api_auth = '/home/tyler/.api_request'
url = "https://forecasting.energy.arizona.edu/erebos/series/adjghi"

##### Below is the function for developing a 9x9 grid of ghi estimated values from EREBOS

## central_lon: longitude of the location in which you are forecasting
## central_lat: latitude of the location in which you are forecasting
## year: the year of the forecast date/time
## month: the month of the forecast date/time
## day: the day of the forecast date/time
## hour: the (initital) hour of the forecast date/time
## minute_start: the starting minute of the forecast date/time (needs to be 2,7,12,17,22,27,32,37,42,47,52,57 only)
## minute_end: the ending minute of the forecast date/time (needs to be 2,7,12,17,22,27,32,37,42,47,52,57 only)
## new_hour: set to false, however when true will reset the minutes and add an hour once the loop is greater than 57 mins.


##### Work flow
### Part 1 of the function:
## a 9x9 grid of latitude and longitude is created which will be centered around the central lats and lons specified

### Part 2 of the function:
## a double for loop is used to loop through all 81 lat lon grid points
## an API request is sent using the year, month, day, lat, and lon arguments specified
## ghi values are obtained at all lat lon points and appended into a pandas dataframe
## start and end dates are created based off of the year, month, day, hour, and mins specified
## if new_hour is true, the hour is increased by 1 for the end date
## the dataframe is then sliced based off of the start and end dates specified
## the rest of part 2 is creating a netcdf (nc) file of the data

### Part 3 of the function:
## all that is done in this part is taking the data from the netcdf file and creating a raster file


###### run time is approximately 2 mins 25 secs ######



def grid_maker(central_lon=None, central_lat=None, year=None, month=None, day=None, hour=None, minute_start=None, minute_end=None, index_minute_end=None,new_hour=False):

    latz = []
    lonz = []

    for i in range(0,9):
        if 0 <= i < 4:
            dalats_S = central_lat - (0.009 * i)
            dalons_E = central_lon + (0.009 * i)
            latz.append(dalats_S)
            lonz.append(dalons_E)
        if i == 4:
            latz.append(central_lat)
            lonz.append(central_lon)
        if i >= 5:
            dalats_N = central_lat + (0.009 * i)
            dalons_W = central_lon - (0.009 * i)
            latz.append(dalats_N)
            lonz.append(dalons_W)

    ghi_array1 = []
    ghi_array2 = []
    ghi_array3 = []
    ghi_array4 = []
    ghi_array5 = []
    ghi_array6 = []
    ghi_array7 = []

    print("Part 1 is done")
    for i in range(len(latz)):
        for j in range(len(lonz)):
            args = {'run_date':'{0}-{1}-{2}'.format(year,month,day),
                        'lon':'{}'.format(lonz[j]),
                        'lat':'{}'.format(latz[i]),
                        'precipitable_water':'1.25',
                        'aod700':'0.05'}
            with open(api_auth) as f:
                auth_text = f.read()
            auth_tuple = tuple(auth_text.split('\n'))[:2]
            x = requests.get(url, params=args, auth=auth_tuple)
            df = pd.DataFrame(x.json())
            df.index = pd.to_datetime(df.index)
            if new_hour == False:
                start = '{0}-{1}-{2} {3}:{4}:00'.format(year,month,day,hour,minute_start)
                end = '{0}-{1}-{2} {3}:{4}:00'.format(year,month,day,hour,minute_end)
                end1 = '{0}-{1}-{2} {3}:{4}:00'.format(year,month,day,hour,index_minute_end)
            if new_hour == True:
                hourz = int(hour)
                hour1 = hourz + 1
                start = '{0}-{1}-{2} {3}:{4}:00'.format(year,month,day,hour,minute_start)
                end = '{0}-{1}-{2} {3}:{4}:00'.format(year,month,day,hour1,minute_end)
                end1 = '{0}-{1}-{2} {3}:{4}:00'.format(year,month,day,hour1,index_minute_end)


            a = df.loc[start:end1]
            ghi = a['results'].squeeze().values
            ghi_array1.append(ghi[0])
            ghi_array2.append(ghi[1])
            ghi_array3.append(ghi[2])
            ghi_array4.append(ghi[3])
            ghi_array5.append(ghi[4])
            ghi_array6.append(ghi[5])
            ghi_array7.append(ghi[6])



    ghi1 = np.array([ghi_array1,ghi_array2,ghi_array3,ghi_array4,ghi_array5,ghi_array6,ghi_array7])


    latz = np.asarray(latz)
    lonz = np.asarray(lonz)


    filename = 'erebos_grid_{0}-{1}-{2}_{3}_{4}.nc'.format(year,month,day,hour,minute_start)

    ghi1 = ghi1.reshape(7,9,9)

    ds = nc.Dataset(filename, 'w', format='NETCDF4')

    date_range1 = pd.date_range(start=start, end=end, freq='5T')
    date_range = date_range1.to_pydatetime()

    mins = np.asarray(date_range1.minute)

    time = ds.createDimension('time', None)
    lat = ds.createDimension('lat', 9)
    lon = ds.createDimension('lon', 9)

    time = ds.createVariable('time', np.float64, ('time',))
    time.units = 'hours since 1800-01-01'
    time.long_name = 'time'

    times = date2num(date_range, time.units)


    lats = ds.createVariable('lat', 'f4', ('lat'))
    lons = ds.createVariable('lon', 'f4', ('lon'))
    ghi = ds.createVariable('ghi', 'f4', ('time', 'lat', 'lon',))
    ghi.units = 'Unknown'

    lats[:] = latz
    lons[:] = lonz
    time[:] = times

    ghi[:, :, :] = ghi1

    ds.close()

    print("Part 2 is done")

    ## read data with xarray, extract values from nc file, and assign variables

    data = xr.open_dataset('/home/tyler/cloud_advection/operational/{}'.format(filename))

    ghi = data['ghi'].values

    lats = data['lat'].values
    lons = data['lon'].values

    time = data['time'].values

    ## calculate transform variable to create raster data

    res = (lons[-1] - lons[0]) / 240.0
    transform = Affine.translation(lons[0] - res / 2, lats[0] - res / 2) * Affine.scale(res, res)


    # open in 'write' mode, unpack profile info to dst
    ## Create raster file
    for i in range(len(mins)):
        ghi_r = np.asarray(ghi[i,:,:])
        if new_hour == False:
            with rio.open(
               'erebos_grid_{0}-{1}-{2}_{3}_{4}.tif'.format(year,month,day,hour,mins[i]),
               "w",
               driver="GTiff",         # output file type
               height=ghi_r.shape[0],      # shape of array
               width=ghi_r.shape[1],
               count=1,                # number of bands
               dtype=ghi_r.dtype,          # output datatype
               crs="+proj=latlong",    # CRS
               transform=transform,    # location and resolution of upper left cell
            ) as dst:
               # check for number of bands
               if dst.count == 1:
                   # write single band
                   dst.write(ghi_r, 1)
               else:
                   # write each band individually
                   for band in range(len(ghi_r)):
                       # write data, band # (starting from 1)
                       dst.write(ghi_r[band], band + 1)

        if new_hour == True:
            if mins[i] < 32:
                if hourz < 23:

                    with rio.open(
                       'erebos_grid_{0}-{1}-{2}_{3}_{4}.tif'.format(year,month,day,hour1,mins[i]),
                       "w",
                       driver="GTiff",         # output file type
                       height=ghi_r.shape[0],      # shape of array
                       width=ghi_r.shape[1],
                       count=1,                # number of bands
                       dtype=ghi_r.dtype,          # output datatype
                       crs="+proj=latlong",    # CRS
                       transform=transform,    # location and resolution of upper left cell
                    ) as dst:
                       # check for number of bands
                       if dst.count == 1:
                           # write single band
                           dst.write(ghi_r, 1)
                       else:
                           # write each band individually
                           for band in range(len(ghi_r)):
                               # write data, band # (starting from 1)
                               dst.write(ghi_r[band], band + 1)



            if mins[i] >= 32:
                with rio.open(
                   'erebos_grid_{0}-{1}-{2}_{3}_{4}.tif'.format(year,month,day,hour,mins[i]),
                   "w",
                   driver="GTiff",         # output file type
                   height=ghi_r.shape[0],      # shape of array
                   width=ghi_r.shape[1],
                   count=1,                # number of bands
                   dtype=ghi_r.dtype,          # output datatype
                   crs="+proj=latlong",    # CRS
                   transform=transform,    # location and resolution of upper left cell
                ) as dst:
                   # check for number of bands
                   if dst.count == 1:
                       # write single band
                       dst.write(ghi_r, 1)
                   else:
                       # write each band individually
                       for band in range(len(ghi_r)):
                           # write data, band # (starting from 1)
                           dst.write(ghi_r[band], band + 1)

    print("Job is Done")

