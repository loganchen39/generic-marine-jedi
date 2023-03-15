#!/usr/bin/env python3

#
# (C) Copyright 2019-2022 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# usage example:
# % rads_sla2ioda.py -s 2016020 -e 2016022  \
#       -i /glade/p/univ/umcp0009/lgchen/data/OISSH_NOAA/fromEric/link  \
#       -o /glade/p/univ/umcp0009/lgchen/data/OISSH_NOAA/fromEric/toIODA


from __future__ import print_function
import sys
import os
import argparse
import netCDF4 as nc
import numpy as np
import datetime
import dateutil.parser
from pathlib import Path

IODA_CONV_PATH = Path(__file__).parent/"../lib/pyiodaconv"
if not IODA_CONV_PATH.is_dir():
    IODA_CONV_PATH = Path(__file__).parent/'..'/'lib-python'
sys.path.append(str(IODA_CONV_PATH.resolve()))

IODA_PATH = Path(__file__).parent/"../lib/python3.9/pyioda"
sys.path.append(str(IODA_PATH.resolve()))

import ioda_conv_engines as iconv
from orddicts import DefaultOrderedDict
from collections import OrderedDict


os.environ["TZ"] = "UTC"

obsvars = ['seaSurfaceHeightAnomaly', 'sla']


locationKeyList = [
    ("latitude", "float", "degrees_north", "lat"),
    ("longitude", "float", "degrees_east", "lon"),
    ("dateTime", "long", "seconds since 1970-01-01T00:00:00Z", "time_mjd")
]
meta_keys = [m_item[0] for m_item in locationKeyList]

iso8601_string = locationKeyList[meta_keys.index('dateTime')][2]
epoch = datetime.datetime.fromisoformat(iso8601_string[14:-1])
refTimeRads = datetime.datetime.fromisoformat("1858-11-17T00:00:00")

metaDataName = iconv.MetaDataName()
obsValName = iconv.OvalName()
obsErrName = iconv.OerrName()
qcName = iconv.OqcName()


float_missing_value = nc.default_fillvals['f4']
int_missing_value = nc.default_fillvals['i4']
double_missing_value = nc.default_fillvals['f8']
long_missing_value = nc.default_fillvals['i8']
string_missing_value = '_'

missing_vals = {'string': string_missing_value,
                'integer': int_missing_value,
                'long': long_missing_value,
                'float': float_missing_value,
                'double': double_missing_value}
dtypes = {'string': object,
          'integer': np.int32,
          'long': np.int64,
          'float': np.float32,
          'double': np.float64}


class radsSsha2ioda(object):

    def __init__(self, file_input, date):
        self.file_input = file_input
        self.date = date

        self.globalAttrs = {
            'converter': os.path.basename(__file__),
            'ioda_version': 2,
            'sourceFiles': self.file_input,
            'datetimeReference': self.date.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'description': "Sea Surface Height Anomaly (SSHA) observations from NESDIS"
        }

        self.varAttrs = DefaultOrderedDict(lambda: DefaultOrderedDict(dict))
        self.data = OrderedDict() 

        # Set units of the MetaData variables and all _FillValues.
        for (locKeyIoda, dtypestr, unit, _) in locationKeyList:
            self.varAttrs[(locKeyIoda, metaDataName)]['_FillValue'] = missing_vals[dtypestr]
            self.varAttrs[(locKeyIoda, metaDataName)]['units'] = unit

        self._read()


    def _read(self):
        ncd = nc.Dataset(self.file_input)
        nlocs = ncd.dimensions['time'].size
        self.dimDict = {'Location': nlocs}
        self.varDims = {obsvars[0]: ['Location']}

        # get the metaData variables' data and deal with the special case of time_mjd conversion
        for (locKeyIoda, dtypestr, _, locKeyRads) in locationKeyList:
            if locKeyIoda == "dateTime":
                time_offset = []
                for i in range(nlocs):
                    time_offset.append(np.int64(round((refTimeRads 
                        + datetime.timedelta(days=np.float64(ncd.variables[locKeyRads][i])) 
                        - epoch).total_seconds())))
                self.data[(locKeyIoda, metaDataName)] = np.array(time_offset, dtype=np.int64)
            else:
                self.data[(locKeyIoda, metaDataName)] = np.array(ncd.variables[locKeyRads][:]
                    , dtype=dtypes[dtypestr])

        # obsvars, varAttrs first, then data
        self.varAttrs[(obsvars[0], obsValName)]['units'] = ncd.variables[obsvars[1]].units
        self.varAttrs[(obsvars[0], obsErrName)]['units'] = ncd.variables[obsvars[1]].units
        self.varAttrs[(obsvars[0], obsValName)]['_FillValue'] = ncd.variables[obsvars[1]]._FillValue
        self.varAttrs[(obsvars[0], obsErrName)]['_FillValue'] = ncd.variables[obsvars[1]]._FillValue
        self.varAttrs[(obsvars[0], qcName)]['_FillValue'] = int_missing_value

        self.data[(obsvars[0], obsValName)] = np.array(ncd.variables[obsvars[1]][:], dtype=np.float32)
        self.data[(obsvars[0], obsErrName)] = np.zeros(nlocs, dtype=np.float32)
        self.data[(obsvars[0], qcName)] = np.zeros(nlocs, dtype=np.int32)

        ncd.close()


def main():
    sats = ['c2', 'j2', 'sa']

    desc = 'Read sea level anomaly (SLA) observations file(s) and convert them to IODA format (SSHA) for use in JEDI system.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '-s', '--start', help="start date of data for conversion in the form of YYYYDOY or YYYYMMDD", 
        type=str, required=True, default=None)
    parser.add_argument(
        '-e', '--end', help="end date of data for conversion in the form of YYYYDOY or YYYYMMDD",
        type=str, required=True, default=None)
    parser.add_argument(
        '-i', '--input_directory', help="input directory",
        type=str, required=True)
    parser.add_argument(
        '-o', '--output_directory', help="output directory",
        type=str, required=True)

    args = parser.parse_args()

    if len(args.start) == 8:  # YYYYMMDD
        st_year, st_mon, st_day = int(args.start[0:4]), int(args.start[4:6]), int(args.start[6:8])
        jday_st = datetime.date(st_year, st_mon, st_day)
    elif len(args.start)  == 7:  # YYYYDOY
        st_year, st_doy = int(args.start[0:4]), int(args.start[4:7])
        jday_st = datetime.date(st_year, 1, 1) + datetime.timedelta(days=st_doy-1)
    else:
        sys.exit('Error: invalid start date')

    if len(args.end) == 8:  # YYYYMMDD
        end_year, end_mon, end_day = int(args.end[0:4]), int(args.end[4:6]), int(args.end[6:8])
        jday_end = datetime.date(end_year, end_mon, end_day)
    elif len(args.end)  == 7:  # YYYYDOY
        end_year, end_doy = int(args.end[0:4]), int(args.end[4:7])
        jday_end = datetime.date(end_year, 1, 1) + datetime.timedelta(days=end_doy-1)
    else:
        sys.exit('Error: invalid end date')

    for sat in sats:
        jday = jday_st
        while jday <= jday_end:
            fdate = datetime.datetime.combine(jday, datetime.time(12))
            fn = sat + '_' + jday.strftime('%Y%j') + '.nc'
            if os.path.isfile(args.input_directory + '/' + fn):
                # Read in the sla data and convert them to IODA format
                obs = radsSsha2ioda(args.input_directory + '/' + fn, fdate)

                # Write out the IODA output file.
                writer = iconv.IodaWriter(args.output_directory + '/' + fn, locationKeyList, obs.dimDict)
                writer.BuildIoda(obs.data, obs.varDims, obs.varAttrs, obs.globalAttrs)

            jday += datetime.timedelta(days=1)


if __name__ == '__main__':
    main()
