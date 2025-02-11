"""
Copyright (c) 2023 Vincent Humphrey (gnssvod)
Copyright (c) 2019 Mustafa Serkan Işık and Volkan Özbey (gnsspy)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from gnssvod.io.readFile import (read_navFile, read_obsFile, read_sp3File,
                                read_clockFile, read_ionFile)
from gnssvod.io.manipulate import (rinex_merge)
from gnssvod.io.preprocess import (preprocess,get_filelist,gather_stations)
from gnssvod.position.atmosphere import (tropospheric_delay)
from gnssvod.position.interpolation import (sp3_interp, ionosphere_interp)
from gnssvod.position.position import (spp, multipath)
from gnssvod.geodesy import (coordinate, projection)
from gnssvod.hemistats.hemistats import (hemibuild)
from gnssvod.funcs.funcs import (gpsweekday, gpswdtodate, jday, julianday2date,
                                doy, doy2date, datetime2doy)
from gnssvod.download import (get_rinex, get_rinex3, get_navigation,
                             get_sp3, get_clock, get_ionosphere)
from gnssvod.analysis.vod_calc import (calc_vod)
from gnssvod import plot

__version__ = '2024.03.10'
__author__ = 'Vincent Humphrey (gnssvod), Mustafa Serkan Isik & Volkan Ozbey (gnsspy)'
