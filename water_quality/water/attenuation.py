# -*- coding: utf-8 -*-
#  Copyright 2023 
#  Center for Global Discovery and Conservation Science, Arizona State University
#
#  Licensed under the:
#  CarbonMapper Modified CC Attribution-ShareAlike 4.0 Int. (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://github.com/CMLandOcean/WaterQuality/blob/main/LICENSE.md
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#
# Translated to Python by:
#  Marcel König, mkoenig3 AT asu.edu 
#
# WaterQuality
#  Code is provided to Planet, PBC as part of the CarbonMapper Land and Ocean Program.
#  It builds on the extensive work of many researchers. For example, models were developed  
#  by Albert & Mobley [1] and Gege [2]; the methodology was mainly developed 
#  by Gege [3,4,5] and Albert & Gege [6].
#
#  Please give proper attribution when using this code for publication:
#
#  König, M., Hondula. K.L., Jamalinia, E., Dai, J., Vaughn, N.R., Asner, G.P. (2023): WaterQuality python package (Version x) [Software]. Available from https://github.com/CMLandOcean/WaterQuality
#
# [1] Albert & Mobley (2003): An analytical model for subsurface irradiance and remote sensing reflectance in deep and shallow case-2 waters. [10.1364/OE.11.002873]
# [2] Gege (2012): Analytic model for the direct and diffuse components of downwelling spectral irradiance in water. [10.1364/AO.51.001407]
# [3] Gege (2004): The water color simulator WASI: an integrating software tool for analysis and simulation of optical in situ spectra. [10.1016/j.cageo.2004.03.005]
# [4] Gege (2014): WASI-2D: A software tool for regionally optimized analysis of imaging spectrometer data from deep and shallow waters. [10.1016/j.cageo.2013.07.022]
# [5] Gege (2021): The Water Colour Simulator WASI. User manual for WASI version 6. 
# [6] Gege & Albert (2006): A Tool for Inverse Modeling of Spectral Measurements in Deep and Shallow Waters. [10.1007/1-4020-3968-9_4]


import numpy as np
from .. surface import air_water

def omega_b(a, b_b):
    """
    # Math: \omega_b = \frac{b_b}{a + b_b}
    """
    return b_b / (a + b_b)

def domega_b_div_dp(a, b_b, da_div_dp, db_b_div_dp):
    """
    # Math: \frac{\partial}{\partial p}\left[\omega_b\right] = \frac{\partial}{\partial p}\left[ b_b \times (a+b_b)^{-1} \right]
    # Math: =\frac{\partial b_b}{\partial p} (a + b_b)^{-1} - b_b(a+b_b)^{-2}(\frac{\partial a}{\partial p} + \frac{\partial b_b}{\partial p})
    # Math: = \frac{a\partial_p b - b \partial_pa}{(a + b_b)^2}
    """
    return (a * db_b_div_dp - b_b * da_div_dp) / (a + b_b)**2

# Attentuation partials have the same form for all fit params
# References to theta all invoke cos(theta). Expect np.cos(theta) as param instead.
def K_d(a,
        b_b, 
        cos_t_sun_p=np.pi/6,
        kappa_0 = 1.0546,
        ):
    """
    Diffuse attenuation for downwelling irradiance as implemented in WASI [1].
    
    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    [2] Albert, A., & Mobley, C. (2003): An analytical model for subsurface irradiance and remote sensing reflectance in deep and shallow case-2 waters [doi.org/10.1364/OE.11.002873]
    
    :param a: spectral absorption coefficient of a water body
    :param b_b: spectral backscattering coefficient of a water body
    :param theta_sun: sun zenith angle in air in units of radians
    :param n1: refrective index of origin medium, default: 1 for air
    :param n2: refractive index of destination medium, default: 1.33 for water
    :param kappa_0: coefficient depending on scattering phase function, default: 1.0546 [2]
    :return: diffuse attenuation for downwelling irradiance

    # Math: K_d(\lambda) = K_0 \frac{a(\lambda) + b_b(\lambda)}{cos\theta_{sun}'}
    """
    K_d = (kappa_0 / cos_t_sun_p) * (a + b_b)
    
    return K_d

def dK_d_div_dp(da_div_dp,
                db_b_div_dp,
                cos_t_sun_p=np.cos(np.pi/6),
                kappa_0=1.0546):
    """
    # Math: \frac{\partial}{\partial p} \left[ \frac{k_0}{cos \theta_{sun}'} (a + b_b) \right] = \frac{k_0}{cos \theta_{sun}'} (\frac{\partial a}{\partial p} + \frac{\partial b_b}{\partial p})
    """
    return (kappa_0 / cos_t_sun_p) * (da_div_dp + db_b_div_dp)
    

def k_uW(a,
         b_b,
         omega_b,
         cos_t_sun_p,
         cos_t_view_p):
    """
    # Math: k_{uW} = \frac{a + b_b}{cos \theta_v'} \times (1 + \omega_b)^{3.5421} \times (1 - \frac{0.2786}{cos \theta_{sun}'})
    """
    return (a + b_b) / cos_t_view_p * (1 + omega_b)**3.5421 * (1 - 0.2786 / cos_t_sun_p)

def dk_uW_div_dp(a, 
                 b_b,
                 omega_b,
                 da_div_dp, 
                 db_b_div_dp,
                 domega_b_div_dp,
                 cos_t_sun_p, 
                 cos_t_view_p):
    """
    # Math: \frac{\partial}{\partial p}\left[k_{uW}\right] = \frac{\partial}{\partial p}\left[\frac{a + b_b}{cos \theta_v'} \times (1 + \omega_b)^{3.5421} \times (1 - \frac{0.2786}{cos \theta_{sun}'})\right]
    # Math: = \frac{1}{cos \theta_v'}\left[ \frac{\partial}{\partial p}(a + b_b) \times (1 + \omega_b)^{3.5421} + (a + b_b) \times \frac{\partial}{\partial p}((1 + \omega_b)^{3.5421})\right]\times (1 - \frac{0.2786}{cos \theta_{sun}'})
    # Math: = \frac{1}{cos \theta_v'}\left[ (\frac{\partial a}{\partial p} + \frac{\partial b}{\partial p}) \times (1 + \omega_b)^{3.5421} + (a + b) \times (3.5421 \times (1 + \omega_b)^{2.5421} \times \frac{\partial \omega_b}{\partial p}) \right] \times (1 - \frac{0.2786}{cos \theta_{sun}'})
    """

    return 1/cos_t_sun_p * \
          (
            ((da_div_dp + db_b_div_dp) * (1 + omega_b)**(3.5421) + \
             (a + b_b) * (3.5421 * (1 + omega_b)**(2.5421) * domega_b_div_dp))
          ) * \
          (1 - 0.2786/ cos_t_view_p)


def k_uB(a,
         b_b,
         omega_b,
         cos_t_sun_p,
         cos_t_view_p):
    """
    # Math: k_{uB} = \frac{a + b_b}{cos \theta_v'} \times (1 + \omega_b)^{2.2658} \times (1 + \frac{0.0577}{cos \theta_{sun}'})
    """
    return (a + b_b) / cos_t_view_p * (1 + omega_b)**2.2658 * (1 + 0.0577 / cos_t_sun_p)

def dk_uB_div_dp(a, 
                 b_b,
                 omega_b, 
                 da_div_dp, 
                 db_b_div_dp,
                 domega_b_div_dp,
                 cos_t_sun_p, 
                 cos_t_view_p):
    """
    # Math: \frac{\partial}{\partial p}\left[k_{uB}\right] = \frac{\partial}{\partial p}\left[\frac{a + b_b}{cos \theta_v'} \times (1 + \omega_b)^{2.2658} \times (1 + \frac{0.0577}{cos \theta_{sun}'})\right]
    # Math: = \frac{1}{cos \theta_v'}\left[ \frac{\partial}{\partial p}(a + b_b) \times (1 + \omega_b)^{2.2658} + (a + b_b) \times \frac{\partial}{\partial p}((1 + \omega_b)^{2.2658})\right]\times (1 + \frac{0.0577}{cos \theta_{sun}'})
    # Math: = \frac{1}{cos \theta_v'}\left[ (\frac{\partial a}{\partial p} + \frac{\partial b}{\partial p}) \times (1 + \omega_b)^{2.2658} + (a + b) \times (2.2658 \times (1 + \omega_b)^{1.2658} \times \frac{\partial \omega_b}{\partial p}) \right] \times (1 + \frac{0.0577}{cos \theta_{sun}'})
    """
    return 1/cos_t_sun_p * \
          (
            ((da_div_dp + db_b_div_dp) * (1 + omega_b)**(2.2658) + \
             (a + b_b) * (2.2658 * (1 + omega_b)**(1.2658) * domega_b_div_dp))
          ) * \
          (1 + 0.0577 / cos_t_view_p)

