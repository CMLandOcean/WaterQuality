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

def K_d(a,
        b_b, 
        theta_sun = np.radians(30),
        n1=1,
        n2=1.33,
        kappa_0 = 1.0546):
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
    K_d = kappa_0 * ((a + b_b) / np.cos(air_water.snell(theta_sun, n1=1, n2=1.33)))
    
    return K_d

def dK_d_div_dC_i(
        da_div_dC_i,
        b_b, 
        theta_sun = np.radians(30),
        n1=1,
        n2=1.33,
        kappa_0 = 1.0546):
    """
    # Math: \frac{\partial}{\partial C_i}K_d(\lambda) = \frac{\partial}{\partial C_i} \left[ K_0 \frac{a(\lambda) + b_b(\lambda)}{cos\theta_{sun}'} \right] = \frac{K_0}{cos\theta_{sum}'} \frac{\partial}{\partial C_i}a
    """
    dK_d_div_dC_i = kappa_0 / np.cos(air_water.snell(theta_sun, n1=1, n2=1.33)) * da_div_dC_i

    return dK_d_div_dC_i

def dK_d_div_dC_Y(
        da_div_dC_Y,
        b_b, 
        theta_sun = np.radians(30),
        n1=1,
        n2=1.33,
        kappa_0 = 1.0546):
    """
    # Math: \frac{\partial}{\partial C_Y}K_d(\lambda) = \frac{\partial}{\partial C_Y} \left[ K_0 \frac{a(\lambda) + b_b(\lambda)}{cos\theta_{sun}'} \right] = \frac{K_0}{cos\theta_{sum}'} \frac{\partial}{\partial C_Y}a
    """
    dK_d_div_dC_Y = kappa_0 / np.cos(air_water.snell(theta_sun, n1=1, n2=1.33)) * da_div_dC_Y

    return dK_d_div_dC_Y

def dK_d_div_dS(
        da_div_dS,
        b_b, 
        theta_sun = np.radians(30),
        n1=1,
        n2=1.33,
        kappa_0 = 1.0546):
    """
    # Math: \frac{\partial}{\partial S}K_d(\lambda) = \frac{\partial}{\partial S} \left[ K_0 \frac{a(\lambda) + b_b(\lambda)}{cos\theta_{sun}'} \right] = \frac{K_0}{cos\theta_{sum}'} \frac{\partial}{\partial S}a
    """
    dK_d_div_dS = kappa_0 / np.cos(air_water.snell(theta_sun, n1=1, n2=1.33)) * da_div_dS

    return dK_d_div_dS

def dK_d_div_dS_NAP(
        da_div_dS_NAP,
        b_b, 
        theta_sun = np.radians(30),
        n1=1,
        n2=1.33,
        kappa_0 = 1.0546):
    """
    # Math: \frac{\partial}{\partial S_{NAP}}K_d(\lambda) = \frac{\partial}{\partial S_{NAP}} \left[ K_0 \frac{a(\lambda) + b_b(\lambda)}{cos\theta_{sun}'} \right] = \frac{K_0}{cos\theta_{sum}'} \frac{\partial}{\partial S_{NAP}}a
    """
    dK_d_div_dS_NAP = kappa_0 / np.cos(air_water.snell(theta_sun, n1=1, n2=1.33)) * da_div_dS_NAP

    return dK_d_div_dS_NAP

def dK_d_div_dC_X(
        da_div_dC_X,
        db_b_div_dC_X, 
        theta_sun = np.radians(30),
        n1=1,
        n2=1.33,
        kappa_0 = 1.0546):
    """
    # Math: \frac{\partial}{\partial C_X}K_d(\lambda) = \frac{\partial}{\partial C_X} \left[ K_0 \frac{a(\lambda) + b_b(\lambda)}{cos\theta_{sun}'} \right] = \frac{K_0}{cos\theta_{sum}'} (\frac{\partial}{\partial C_X}a + \frac{\partial}{\partial C_X}b_b)
    """
    dK_d_div_dC_X = (kappa_0 / np.cos(air_water.snell(theta_sun, n1=1, n2=1.33))) * (da_div_dC_X + db_b_div_dC_X)

    return dK_d_div_dC_X

def dK_d_div_dC_Mie(
        da_div_dC_Mie,
        db_b_div_dC_Mie, 
        theta_sun = np.radians(30),
        n1=1,
        n2=1.33,
        kappa_0 = 1.0546):
    """
    # Math: \frac{\partial}{\partial C_{Mie}}K_d(\lambda) = \frac{\partial}{\partial C_{Mie}} \left[ K_0 \frac{a(\lambda) + b_b(\lambda)}{cos\theta_{sun}'} \right] = \frac{K_0}{cos\theta_{sum}'} (\frac{\partial}{\partial C_{Mie}}a + \frac{\partial}{\partial C_{Mie}}b_b)
    """
    dK_d_div_dC_Mie = (kappa_0 / np.cos(air_water.snell(theta_sun, n1=1, n2=1.33))) * (da_div_dC_Mie + db_b_div_dC_Mie)

    return dK_d_div_dC_Mie

def dK_d_div_dn(
        a,
        db_b_div_dn, 
        theta_sun = np.radians(30),
        n1=1,
        n2=1.33,
        kappa_0 = 1.0546):
    """
    # Math: \frac{\partial}{\partial n}K_d(\lambda) = \frac{\partial}{\partial n} \left[ K_0 \frac{a(\lambda) + b_b(\lambda)}{cos\theta_{sun}'} \right] = \frac{K_0}{cos\theta_{sum}'} \frac{\partial}{\partial n}b_b
    """
    dK_d_div_dn = kappa_0 / np.cos(air_water.snell(theta_sun, n1=1, n2=1.33)) * db_b_div_dn

    return dK_d_div_dn

def dK_d_div_C_phy(
        a,
        db_b_div_C_phy, 
        theta_sun = np.radians(30),
        n1=1,
        n2=1.33,
        kappa_0 = 1.0546):
    """
    # Math: \frac{\partial}{\partial C_{phy}}K_d(\lambda) = \frac{\partial}{\partial C_{phy}} \left[ K_0 \frac{a(\lambda) + b_b(\lambda)}{cos\theta_{sun}'} \right] = \frac{K_0}{cos\theta_{sum}'} \frac{\partial}{\partial C_{phy}}b_b
    """
    dK_d_div_dC_phy = kappa_0 / np.cos(air_water.snell(theta_sun, n1=1, n2=1.33)) * db_b_div_C_phy

    return dK_d_div_dC_phy
