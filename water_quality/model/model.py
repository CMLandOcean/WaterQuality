#!/usr/bin/env python
#  Copyright 2023 
#  Center for Global Discovery and Conservation Science, Arizona State University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#
# Translated to Python by:
#  Marcel Koenig, mkoenig3 AT asu.edu 
#
# WaterQuality
#  Code is provided to Planet, PBC as part of the CarbonMapper Land and Ocean Program.
#  It builds on the extensive work of many researchers. For example, models were developed  
#  by Albert & Mobley [1] and Gege [2]; the methodology was mainly developed 
#  by Gege [3,4,5] and Albert & Gege [6].
#
#  Please give proper attribution when using this code for publication:
#
#  Koenig, M., Hondula. K.L., Jamalinia, E., Dai, J., Asner, G.P. (2023): WaterQuality python package (Version x) [Software]. Available from https://github.com/CMLandOcean/WaterQuality
#
# [1] Albert & Mobley (2003): An analytical model for subsurface irradiance and remote sensing reflectance in deep and shallow case-2 waters. [10.1364/OE.11.002873]
# [2] Gege (2012): Analytic model for the direct and diffuse components of downwelling spectral irradiance in water. [10.1364/AO.51.001407]
# [3] Gege (2004): The water color simulator WASI: an integrating software tool for analysis and simulation of optical in situ spectra. [10.1016/j.cageo.2004.03.005]
# [4] Gege (2014): WASI-2D: A software tool for regionally optimized analysis of imaging spectrometer data from deep and shallow waters. [10.1016/j.cageo.2013.07.022]
# [5] Gege (2021): The Water Colour Simulator WASI. User manual for WASI version 6. 
# [6] Gege & Albert (2006): A Tool for Inverse Modeling of Spectral Measurements in Deep and Shallow Waters. [10.1007/1-4020-3968-9_4]


import numpy as np
from lmfit import minimize, Parameters
from .. water import absorption, backscattering, temperature_gradient, attenuation, bottom_reflectance
from .. surface import surface, air_water
from .. helper import resampling


def r_rs_dp(u, 
            theta_sun=np.radians(30), 
            theta_view=np.radians(0), 
            n1=1, 
            n2=1.33):
    """
    Subsurface radiance reflectance of optically deep water after Albert & Mobley (2003) [1].
    
    [1] Albert & Mobley (2003): An analytical model for subsurface irradiance and remote sensing reflectance in deep and shallow case-2 waters. [10.1364/OE.11.002873]

    
    :param u: ratio of backscattering coefficient to the sum of absorption and backscattering coefficients
    :param theta_sun: sun zenith angle in air in units of radians, is converted to in water using Snell's law
    :param theta_view: viewing angle in air in units of radians, is converted to in water using Snell's law
    :param n1: refrective index of origin medium, default: 1 for air
    :param n2: refrective index of destination medium, default: 1.33 for water
    :return: radiance reflectance of deep water just below the water surface in units of sr-1.
    """    
    f_rs = 0.0512 * (1 + 4.6659 * u - 7.8387 * u**2 + 5.4571 * u**3) * (1 + 0.1098/np.cos(air_water.snell(theta_sun, n1, n2))) * (1 + 0.4021/np.cos(air_water.snell(theta_view, n1, n2)))
    r_rs_dp = f_rs * u
    
    return r_rs_dp


def r_rs_sh(C_0 = 0,
            C_1 = 0,
            C_2 = 0,
            C_3 = 0,
            C_4 = 0,
            C_5 = 0,
            C_Y: float = 0,
            C_X: float = 0,
            C_Mie: float = 0,
            f_0 = 0,
            f_1 = 0,
            f_2 = 0,
            f_3 = 0,
            f_4 = 0,
            f_5 = 0,
            B_0 = 1/np.pi,
            B_1 = 1/np.pi,
            B_2 = 1/np.pi,
            B_3 = 1/np.pi,
            B_4 = 1/np.pi,
            B_5 = 1/np.pi,
            b_bphy_spec: float = 0.0010,
            b_bMie_spec: float = 0.0042,
            b_bX_spec: float = 0.0086,
            b_bX_norm_factor: float = 1,
            a_NAP_spec_lambda_0: float = 0.041,
            A: float = 0.0006,
            B: float = -0.37,
            S: float = 0.014,
            K: float = 0.0,
            S_NAP: float = 0.011,
            n: float = -1,
            lambda_0: float = 440,
            lambda_S: float = 500,
            theta_sun = np.radians(30),
            theta_view = np.radians(0),
            n1 = 1,
            n2 = 1.33,
            kappa_0 = 1.0546,
            fresh: bool = False,
            T_W=20,
            T_W_0=20,
            zB = 1,
            wavelengths: np.array = np.arange(400,800),
            a_i_spec_res=[],
            a_w_res=[],
            a_Y_N_res = [],
            a_NAP_N_res = [],
            b_phy_norm_res = [],
            b_bw_res = [],
            b_X_norm_res=[],
            b_Mie_norm_res=[],
            R_i_b_res = [],
            da_W_div_dT_res=[]
            ):
    """
    Subsurface radiance reflectance of optically shallow water after Albert & Mobley (2003) [1].
    
    [1] Albert & Mobley (2003): An analytical model for subsurface irradiance and remote sensing reflectance in deep and shallow case-2 waters. [10.1364/OE.11.002873]

    
    """
    # Backscattering and absorption coefficients of the water body depending on the concentration of optically active water constituents
    bs = backscattering.b_b(C_X=C_X,
                            C_Mie=C_Mie,
                            C_phy=np.sum([C_0,C_1,C_2,C_3,C_4,C_5]),
                            b_bphy_spec=b_bphy_spec,
                            wavelengths=wavelengths,
                            b_bMie_spec=b_bMie_spec,
                            lambda_S=lambda_S,
                            n=n,
                            b_bX_spec=b_bX_spec,
                            b_bX_norm_factor=b_bX_norm_factor,
                            fresh=fresh,
                            b_phy_norm_res = b_phy_norm_res,
                            b_bw_res = b_bw_res,
                            b_Mie_norm_res = b_Mie_norm_res,
                            b_X_norm_res = b_X_norm_res)

    ab = absorption.a(C_0,
                      C_1,
                      C_2,
                      C_3,
                      C_4,
                      C_5,
                      C_Y,
                      C_X,
                      C_Mie,
                      wavelengths=wavelengths,
                      S=S,
                      K=K,
                      a_NAP_spec_lambda_0=a_NAP_spec_lambda_0,
                      S_NAP=S_NAP,
                      lambda_0=lambda_0,
                      T_W=20,
                      T_W_0=20,
                      a_i_spec_res = a_i_spec_res,
                      a_w_res = a_w_res,
                      a_Y_N_res = a_Y_N_res,
                      a_NAP_N_res = a_NAP_N_res,
                      da_W_div_dT_res=da_W_div_dT_res,
                      )

    u = bs / (ab + bs)

    # Diffuse attenuation coefficient
    diffuse_attenuation = attenuation.K_d(a=ab, b_b=bs, theta_sun=theta_sun, n1=n1, n2=n2, kappa_0=kappa_0)

    # Attenuation coefficients for upwelling radiance of the water body (k_uW) and the bottom (k_uB)
    k_uW = ((ab + bs) / np.cos(air_water.snell(theta_view, n1, n2))) * (1 + u)**3.5421 * (1 - (0.2786 / np.cos(air_water.snell(theta_sun, n1, n2))))
    k_uB = ((ab + bs) / np.cos(air_water.snell(theta_view, n1, n2))) * (1 + u)**2.2658 * (1 - (0.0577 / np.cos(air_water.snell(theta_sun, n1, n2))))

    A_rs_1 = 1.1576 # (+/- 0.0038)
    A_rs_2 = 1.0389 # (+/- 0.0014) 

    r_rs_sh = r_rs_dp(u=u, theta_sun=theta_sun, theta_view=theta_view, n1=n1, n2=n2) * \
              (1 - A_rs_1 * np.exp(-(diffuse_attenuation + k_uW) * zB)) + \
              (A_rs_2 * bottom_reflectance.R_rs_b(f_0=f_0, f_1=f_1, f_2=f_2, f_3=f_3, f_4=f_4, f_5=f_5, B_0=B_0, B_1=B_1, B_2=B_2, B_3=B_3, B_4=B_4, B_5=B_5, wavelengths=wavelengths, R_i_b_res=R_i_b_res) * \
               np.exp(-(diffuse_attenuation + k_uB) * zB))

    return r_rs_sh
    

# Inversion
    
def func2opt(params, 
             R_rs,
             wavelengths, 
             weights = [],
             a_i_spec_res=[],
             a_w_res=[],
             a_Y_N_res = [],
             a_NAP_N_res = [],
             b_phy_norm_res = [],
             b_bw_res = [],
             b_X_norm_res=[],
             b_Mie_norm_res=[],
             R_i_b_res = [],
             da_W_div_dT_res=[],
             E_0_res=[],
             a_oz_res=[],
             a_ox_res=[],
             a_wv_res=[],
             E_dd_res=[],
             E_dsa_res=[],
             E_dsr_res=[],
             E_d_res=[]):
    """
    Error function around model to be minimized by changing fit parameters.
    
    :param params:
    :param R_rs:
    :param wavelengths:
    :return:
    """
    
    if len(weights)==0:
        weights = np.ones(len(wavelengths))
    
    if params['fit_surface'] == True:
    
        R_rs_sim = air_water.below2above(
                            r_rs_sh(C_0 = params['C_0'],
                                    C_1 = params['C_1'],
                                    C_2 = params['C_2'],
                                    C_3 = params['C_3'],
                                    C_4 = params['C_4'],
                                    C_5 = params['C_5'],
                                    C_Y = params['C_Y'],
                                    C_X = params['C_X'],
                                    C_Mie = params['C_Mie'],
                                    f_0 = params['f_0'],
                                    f_1 = params['f_1'],
                                    f_2 = params['f_2'],
                                    f_3 = params['f_3'],
                                    f_4 = params['f_4'],
                                    f_5 = params['f_5'],
                                    B_0 = params['B_0'],
                                    B_1 = params['B_1'],
                                    B_2 = params['B_2'],
                                    B_3 = params['B_3'],
                                    B_4 = params['B_4'],
                                    B_5 = params['B_5'],
                                    b_bphy_spec = params['b_bphy_spec'],
                                    b_bMie_spec = params['b_bMie_spec'],
                                    b_bX_spec = params['b_bX_spec'],
                                    b_bX_norm_factor = params['b_bX_norm_factor'],
                                    a_NAP_spec_lambda_0 = params['a_NAP_spec_lambda_0'],
                                    A = params['A'],
                                    B = params['B'],
                                    S = params['S'],
                                    K = params['K'],
                                    S_NAP = params['S_NAP'],
                                    n = params['n'],
                                    lambda_0 = params['lambda_0'],
                                    lambda_S = params['lambda_S'],
                                    theta_sun = params['theta_sun'],
                                    theta_view = params['theta_view'],
                                    n1 = params['n1'],
                                    n2 = params['n2'],
                                    kappa_0 = params["kappa_0"],
                                    fresh = params["fresh"],
                                    T_W = params["T_W"],
                                    T_W_0 = params["T_W_0"],
                                    zB = params['zB'],
                                    wavelengths = wavelengths,
                                    a_i_spec_res=a_i_spec_res,
                                    a_w_res=a_w_res,
                                    a_Y_N_res = a_Y_N_res,
                                    a_NAP_N_res = a_NAP_N_res,
                                    b_phy_norm_res=b_phy_norm_res,
                                    b_bw_res=b_bw_res,
                                    b_X_norm_res=b_X_norm_res,
                                    b_Mie_norm_res=b_Mie_norm_res,
                                    R_i_b_res=R_i_b_res,
                                    da_W_div_dT_res=da_W_div_dT_res))  + \
                            surface.R_rs_surf(wavelengths = wavelengths, 
                                              theta_sun=params['theta_sun'], 
                                              P=params['P'], 
                                              AM=params['AM'], 
                                              RH=params['RH'], 
                                              H_oz=params['H_oz'], 
                                              WV=params['WV'], 
                                              alpha=params['alpha'],
                                              beta=params['beta'],
                                              g_dd=params['g_dd'],
                                              g_dsr=params['g_dsr'],
                                              g_dsa=params['g_dsa'],
                                              f_dd=params['f_dd'], 
                                              f_ds=params['f_ds'],
                                              rho_L=params['rho_L'],
                                              E_0_res=E_0_res,
                                              a_oz_res=a_oz_res,
                                              a_ox_res=a_ox_res,
                                              a_wv_res=a_wv_res,
                                              E_dd_res=E_dd_res,
                                              E_dsa_res=E_dsa_res,
                                              E_dsr_res=E_dsr_res,
                                              E_d_res=E_d_res) + \
                            params['offset']
                    
    elif params['fit_surface']==False:

        R_rs_sim = air_water.below2above(
                            r_rs_sh(C_0 = params['C_0'],
                                    C_1 = params['C_1'],
                                    C_2 = params['C_2'],
                                    C_3 = params['C_3'],
                                    C_4 = params['C_4'],
                                    C_5 = params['C_5'],
                                    C_Y = params['C_Y'],
                                    C_X = params['C_X'],
                                    C_Mie = params['C_Mie'],
                                    f_0 = params['f_0'],
                                    f_1 = params['f_1'],
                                    f_2 = params['f_2'],
                                    f_3 = params['f_3'],
                                    f_4 = params['f_4'],
                                    f_5 = params['f_5'],
                                    B_0 = params['B_0'],
                                    B_1 = params['B_1'],
                                    B_2 = params['B_2'],
                                    B_3 = params['B_3'],
                                    B_4 = params['B_4'],
                                    B_5 = params['B_5'],
                                    b_bphy_spec = params['b_bphy_spec'],
                                    b_bMie_spec = params['b_bMie_spec'],
                                    b_bX_spec = params['b_bX_spec'],
                                    b_bX_norm_factor = params['b_bX_norm_factor'],
                                    a_NAP_spec_lambda_0 = params['a_NAP_spec_lambda_0'],
                                    A = params['A'],
                                    B = params['B'],
                                    S = params['S'],
                                    K = params['K'],
                                    S_NAP = params['S_NAP'],
                                    n = params['n'],
                                    lambda_0 = params['lambda_0'],
                                    lambda_S = params['lambda_S'],
                                    theta_sun = params['theta_sun'],
                                    theta_view = params['theta_view'],
                                    n1 = params['n1'],
                                    n2 = params['n2'],
                                    kappa_0 = params["kappa_0"],
                                    fresh = params["fresh"],
                                    T_W = params["T_W"],
                                    T_W_0 = params["T_W_0"],
                                    zB = params['zB'],
                                    wavelengths = wavelengths,
                                    a_i_spec_res=a_i_spec_res,
                                    a_w_res=a_w_res,
                                    a_Y_N_res = a_Y_N_res,
                                    a_NAP_N_res = a_NAP_N_res,
                                    b_phy_norm_res=b_phy_norm_res,
                                    b_bw_res=b_bw_res,
                                    b_X_norm_res=b_X_norm_res,
                                    b_Mie_norm_res=b_Mie_norm_res,
                                    R_i_b_res=R_i_b_res,
                                    da_W_div_dT_res=da_W_div_dT_res)) + \
                            params['offset']
    
    # absolute differences
    err = np.abs(R_rs-R_rs_sim) * weights
        
    return err
    
    
def invert(params, 
           R_rs, 
           wavelengths,
           weights=[],
           a_i_spec_res=[],
           a_w_res=[],
           a_Y_N_res = [],
           a_NAP_N_res = [],
           b_phy_norm_res = [],
           b_bw_res = [],
           b_X_norm_res=[],
           b_Mie_norm_res=[],
           R_i_b_res = [],
           da_W_div_dT_res=[],
           E_0_res=[],
           a_oz_res=[],
           a_ox_res=[],
           a_wv_res=[],
           E_dd_res=[],
           E_dsa_res=[],
           E_dsr_res=[],
           E_d_res=[],
           method="least-squares", 
           max_nfev=400
           ):
    """
    Function to inversely fit a modeled spectrum to a measurement spectrum.
    
    :param params:
    :param r_rs:
    :param wavelengths:
    :param method:
    :param max_nfev:
    :return:
    """    
    
    if params['fit_surface']==True:
        res = minimize(func2opt, 
                       params, 
                       args=(R_rs, 
                             wavelengths, 
                             weights, 
                             a_i_spec_res, 
                             a_w_res, 
                             a_Y_N_res, 
                             a_NAP_N_res, 
                             b_phy_norm_res, 
                             b_bw_res, 
                             b_X_norm_res, 
                             b_Mie_norm_res, 
                             R_i_b_res, 
                             da_W_div_dT_res,
                             E_0_res, 
                             a_oz_res, 
                             a_ox_res, 
                             a_wv_res,
                             E_dd_res,
                             E_dsa_res,
                             E_dsr_res,
                             E_d_res), 
                       method=method, 
                       max_nfev=max_nfev) 
                       
    elif params['fit_surface']==False:
        res = minimize(func2opt, 
                       params, 
                       args=(R_rs, 
                             wavelengths, 
                             weights, 
                             a_i_spec_res, 
                             a_w_res, 
                             a_Y_N_res, 
                             a_NAP_N_res, 
                             b_phy_norm_res, 
                             b_bw_res, 
                             b_X_norm_res, 
                             b_Mie_norm_res, 
                             R_i_b_res, 
                             da_W_div_dT_res), 
                       method=method, 
                       max_nfev=max_nfev) 
    return res


def forward(params,
            wavelengths,
            weights=[],
            a_i_spec_res=[],
            a_w_res=[],
            a_Y_N_res = [],
            a_NAP_N_res = [],
            b_phy_norm_res = [],
            b_bw_res = [],
            b_X_norm_res=[],
            b_Mie_norm_res=[],
            R_i_b_res = [],
            da_W_div_dT_res=[],
            E_0_res=[],
            a_oz_res=[],
            a_ox_res=[],
            a_wv_res=[],
            E_dd_res=[],
            E_dsa_res=[],
            E_dsr_res=[],
            E_d_res=[]):
    """
    Forward simulation of a shallow water spectrum based on the provided parameterization.
    
    :param params:
    :param wavelengths: np.array of wavelengths 
    :return: R_rs
    """
    
    if params['fit_surface']==True:
        
        R_rs_sim = air_water.below2above(
                            r_rs_sh(C_0 = params['C_0'],
                                    C_1 = params['C_1'],
                                    C_2 = params['C_2'],
                                    C_3 = params['C_3'],
                                    C_4 = params['C_4'],
                                    C_5 = params['C_5'],
                                    C_Y = params['C_Y'],
                                    C_X = params['C_X'],
                                    C_Mie = params['C_Mie'],
                                    f_0 = params['f_0'],
                                    f_1 = params['f_1'],
                                    f_2 = params['f_2'],
                                    f_3 = params['f_3'],
                                    f_4 = params['f_4'],
                                    f_5 = params['f_5'],
                                    B_0 = params['B_0'],
                                    B_1 = params['B_1'],
                                    B_2 = params['B_2'],
                                    B_3 = params['B_3'],
                                    B_4 = params['B_4'],
                                    B_5 = params['B_5'],
                                    b_bphy_spec = params['b_bphy_spec'],
                                    b_bMie_spec = params['b_bMie_spec'],
                                    b_bX_spec = params['b_bX_spec'],
                                    b_bX_norm_factor = params['b_bX_norm_factor'],
                                    a_NAP_spec_lambda_0 = params['a_NAP_spec_lambda_0'],
                                    A = params['A'],
                                    B = params['B'],
                                    S = params['S'],
                                    K = params['K'],
                                    S_NAP = params['S_NAP'],
                                    n = params['n'],
                                    lambda_0 = params['lambda_0'],
                                    lambda_S = params['lambda_S'],
                                    theta_sun = params['theta_sun'],
                                    theta_view = params['theta_view'],
                                    n1 = params['n1'],
                                    n2 = params['n2'],
                                    kappa_0 = params["kappa_0"],
                                    fresh = params["fresh"],
                                    T_W = params["T_W"],
                                    T_W_0 = params["T_W_0"],
                                    zB = params['zB'],
                                    wavelengths = wavelengths,
                                    a_i_spec_res=a_i_spec_res,
                                    a_w_res=a_w_res,
                                    a_Y_N_res = a_Y_N_res,
                                    a_NAP_N_res = a_NAP_N_res,
                                    b_phy_norm_res=b_phy_norm_res,
                                    b_bw_res=b_bw_res,
                                    b_X_norm_res=b_X_norm_res,
                                    b_Mie_norm_res=b_Mie_norm_res,
                                    R_i_b_res=R_i_b_res,
                                    da_W_div_dT_res=da_W_div_dT_res)) + \
                            surface.R_rs_surf(wavelengths = wavelengths, 
                                              theta_sun=params['theta_sun'], 
                                              P=params['P'], 
                                              AM=params['AM'], 
                                              RH=params['RH'], 
                                              H_oz=params['H_oz'], 
                                              WV=params['WV'], 
                                              alpha=params['alpha'],
                                              beta=params['beta'],
                                              g_dd=params['g_dd'],
                                              g_dsr=params['g_dsr'],
                                              g_dsa=params['g_dsa'],
                                              f_dd=params['f_dd'], 
                                              f_ds=params['f_ds'],
                                              rho_L=params['rho_L'],
                                              E_0_res=E_0_res,
                                              a_oz_res=a_oz_res,
                                              a_ox_res=a_ox_res,
                                              a_wv_res=a_wv_res,
                                              E_dd_res=E_dd_res,
                                              E_dsa_res=E_dsa_res,
                                              E_dsr_res=E_dsr_res,
                                              E_d_res=E_d_res) + \
                            params['offset']
                            
    elif params['fit_surface']==False:

        R_rs_sim = air_water.below2above(
                            r_rs_sh(C_0 = params['C_0'],
                                    C_1 = params['C_1'],
                                    C_2 = params['C_2'],
                                    C_3 = params['C_3'],
                                    C_4 = params['C_4'],
                                    C_5 = params['C_5'],
                                    C_Y = params['C_Y'],
                                    C_X = params['C_X'],
                                    C_Mie = params['C_Mie'],
                                    f_0 = params['f_0'],
                                    f_1 = params['f_1'],
                                    f_2 = params['f_2'],
                                    f_3 = params['f_3'],
                                    f_4 = params['f_4'],
                                    f_5 = params['f_5'],
                                    B_0 = params['B_0'],
                                    B_1 = params['B_1'],
                                    B_2 = params['B_2'],
                                    B_3 = params['B_3'],
                                    B_4 = params['B_4'],
                                    B_5 = params['B_5'],
                                    b_bphy_spec = params['b_bphy_spec'],
                                    b_bMie_spec = params['b_bMie_spec'],
                                    b_bX_spec = params['b_bX_spec'],
                                    b_bX_norm_factor = params['b_bX_norm_factor'],
                                    a_NAP_spec_lambda_0 = params['a_NAP_spec_lambda_0'],
                                    A = params['A'],
                                    B = params['B'],
                                    S = params['S'],
                                    K = params['K'],
                                    S_NAP = params['S_NAP'],
                                    n = params['n'],
                                    lambda_0 = params['lambda_0'],
                                    lambda_S = params['lambda_S'],
                                    theta_sun = params['theta_sun'],
                                    theta_view = params['theta_view'],
                                    n1 = params['n1'],
                                    n2 = params['n2'],
                                    kappa_0 = params["kappa_0"],
                                    fresh = params["fresh"],
                                    T_W = params["T_W"],
                                    T_W_0 = params["T_W_0"],
                                    zB = params['zB'],
                                    wavelengths = wavelengths,
                                    a_i_spec_res=a_i_spec_res,
                                    a_w_res=a_w_res,
                                    a_Y_N_res = a_Y_N_res,
                                    a_NAP_N_res = a_NAP_N_res,
                                    b_phy_norm_res=b_phy_norm_res,
                                    b_bw_res=b_bw_res,
                                    b_X_norm_res=b_X_norm_res,
                                    b_Mie_norm_res=b_Mie_norm_res,
                                    R_i_b_res=R_i_b_res,
                                    da_W_div_dT_res=da_W_div_dT_res)) + \
                            params['offset']
                            
    return R_rs_sim
