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
from .. water import absorption, backscattering, temperature_gradient, attenuation, bottom_reflectance
from .. surface import surface, air_water
from .. helper import resampling

def f_rs(omega_b,
        cos_t_sun_p = np.cos(np.radians(30)),
        cos_t_view_p = np.cos(np.radians(0))):
    """
    # Math: f_{rs} = 0.0512 \times (1 + 4.6659 \times \omega_b - 7.8387 \times \omega_b^2 + 5.4571 \times \omega_b^3) \times (1 + \frac{0.1098}{cos \theta_{sun}'}) \times (1 + \frac{0.4021}{cos \theta_{sun}'})
    
    Applying Horner's method:
    # Math: f_{rs} = 0.0512 \times (1 + \omega_b \times (4.6659 + \omega_b \times(-7.8387 + \omega_b \times (5.4571))) \times (1 + \frac{0.1098}{cos \theta_{sun}'}) \times (1 + \frac{0.4021}{cos \theta_{sun}'})
    """
    
    return 0.0512 * (1 + omega_b * (4.6659 + omega_b * (-7.8387 + omega_b * (5.4571)))) * (1 + 0.1098 / cos_t_sun_p) * (1 + 0.4021 / cos_t_view_p)
    
def df_rs_div_dp(omega_b,
                  domega_b_div_dp):
    """
    Generalized partial derivative for f_rs with respect to Fit Param (p)
    The only independent variable of the polynomial f_rs is omega_b
    and the derivative will have the same underlying for for all lower
    level variables.

    # Math: f_{rs} = 0.0512 \times (1 + \omega_b \times (4.6659 + \omega_b \times(-7.8387 + \omega_b \times (5.4571))) \times (1 + \frac{0.1098}{cos \theta_{sun}'}) \times (1 + \frac{0.4021}{cos \theta_{sun}'})
    # Math: \frac{\partial}{\partial p}\left[f_{rs}\right] = 0.0512 \times \frac{\partial \omega_b}{\partial p} \times (4.6659 + \omega_b \times (2 \times  (-7.8387) + \omega_b \times (3 \times 5.4571))
    # Math: \frac{\partial}{\partial p}\left[f_{rs}\right] = 0.0512 \times \frac{\partial \omega_b}{\partial p} \times (4.6659 + \omega_b \times (-15.6774 + \omega_b \times (16.3713))
    """
    return 0.0512 * domega_b_div_dp * (4.6659 + omega_b * (-15.6774 + omega_b * (16.3713)))


def r_rs_deep(f_rs, omega_b):
    """
    Subsurface radiance reflectance of optically deep water after Albert & Mobley (2003) [1].
    
    [1] Albert & Mobley (2003): An analytical model for subsurface irradiance and remote sensing reflectance in deep and shallow case-2 waters. [10.1364/OE.11.002873]

    :param u: ratio of backscattering coefficient to the sum of absorption and backscattering coefficients
    :param theta_sun: sun zenith angle in air [radians], is converted to in water using Snell's law, default: np.radians(30)
    :param theta_view: viewing angle in air in units [radians], is converted to in water using Snell's law, np.radians(0)
    :param n1: refrective index of origin medium, default: 1 for air
    :param n2: refrective index of destination medium, default: 1.33 for water
    :return: subsurface radiance reflectance of deep water [sr-1]
    """
    return f_rs * omega_b

def d_r_rs_deep_div_dp(f_rs,
                       df_rs_div_dp,
                       omega_b,
                       domega_b_div_dp):
    """
    # Math: \frac{\partial}{\partial p} \left[ f_rs * \omega_b \right] = \frac{\partial f_rs}{\partial p} * \omega_b + f_rs * \frac{\partial \omega_b}{\partial p}
    """
    return df_rs_div_dp * omega_b + f_rs * domega_b_div_dp


def r_rs_shallow(r_rs_deep,
                 K_d,
                 k_uW,
                 zB,
                 R_rs_b,
                 k_uB,
                 A_rs1=1.1576,
                 A_rs2=1.0389):
    """
    Subsurface radiance reflectance of optically shallow water after Albert & Mobley (2003) [1].
    
    [1] Albert & Mobley (2003): An analytical model for subsurface irradiance and remote sensing reflectance in deep and shallow case-2 waters. [10.1364/OE.11.002873]
    [2] Heege, T. (2000): Flugzeuggestützte Fernerkundung von Wasserinhaltsstoffen am Bodensee. PhD thesis. DLR-Forschungsbericht 2000-40, 134 p.
    [3] Albert, A., & Mobley, C. (2003): An analytical model for subsurface irradiance and remote sensing reflectance in deep and shallow case-2 waters [doi.org/10.1364/OE.11.002873]

    # Math: r_{rs}^{sh-} = r_{rs}^{deep-} * \left[ 1 - A_{rs,1} * e^{-(K_d + k_{uW}) * zB} \right] + A_{rs,2} * R_{rs}^b * e^{-(K_d + k_{uB}) * zB}
    """
    return r_rs_deep * \
            (1 - A_rs1 * np.exp*(-(K_d + k_uW) * zB)) + \
            A_rs2 * R_rs_b * np.exp(-(K_d + k_uB) * zB)

def drs_rs_shallow_div_dp(r_rs_deep,
                          r_rs_deep_div_dp,
                          K_d,
                          dK_d_div_dp,
                          k_uW,
                          dk_uW_div_dp,
                          r_rs_b,
                          d_r_rs_b_div_dp,
                          k_uB,
                          dk_uB_div_dp,
                          zB,
                          A_rs1=1.1576,
                          A_rs2=1.0389):
    """
    # Math: \frac{\partial}{\partial p}\left[ r_{rs}^{sh-} \right] = \left [ \frac{\partial r_{rs}^{deep-}}{\partial p} * \left[ 1 - A_{rs,1} * e^{-(K_d + k_{uW}) * zB} \right] + r_{rs}^{deep-} * \left[ 1 - A_{rs,1} * \frac{\partial e^{-(K_d + k_{uW}) * zB}}{\partial p} \right] \right] + \left[ A_{rs,2} * \frac{\partial R_{rs}^b}{\partial p} * e^{-(K_d + k_{uB}) * zB} + A_{rs,2} * R_{rs}^b * \frac{\partial e^{-(K_d + k_{uB}) * zB}}{\partial p} \right]
    # Math: = \left [ \frac{\partial r_{rs}^{deep-}}{\partial p} * \left[ 1 - A_{rs,1} * e^{-(K_d + k_{uW}) * zB} \right] + r_{rs}^{deep-} * \left[ 1 - A_{rs,1} * -(\frac{\partial K_d}{\partial p} + \frac{\partial k_{uW}}{\partial p})e^{-(K_d + k_{uW}) * zB} \right] \right] + \left[ A_{rs,2} * \frac{\partial R_{rs}^b}{\partial p} * e^{-(K_d + k_{uB}) * zB} + A_{rs,2} * R_{rs}^b * -(\frac{\partial K_d}{\partial p} + \frac{\partial k_{uB}}{\partial p}e^{-(K_d + k_{uB}) * zB} \right]
    """
    return (d_r_rs_deep_div_dp * (1 - A_rs1 * np.exp(-(K_d + k_uW)*zB)) + \
            r_rs_deep * (1 - A_rs1 * -(dK_d_div_dp + dk_uW_div_dp)*np.exp(-(K_d + k_uW)*zB))) + \
            (A_rs2 * d_r_rs_b_div_dp * np.exp(-(K_d + k_uB)*zB) + \
             A_rs2 * r_rs_b * -(dK_d_div_dp + dk_uB_div_dp) * np.exp(-(K_d + k_uB)))


def R_rs(r_rs_minus, zeta=0.52, gamma=1.6):
    return (zeta * r_rs_minus) / (1 - gamma * r_rs_minus)


def residual(R_rs_sim, 
             R_rs,
             weights = [],
             ):
    """
    Error function around model to be minimized by changing fit parameters.
    
    :return: weighted difference between measured and simulated R_rs

    # Math: L_i = R_{rs}(\lambda_i) - R_{rs_{sim}}(\lambda_i)
    """
    
    if len(weights)==0:
        weights = np.ones(len(R_rs))
    
    # sum absolute differences
    err = (R_rs-R_rs_sim) * weights
        
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
           max_nfev=400,
           jac="2-point"):
    """
    Function to inversely fit a modeled spectrum to a measurement spectrum.
    
    :param params: lmfit Parameters object containing all Parameter objects that are required to specify the model
    :param R_rs: Remote sensing reflectance spectrum [sr-1]
    :param wavelengths: wavelengths of R_rs bands [nm]
    :param weights: spectral weighing coefficients
    :param a_i_spec_res: optional, specific absorption coefficients of phytoplankton types resampled to sensor's band settings. Will be computed within function if not provided.
    :param a_w_res: optional, absorption of pure water resampled to sensor's band settings. Will be computed within function if not provided.
    :param a_Y_N_res: optional, normalized absorption coefficients of CDOM resampled to sensor's band settings. Will be computed within function if not provided.
    :param a_NAP_N_res: optional, normalized absorption coefficients of NAP resampled to sensor's band settings. Will be computed within function if not provided.
    :param b_phy_norm_res: optional, preresampling b_phy_norm saves a lot of time during inversion. Will be computed within function if not provided.
    :param b_bw_res: optional, precomputing b_bw b_bw saves a lot of time during inversion. Will be computed within function if not provided.
    :param b_X_norm_res: optional, precomputing b_bX_norm before inversion saves a lot of time. Will be computed within function if not provided.
    :param b_Mie_norm_res: optional, if n and lambda_S are not fit params, the last part of the equation can be precomputed to save time. Will be computed within function if not provided.
    :param R_i_b_res: optional, preresampling R_i_b before inversion saves a lot of time. Will be computed within function if not provided.
    :param da_W_div_dT_res: optional, temperature gradient of pure water absorption resampled  to sensor's band settings. Will be computed within function if not provided.
    :param E_0_res: optional, precomputing E_0 saves a lot of time. Will be computed within function if not provided.
    :param a_oz_res: optional, precomputing a_oz saves a lot of time. Will be computed within function if not provided.
    :param a_ox_res: optional, precomputing a_ox saves a lot of time. Will be computed within function if not provided.
    :param a_wv_res: optional, precomputing a_wv saves a lot of time. Will be computed within function if not provided.
    :param E_dd_res: optional, preresampling E_dd before inversion saves a lot of time. Will be computed within function if not provided.
    :param E_dsa_res: optional, preresampling E_dsa before inversion saves a lot of time. Will be computed within function if not provided.
    :param E_dsr_res: optional, preresampling E_dsr before inversion saves a lot of time. Will be computed within function if not provided.
    :param E_d_res: optional, preresampling E_d before inversion saves a lot of time. Will be computed within function if not provided.
    :param method: name of the fitting method to use by lmfit, default: 'least-squares'
    :param max_nfev: maximum number of function evaluations, default: 400
    :return: object containing the optimized parameters and several goodness-of-fit statistics.
    """    
    
    if params['fit_surface']==True:
        min = Minimizer(residual, 
                       params, 
                       fcn_args=(R_rs, 
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
                       max_nfev=max_nfev)
                       
    elif params['fit_surface']==False:

        params.add('P', vary=False) 
        params.add('AM', vary=False) 
        params.add('RH', vary=False) 
        params.add('H_oz', vary=False)
        params.add('WV', vary=False) 
        params.add('alpha', vary=False) 
        params.add('beta', vary=False) 
        params.add('g_dd', vary=False) 
        params.add('g_dsr', vary=False) 
        params.add('g_dsa', vary=False) 
        params.add('f_dd', vary=False) 
        params.add('f_ds', vary=False) 
        params.add('rho_L', vary=False) 

        min = Minimizer(residual, 
                       params, 
                       fcn_args=(R_rs, 
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
                       max_nfev=max_nfev) 
        
    res = min.least_squares(jac=jac)
    
    return res


def fwd(p,
        wavelengths,
        theta_sun,
        theta_view,
        depth,
        kappa_0=1.0546,
        n1=1,
        n2=1.33,
        A_rs1 = 1.1567,
        A_rs2 = 1.0389,
        a_w_res=[],
        da_W_div_dT_res=[],
        a_i_spec_res=[],
        a_Y_N_res = [],
        a_NAP_N_res = [],
        b_phy_norm_res = [],
        b_bw_res = [],
        b_X_norm_res=[],
        b_Mie_norm_res=[],
        R_i_b_res = [],
        E_0_res=[],
        a_oz_res=[],
        a_ox_res=[],
        a_wv_res=[],
        E_dd_res=[],
        E_dsa_res=[],
        E_dsr_res=[],
        E_d_res=[]):
    """
    Forward simulation of a shallow water remote sensing reflectance spectrum based on the provided parameterization.
    """
    ctsp = np.cos(air_water.snell(theta_sun, n1=n1, n2=n2))  #cos of theta_sun_prime. theta_sun_prime = snell(theta_sun, n1, n2)
    ctvp = np.cos(air_water.snell(theta_view, n1=n1, n2=n2))

    a_sim = absorption.a(C_0=p[0], C_1=p[1], C_2=p[2], C_3=p[3], C_4=p[4], C_5=p[5], 
                        C_Y=p[6], C_X=p[7], C_Mie=p[8], S=p[9], 
                        S_NAP=p[10], wavelengths=wavelengths, 
                        a_w_res=a_w_res,
                        da_W_div_dT_res=da_W_div_dT_res, 
                        a_i_spec_res=a_i_spec_res, 
                        a_Y_N_res=a_Y_N_res,
                        a_NAP_N_res=a_NAP_N_res)
    
    b_b_sim = backscattering.b_b(C_X=p[7], C_Mie=p[8], C_phy=np.sum(p[:6]), wavelengths=wavelengths, fresh=True, 
                        b_bw_res=b_bw_res, 
                        b_phy_norm_res=b_phy_norm_res, 
                        b_X_norm_res=b_X_norm_res, 
                        b_Mie_norm_res=b_Mie_norm_res)

    Rrsb = bottom_reflectance.R_rs_b(p[11], p[12], p[13], p[14], p[15], p[16], R_i_b_res=R_i_b_res)

    ob = attenuation.omega_b(a_sim, b_b_sim) #ob is omega_b. Shortened to distinguish between new var and function params.

    frs = f_rs(omega_b=ob, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)

    rrsd = r_rs_deep(f_rs=frs, omega_b=ob)

    Kd = attenuation.K_d(a=a_sim, b_b=b_b_sim, cos_t_sun_p=ctsp, kappa_0=kappa_0)

    kuW = attenuation.k_uW(a=a_sim, b_b=b_b_sim, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)

    kuB = attenuation.k_uB(a=a_sim, b_b=b_b_sim, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)

    Ars1 = A_rs1

    Ars2 = A_rs2

    R_rs_sim = air_water.below2above(r_rs_shallow(r_rs_deep=rrsd, K_d=Kd, k_uW=kuW, zB=depth, R_rs_b=Rrsb, k_uB=kuB, A_rs1=Ars1, A_rs2=Ars2))

    return R_rs_sim
'''    
    if params['fit_surface']==True:
        R_rs_sim += surface.R_rs_surf(wavelengths = wavelengths, 
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
                                      E_d_res=E_d_res)
    else:
        R_rs_sim += params['offset']
''' 

    

def wrap(outerfunc, *outer_args, **outer_kwargs):
      def inner_func(*inner_args, **inner_kwargs):
            args = list(outer_args) + list(inner_args)
            kwargs = {**outer_kwargs, **inner_kwargs}
            return outerfunc(*args, **kwargs)
      return inner_func


def build_jac(params):
    """
    Returns the Jacobian function.
    The Jacobian function must be passed to lmfit minimizer
    i.e.
      jac = build_jac(params)
      invert(fun=fun, jac=jac, params,...
    """
    fun_list = []

    if 'C_0' in params and params['C_0']['vary'] == True:
        fun_list.append(wrap(dfwd_div_dC_i, 0))
    if 'C_1' in params and params['C_0']['vary'] == True:
        fun_list.append(wrap(dfwd_div_dC_i, 1))
    if 'C_2' in params and params['C_0']['vary'] == True:
        fun_list.append(wrap(dfwd_div_dC_i, 2))
    if 'C_3' in params and params['C_0']['vary'] == True:
        fun_list.append(wrap(dfwd_div_dC_i, 3))
    if 'C_4' in params and params['C_0']['vary'] == True:
        fun_list.append(wrap(dfwd_div_dC_i, 4))
    if 'C_5' in params and params['C_0']['vary'] == True:
        fun_list.append(wrap(dfwd_div_dC_i, 5))
    
    if 'C_X' in params and params['X']['vary'] == True:
        fun_list.append(wrap(dfwd_div_dC_i, 5))
    if 'C_Y' in params and params['Y']['vary'] == True:
        fun_list.append(wrap(dfwd_div_dC_i, 5))
    if 'C_Mie' in params and params['C_Mie']['vary'] == True:
        fun_list.append(wrap(dfwd_div_dC_i, 5))
    if 'S' in params and params['S']['vary'] == True:
        fun_list.append(wrap(dfwd_div_dC_i, 5))
    if 'S_NAP' in params and params['S_NAP']['vary'] == True:
        fun_list.append(wrap(dfwd_div_dC_i, 5))
    if 'n' in params and params['n']['vary'] == True:
        fun_list.append(wrap(dfwd_div_dC_i, 5))
    if 'C_phy' in params and params['C_phy']['vary'] == True:
        fun_list.append(wrap(dfwd_div_dC_i, 5))
    
    if 'f_0' in params and params['f_0']['vary'] == True:
        fun_list.append(wrap(dfwd_div_df_i, 0))
    if 'f_1' in params and params['f_1']['vary'] == True:
        fun_list.append(wrap(dfwd_div_df_i, 1))
    if 'f_2' in params and params['f_2']['vary'] == True:
        fun_list.append(wrap(dfwd_div_df_i, 2))
    if 'f_3' in params and params['f_3']['vary'] == True:
        fun_list.append(wrap(dfwd_div_df_i, 3))
    if 'f_4' in params and params['f_4']['vary'] == True:
        fun_list.append(wrap(dfwd_div_df_i, 4))
    if 'f_5' in params and params['f_5']['vary'] == True:
        fun_list.append(wrap(dfwd_div_df_i, 5))

    if 'B_0' in params and params['B_0']['vary'] == True:
        fun_list.append(wrap(dfwd_div_dB_i, 0))
    if 'B_1' in params and params['B_1']['vary'] == True:
        fun_list.append(wrap(dfwd_div_dB_i, 1))
    if 'B_2' in params and params['B_2']['vary'] == True:
        fun_list.append(wrap(dfwd_div_dB_i, 2))
    if 'B_3' in params and params['B_3']['vary'] == True:
        fun_list.append(wrap(dfwd_div_dB_i, 3))
    if 'B_4' in params and params['B_4']['vary'] == True:
        fun_list.append(wrap(dfwd_div_dB_i, 4))
    if 'B_5' in params and params['B_5']['vary'] == True:
        fun_list.append(wrap(dfwd_div_dB_i, 5))
    

    def eval_jac(*args, **kwargs):
        return np.array([fun(*args, **kwargs) for fun in fun_list]).T
        
    return eval_jac
