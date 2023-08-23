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
from .. water import absorption, backscattering, attenuation, bottom_reflectance
from .. water import albert_mobley as water_alg
from .. atmosphere import sky_radiance, downwelling_irradiance
from .. surface import surface, air_water

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
        err = (R_rs-R_rs_sim)
    else:
        err = (R_rs-R_rs_sim) * weights
        
    return err

def get_residuals(innerfunc, data, weights=[], *outer_args, **outer_kwargs):
    def outerfunc(*inner_args, **inner_kwargs):
          args = list(outer_args) + list(inner_args)
          kwargs = {**outer_kwargs, **inner_kwargs}
          if len(weights)==0:
            return (data - innerfunc(*args, **kwargs))
          else:
            return (data - innerfunc(*args, **kwargs)) * weights
    return outerfunc

# p expected as list:
# C_0...C_5, C_Y, C_X, C_Mie, S, S_NAP, f_0...f_5
def fun(p,              # Fit params only. In list for Scipy compatibility. Using different fit params will necessitate changing this function signature.
        wavelengths,

        alpha=1.317,
        AM=1,
        beta=0.2606,
        a_NAP_spec_lambda_0=0.041,
        A_rs1=1.1576,
        A_rs2=1.0389,
        b_bphy_spec=.001,
        b_bMie_spec=0.0042,
        b_bx_spec=0.0086,
        b_bx_norm_factor=1,
        depth=2,
        f_dd=1,
        f_ds=1,
        fit_surface=True,
        fresh=True,
        H_oz=0.38,
        K=0,
        kappa_0=1.0546,
        lambda_0=440,
        lambda_S=500,
        n=-1,
        n1=1,
        n2=1.33,
        offset=0,
        P=1013.25,
        RH=60,
        rho_L=0.02,
        theta_sun=np.radians(30),
        theta_view=0,
        T_W=20,
        T_W_0=20,
        WV=2.5,
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
                        S_NAP=p[10], 
                        a_NAP_spec_lambda_0=a_NAP_spec_lambda_0,
                        lambda_0=lambda_0,
                        K=K,
                        wavelengths=wavelengths,
                        T_W=T_W,
                        T_W_0=T_W_0,
                        a_w_res=a_w_res,
                        da_W_div_dT_res=da_W_div_dT_res, 
                        a_i_spec_res=a_i_spec_res, 
                        a_Y_N_res=a_Y_N_res,
                        a_NAP_N_res=a_NAP_N_res)
    
    b_b_sim = backscattering.b_b(C_X=p[7], C_Mie=p[8], C_phy=np.sum(p[:6]), wavelengths=wavelengths, 
                        fresh=fresh,
                        b_bphy_spec=b_bphy_spec,
                        b_bMie_spec=b_bMie_spec,
                        b_bX_spec=b_bx_spec,
                        b_bX_norm_factor=b_bx_norm_factor,
                        lambda_S=lambda_S,
                        n=n,
                        b_bw_res=b_bw_res, 
                        b_phy_norm_res=b_phy_norm_res, 
                        b_X_norm_res=b_X_norm_res, 
                        b_Mie_norm_res=b_Mie_norm_res)

    Rrsb = bottom_reflectance.R_rs_b(p[11], p[12], p[13], p[14], p[15], p[16], wavelengths=wavelengths, R_i_b_res=R_i_b_res)

    ob = attenuation.omega_b(a_sim, b_b_sim) #ob is omega_b. Shortened to distinguish between new var and function params.

    frs = water_alg.f_rs(omega_b=ob, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)

    rrsd = water_alg.r_rs_deep(f_rs=frs, omega_b=ob)

    Kd =  attenuation.K_d(a=a_sim, b_b=b_b_sim, cos_t_sun_p=ctsp, kappa_0=kappa_0)

    kuW = attenuation.k_uW(a=a_sim, b_b=b_b_sim, omega_b=ob, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)

    kuB = attenuation.k_uB(a=a_sim, b_b=b_b_sim, omega_b=ob, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)

    Ars1 = A_rs1

    Ars2 = A_rs2

    R_rs_water = air_water.below2above(water_alg.r_rs_shallow(r_rs_deep=rrsd, K_d=Kd, k_uW=kuW, zB=depth, R_rs_b=Rrsb, k_uB=kuB, A_rs1=Ars1, A_rs2=Ars2)) # zeta & gamma

    if fit_surface==True:        
        if len(E_dd_res) == 0:
            E_dd  = downwelling_irradiance.E_dd(wavelengths, theta_sun, P, AM, RH, H_oz, WV, alpha, beta, E_0_res, a_oz_res, a_ox_res, a_wv_res, E_dd_res)
        else:
            E_dd = E_dd_res

        if len(E_dsa_res) == 0:
            E_dsa = downwelling_irradiance.E_dsa(wavelengths, theta_sun, P, AM, RH, H_oz, WV, alpha, beta, E_0_res, a_oz_res, a_ox_res, a_wv_res, E_dsa_res)
        else:
            E_dsa = E_dsa_res

        if len(E_dsr_res) == 0:
            E_dsr = downwelling_irradiance.E_dsr(wavelengths, theta_sun, P, AM, RH, H_oz, WV, E_0_res, a_oz_res, a_ox_res, a_wv_res, E_dsr_res)
        else:
            E_dsr = E_dsr_res

        E_ds = downwelling_irradiance.E_ds(E_dsr, E_dsa)

        if len(E_d_res) == 0:
            E_d = downwelling_irradiance.E_d(E_dd, E_ds, f_dd, f_ds)
        else:
            E_d = E_d_res

        L_s = sky_radiance.L_s(p[17], E_dd, p[18], E_dsr, p[19], E_dsa)

        R_rs_surface = surface.R_rs_surf(L_s, E_d, rho_L)

        R_rs_sim = R_rs_water + R_rs_surface + offset

        return R_rs_sim    
    else:
        R_rs_sim = R_rs_water + offset
        return R_rs_sim

def dfun(p,
        wavelengths,
        alpha=1.317,
        AM=1,
        beta=0.2606,
        a_NAP_spec_lambda_0=0.041,
        A_rs1=1.1576,
        A_rs2=1.0389,
        b_bphy_spec=.001,
        b_bMie_spec=0.0042,
        b_bx_spec=0.0086,
        b_bx_norm_factor=1,
        depth=2,
        f_dd=1,
        f_ds=1,
        fit_surface=True,
        fresh=True,
        H_oz=0.38,
        K=0,
        kappa_0=1.0546,
        lambda_0=440,
        lambda_S=500,
        n=-1,
        n1=1,
        n2=1.33,
        offset=0,
        P=1013.25,
        RH=60,
        rho_L=0.02,
        theta_sun=np.radians(30),
        theta_view=0,
        T_W=20,
        T_W_0=20,
        WV=2.5,
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
    ctsp = np.cos(air_water.snell(theta_sun, n1=n1, n2=n2))  #cos of theta_sun_prime. theta_sun_prime = snell(theta_sun, n1, n2)
    ctvp = np.cos(air_water.snell(theta_view, n1=n1, n2=n2))

    a_sim = absorption.a(C_0=p[0], C_1=p[1], C_2=p[2], C_3=p[3], C_4=p[4], C_5=p[5], 
                        C_Y=p[6], C_X=p[7], C_Mie=p[8], S=p[9], 
                        S_NAP=p[10], 
                        a_NAP_spec_lambda_0=a_NAP_spec_lambda_0,
                        lambda_0=lambda_0,
                        K=K,
                        wavelengths=wavelengths,
                        T_W=T_W,
                        T_W_0=T_W_0,
                        a_w_res=a_w_res,
                        da_W_div_dT_res=da_W_div_dT_res, 
                        a_i_spec_res=a_i_spec_res, 
                        a_Y_N_res=a_Y_N_res,
                        a_NAP_N_res=a_NAP_N_res)
    
    b_b_sim = backscattering.b_b(C_X=p[7], C_Mie=p[8], C_phy=np.sum(p[:6]), wavelengths=wavelengths, 
                        fresh=fresh,
                        b_bphy_spec=b_bphy_spec,
                        b_bMie_spec=b_bMie_spec,
                        b_bX_spec=b_bx_spec,
                        b_bX_norm_factor=b_bx_norm_factor,
                        lambda_S=lambda_S,
                        n=n,
                        b_bw_res=b_bw_res, 
                        b_phy_norm_res=b_phy_norm_res, 
                        b_X_norm_res=b_X_norm_res, 
                        b_Mie_norm_res=b_Mie_norm_res)

    Rrsb = bottom_reflectance.R_rs_b(p[11], p[12], p[13], p[14], p[15], p[16], wavelengths=wavelengths, R_i_b_res=R_i_b_res)

    ob = attenuation.omega_b(a_sim, b_b_sim) #ob is omega_b. Shortened to distinguish between new var and function params.

    frs = water_alg.f_rs(omega_b=ob, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)

    rrsd = water_alg.r_rs_deep(f_rs=frs, omega_b=ob)

    Kd =  attenuation.K_d(a=a_sim, b_b=b_b_sim, cos_t_sun_p=ctsp, kappa_0=kappa_0)

    kuW = attenuation.k_uW(a=a_sim, b_b=b_b_sim, omega_b=ob, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)

    kuB = attenuation.k_uB(a=a_sim, b_b=b_b_sim, omega_b=ob, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)

    Ars1 = A_rs1

    Ars2 = A_rs2

    R_rs_water = air_water.below2above(water_alg.r_rs_shallow(r_rs_deep=rrsd, K_d=Kd, k_uW=kuW, zB=depth, R_rs_b=Rrsb, k_uB=kuB, A_rs1=Ars1, A_rs2=Ars2)) # zeta & gamma
            
    if len(E_dd_res) == 0:
        E_dd  = downwelling_irradiance.E_dd(wavelengths, theta_sun, P, AM, RH, H_oz, WV, alpha, beta, E_0_res, a_oz_res, a_ox_res, a_wv_res, E_dd_res)
    else:
        E_dd = E_dd_res
    
    if len(E_dsa_res) == 0:
        E_dsa = downwelling_irradiance.E_dsa(wavelengths, theta_sun, P, AM, RH, H_oz, WV, alpha, beta, E_0_res, a_oz_res, a_ox_res, a_wv_res, E_dsa_res)
    else:
        E_dsa = E_dsa_res
    
    if len(E_dsr_res) == 0:
        E_dsr = downwelling_irradiance.E_dsr(wavelengths, theta_sun, P, AM, RH, H_oz, WV, E_0_res, a_oz_res, a_ox_res, a_wv_res, E_dsr_res)
    else:
        E_dsr = E_dsr_res

    E_ds = downwelling_irradiance.E_ds(E_dsr, E_dsa)

    if len(E_d_res) == 0:
        E_d = downwelling_irradiance.E_d(E_dd, E_ds, f_dd, f_ds)
    else:
        E_d = E_d_res

    L_s = sky_radiance.L_s(p[17], E_dd, p[18], E_dsr, p[19], E_dsa)

    R_rs_surface = surface.R_rs_surf(L_s, E_d, rho_L) # + offset

    dbdcphy = backscattering.db_b_div_dC_phy(wavelengths, b_bphy_spec, b_phy_norm_res)

    dadC0   = absorption.da_div_dC_i(0, wavelengths, a_i_spec_res)
    domegadC0 = attenuation.domega_b_div_dp(a_sim, b_b_sim, dadC0, dbdcphy)
    dfrsdC0 = water_alg.df_rs_div_dp(ob, domegadC0)
    df_div_dC_0 = air_water.dbelow2above_div_dp(R_rs_water, 
                                                water_alg.drs_rs_shallow_div_dp(r_rs_deep=rrsd,
                                                                                dr_rs_deep_div_dp=water_alg.dr_rs_deep_div_dp(frs, dfrsdC0, ob, domegadC0),
                                                                                K_d=Kd,
                                                                                dK_d_div_dp=attenuation.dK_d_div_dp(dadC0, dbdcphy, ctsp, kappa_0),
                                                                                k_uW=kuW,
                                                                                dk_uW_div_dp=attenuation.dk_uW_div_dp(a_sim, b_b_sim, ob, dadC0, dbdcphy, ctsp, ctvp),
                                                                                r_rs_b=Rrsb,
                                                                                d_r_rs_b_div_dp=0,
                                                                                k_uB=kuB,
                                                                                dk_uB_div_dp=attenuation.dk_uB_div_dp(a_sim, b_b_sim, ob, dadC0, dbdcphy, ctsp, ctvp),
                                                                                zB=depth,
                                                                                A_rs1=Ars1,
                                                                                A_rs2=Ars2
                                                                            )
                                                )

    dadC1   = absorption.da_div_dC_i(1, wavelengths, a_i_spec_res)
    domegadC1 = attenuation.domega_b_div_dp(a_sim, b_b_sim, dadC1, dbdcphy)
    dfrsdC1 = water_alg.df_rs_div_dp(ob, domegadC1)
    df_div_dC_1 = air_water.dbelow2above_div_dp(R_rs_water, 
                                                water_alg.drs_rs_shallow_div_dp(r_rs_deep=rrsd,
                                                                                dr_rs_deep_div_dp=water_alg.dr_rs_deep_div_dp(frs, dfrsdC1, ob, domegadC1),
                                                                                K_d=Kd,
                                                                                dK_d_div_dp=attenuation.dK_d_div_dp(dadC1, dbdcphy, ctsp, kappa_0),
                                                                                k_uW=kuW,
                                                                                dk_uW_div_dp=attenuation.dk_uW_div_dp(a_sim, b_b_sim, ob, dadC1, dbdcphy, ctsp, ctvp),
                                                                                r_rs_b=Rrsb,
                                                                                d_r_rs_b_div_dp=0,
                                                                                k_uB=kuB,
                                                                                dk_uB_div_dp=attenuation.dk_uB_div_dp(a_sim, b_b_sim, ob, dadC1, dbdcphy, ctsp, ctvp),
                                                                                zB=depth,
                                                                                A_rs1=Ars1,
                                                                                A_rs2=Ars2
                                                                            )
                                                )
    
    dadC2   = absorption.da_div_dC_i(2, wavelengths, a_i_spec_res)
    domegadC2 = attenuation.domega_b_div_dp(a_sim, b_b_sim, dadC2, dbdcphy)
    dfrsdC2 = water_alg.df_rs_div_dp(ob, domegadC2)
    df_div_dC_2 = air_water.dbelow2above_div_dp(R_rs_water, 
                                                water_alg.drs_rs_shallow_div_dp(r_rs_deep=rrsd,
                                                                                dr_rs_deep_div_dp=water_alg.dr_rs_deep_div_dp(frs, dfrsdC2, ob, domegadC2),
                                                                                K_d=Kd,
                                                                                dK_d_div_dp=attenuation.dK_d_div_dp(dadC2, dbdcphy, ctsp, kappa_0),
                                                                                k_uW=kuW,
                                                                                dk_uW_div_dp=attenuation.dk_uW_div_dp(a_sim, b_b_sim, ob, dadC2, dbdcphy, ctsp, ctvp),
                                                                                r_rs_b=Rrsb,
                                                                                d_r_rs_b_div_dp=0,
                                                                                k_uB=kuB,
                                                                                dk_uB_div_dp=attenuation.dk_uB_div_dp(a_sim, b_b_sim, ob, dadC2, dbdcphy, ctsp, ctvp),
                                                                                zB=depth,
                                                                                A_rs1=Ars1,
                                                                                A_rs2=Ars2
                                                                            )
                                                )
    
    dadC3   = absorption.da_div_dC_i(3, wavelengths, a_i_spec_res)
    domegadC3 = attenuation.domega_b_div_dp(a_sim, b_b_sim, dadC3, dbdcphy)
    dfrsdC3 = water_alg.df_rs_div_dp(ob, domegadC3)
    df_div_dC_3 = air_water.dbelow2above_div_dp(R_rs_water, 
                                                water_alg.drs_rs_shallow_div_dp(r_rs_deep=rrsd,
                                                                                dr_rs_deep_div_dp=water_alg.dr_rs_deep_div_dp(frs, dfrsdC3, ob, domegadC3),
                                                                                K_d=Kd,
                                                                                dK_d_div_dp=attenuation.dK_d_div_dp(dadC3, dbdcphy, ctsp, kappa_0),
                                                                                k_uW=kuW,
                                                                                dk_uW_div_dp=attenuation.dk_uW_div_dp(a_sim, b_b_sim, ob, dadC3, dbdcphy, ctsp, ctvp),
                                                                                r_rs_b=Rrsb,
                                                                                d_r_rs_b_div_dp=0,
                                                                                k_uB=kuB,
                                                                                dk_uB_div_dp=attenuation.dk_uB_div_dp(a_sim, b_b_sim, ob, dadC3, dbdcphy, ctsp, ctvp),
                                                                                zB=depth,
                                                                                A_rs1=Ars1,
                                                                                A_rs2=Ars2
                                                                            )
                                                )

    dadC4   = absorption.da_div_dC_i(4, wavelengths, a_i_spec_res)
    domegadC4 = attenuation.domega_b_div_dp(a_sim, b_b_sim, dadC4, dbdcphy)
    dfrsdC4 = water_alg.df_rs_div_dp(ob, domegadC4)
    df_div_dC_4 = air_water.dbelow2above_div_dp(R_rs_water, 
                                                water_alg.drs_rs_shallow_div_dp(r_rs_deep=rrsd,
                                                                                dr_rs_deep_div_dp=water_alg.dr_rs_deep_div_dp(frs, dfrsdC4, ob, domegadC4),
                                                                                K_d=Kd,
                                                                                dK_d_div_dp=attenuation.dK_d_div_dp(dadC4, dbdcphy, ctsp, kappa_0),
                                                                                k_uW=kuW,
                                                                                dk_uW_div_dp=attenuation.dk_uW_div_dp(a_sim, b_b_sim, ob, dadC4, dbdcphy, ctsp, ctvp),
                                                                                r_rs_b=Rrsb,
                                                                                d_r_rs_b_div_dp=0,
                                                                                k_uB=kuB,
                                                                                dk_uB_div_dp=attenuation.dk_uB_div_dp(a_sim, b_b_sim, ob, dadC4, dbdcphy, ctsp, ctvp),
                                                                                zB=depth,
                                                                                A_rs1=Ars1,
                                                                                A_rs2=Ars2
                                                                            )
                                                )

    dadC5   = absorption.da_div_dC_i(5, wavelengths, a_i_spec_res)
    domegadC5 = attenuation.domega_b_div_dp(a_sim, b_b_sim, dadC5, dbdcphy)
    dfrsdC5 = water_alg.df_rs_div_dp(ob, domegadC5)
    df_div_dC_5 = air_water.dbelow2above_div_dp(R_rs_water, 
                                                water_alg.drs_rs_shallow_div_dp(r_rs_deep=rrsd,
                                                                                dr_rs_deep_div_dp=water_alg.dr_rs_deep_div_dp(frs, dfrsdC5, ob, domegadC5),
                                                                                K_d=Kd,
                                                                                dK_d_div_dp=attenuation.dK_d_div_dp(dadC5, dbdcphy, ctsp, kappa_0),
                                                                                k_uW=kuW,
                                                                                dk_uW_div_dp=attenuation.dk_uW_div_dp(a_sim, b_b_sim, ob, dadC5, dbdcphy, ctsp, ctvp),
                                                                                r_rs_b=Rrsb,
                                                                                d_r_rs_b_div_dp=0,
                                                                                k_uB=kuB,
                                                                                dk_uB_div_dp=attenuation.dk_uB_div_dp(a_sim, b_b_sim, ob, dadC5, dbdcphy, ctsp, ctvp),
                                                                                zB=depth,
                                                                                A_rs1=Ars1,
                                                                                A_rs2=Ars2
                                                                            )
                                                )
    
    dadCY = absorption.da_div_dC_Y(wavelengths=wavelengths, S=p[9], lambda_0=lambda_0, a_Y_N_res=a_Y_N_res)
    dbdCY = 0
    domegadCY = attenuation.domega_b_div_dp(a_sim, b_b_sim, dadCY, dbdCY)
    dfrsdCY = water_alg.df_rs_div_dp(ob, domegadCY)
    df_div_dC_Y = air_water.dbelow2above_div_dp(R_rs_water, 
                                                water_alg.drs_rs_shallow_div_dp(r_rs_deep=rrsd,
                                                                                dr_rs_deep_div_dp=water_alg.dr_rs_deep_div_dp(frs, dfrsdCY, ob, domegadCY),
                                                                                K_d=Kd,
                                                                                dK_d_div_dp=attenuation.dK_d_div_dp(dadCY, dbdCY, ctsp, kappa_0),
                                                                                k_uW=kuW, 
                                                                                dk_uW_div_dp=attenuation.dk_uW_div_dp(a_sim, b_b_sim, ob, dadCY, dbdCY, ctsp, ctvp),
                                                                                r_rs_b=Rrsb,
                                                                                d_r_rs_b_div_dp=0,
                                                                                k_uB=kuB,
                                                                                dk_uB_div_dp=attenuation.dk_uB_div_dp(a_sim, b_b_sim, ob, dadCY, dbdCY, ctsp, ctvp),
                                                                                zB=depth,
                                                                                A_rs1=Ars1,
                                                                                A_rs2=Ars2
                                                                                )
                                                )
    
    dadCX = absorption.da_div_dC_X(wavelengths=wavelengths, lambda_0=lambda_0, a_NAP_spec_lambda_0=a_NAP_spec_lambda_0, S_NAP=p[10], a_NAP_N_res=a_NAP_N_res)
    dbdCX = backscattering.db_b_div_dC_X(wavelengths=wavelengths, b_X_norm_res=b_X_norm_res)
    domegadCX = attenuation.domega_b_div_dp(a_sim, b_b_sim, dadCX, dbdCX)
    dfrsdCX = water_alg.df_rs_div_dp(ob, domegadCX)
    df_dic_dC_X = air_water.dbelow2above_div_dp(R_rs_water,
                                                water_alg.drs_rs_shallow_div_dp(r_rs_deep=rrsd,
                                                                                dr_rs_deep_div_dp=water_alg.dr_rs_deep_div_dp(frs, dfrsdCX, ob, domegadCX),
                                                                                K_d=Kd,
                                                                                dK_d_div_dp=attenuation.dK_d_div_dp(dadCX, dbdCX, ctsp, ctvp),
                                                                                k_uW=kuW,
                                                                                dk_uW_div_dp=attenuation.dk_uW_div_dp(a_sim, b_b_sim, ob, dadCX, dbdCX, ctsp, ctvp),
                                                                                r_rs_b=Rrsb,
                                                                                d_r_rs_b_div_dp=0,
                                                                                k_uB=kuB,
                                                                                dk_uB_div_dp=attenuation.dk_uB_div_dp(a_sim, b_b_sim, ob, dadCX, dbdCX, ctsp, ctvp),
                                                                                zB=depth,
                                                                                A_rs1=Ars1,
                                                                                A_rs2=Ars2
                                                                                )                                                
                    )

    dadCMie = absorption.da_div_dC_Mie(wavelengths=wavelengths, lambda_0=lambda_0, a_NAP_spec_lambda_0=a_NAP_spec_lambda_0, S_NAP=p[10], a_NAP_N_res=a_NAP_N_res)
    dbdCMie = backscattering.db_b_div_dC_Mie(wavelengths=wavelengths, n=n, b_bMie_spec=b_bMie_spec, lambda_S=lambda_S, b_bMie_norm_res=b_Mie_norm_res)
    domegadCMie = attenuation.domega_b_div_dp(a_sim, b_b_sim, dadCMie, dbdCMie)
    dfrsdCMie = water_alg.df_rs_div_dp(ob, domegadCMie)
    df_div_dC_Mie = air_water.dbelow2above_div_dp(R_rs_water,
                                                  water_alg.drs_rs_shallow_div_dp(r_rs_deep=rrsd,
                                                                                  dr_rs_deep_div_dp=water_alg.dr_rs_deep_div_dp(frs, dfrsdCMie, ob, domegadCMie),
                                                                                  K_d=Kd,
                                                                                  dK_d_div_dp=attenuation.dK_d_div_dp(dadCMie, dbdCMie, ctsp, ctvp),
                                                                                  k_uW=kuW,
                                                                                  dk_uW_div_dp=attenuation.dk_uW_div_dp(a_sim, b_b_sim, ob, dadCMie, dbdCMie, ctsp, ctvp),
                                                                                  r_rs_b=Rrsb,
                                                                                  d_r_rs_b_div_dp=0,
                                                                                  k_uB=kuB,
                                                                                  dk_uB_div_dp=attenuation.dk_uB_div_dp(a_sim, b_b_sim, ob, dadCMie, dbdCMie, ctsp, ctvp),
                                                                                  zB=depth,
                                                                                  A_rs1=Ars1,
                                                                                  A_rs2=Ars2              
                                                                                )
                    )
    
    df_div_df_0 = air_water.dbelow2above_div_dp(R_rs_water,
                                                water_alg.drs_rs_shallow_div_dp(r_rs_deep=rrsd,
                                                                                dr_rs_deep_div_dp=0,
                                                                                K_d=Kd,
                                                                                dK_d_div_dp=0,
                                                                                k_uW=kuW,
                                                                                dk_uW_div_dp=0,
                                                                                r_rs_b=Rrsb,
                                                                                d_r_rs_b_div_dp=bottom_reflectance.dR_rs_b_div_df_i(0, wavelengths=wavelengths, R_i_b_res=R_i_b_res),
                                                                                k_uB=kuB,
                                                                                dk_uB_div_dp=0,
                                                                                zB=depth,
                                                                                A_rs1=Ars1,
                                                                                A_rs2=Ars2
                                                )
                                            )
    
    df_div_df_1 = air_water.dbelow2above_div_dp(R_rs_water,
                                                water_alg.drs_rs_shallow_div_dp(r_rs_deep=rrsd,
                                                                                dr_rs_deep_div_dp=0,
                                                                                K_d=Kd,
                                                                                dK_d_div_dp=0,
                                                                                k_uW=kuW,
                                                                                dk_uW_div_dp=0,
                                                                                r_rs_b=Rrsb,
                                                                                d_r_rs_b_div_dp=bottom_reflectance.dR_rs_b_div_df_i(1, wavelengths=wavelengths, R_i_b_res=R_i_b_res),
                                                                                k_uB=kuB,
                                                                                dk_uB_div_dp=0,
                                                                                zB=depth,
                                                                                A_rs1=Ars1,
                                                                                A_rs2=Ars2
                                                )
                                            )
    
    df_div_df_2 = air_water.dbelow2above_div_dp(R_rs_water,
                                                water_alg.drs_rs_shallow_div_dp(r_rs_deep=rrsd,
                                                                                dr_rs_deep_div_dp=0,
                                                                                K_d=Kd,
                                                                                dK_d_div_dp=0,
                                                                                k_uW=kuW,
                                                                                dk_uW_div_dp=0,
                                                                                r_rs_b=Rrsb,
                                                                                d_r_rs_b_div_dp=bottom_reflectance.dR_rs_b_div_df_i(2, wavelengths=wavelengths, R_i_b_res=R_i_b_res),
                                                                                k_uB=kuB,
                                                                                dk_uB_div_dp=0,
                                                                                zB=depth,
                                                                                A_rs1=Ars1,
                                                                                A_rs2=Ars2
                                                )
                                            )
    
    df_div_df_3 = air_water.dbelow2above_div_dp(R_rs_water,
                                                water_alg.drs_rs_shallow_div_dp(r_rs_deep=rrsd,
                                                                                dr_rs_deep_div_dp=0,
                                                                                K_d=Kd,
                                                                                dK_d_div_dp=0,
                                                                                k_uW=kuW,
                                                                                dk_uW_div_dp=0,
                                                                                r_rs_b=Rrsb,
                                                                                d_r_rs_b_div_dp=bottom_reflectance.dR_rs_b_div_df_i(3, wavelengths=wavelengths, R_i_b_res=R_i_b_res),
                                                                                k_uB=kuB,
                                                                                dk_uB_div_dp=0,
                                                                                zB=depth,
                                                                                A_rs1=Ars1,
                                                                                A_rs2=Ars2
                                                )
                                            )
    
    df_div_df_4 = air_water.dbelow2above_div_dp(R_rs_water,
                                                water_alg.drs_rs_shallow_div_dp(r_rs_deep=rrsd,
                                                                                dr_rs_deep_div_dp=0,
                                                                                K_d=Kd,
                                                                                dK_d_div_dp=0,
                                                                                k_uW=kuW,
                                                                                dk_uW_div_dp=0,
                                                                                r_rs_b=Rrsb,
                                                                                d_r_rs_b_div_dp=bottom_reflectance.dR_rs_b_div_df_i(4, wavelengths=wavelengths, R_i_b_res=R_i_b_res),
                                                                                k_uB=kuB,
                                                                                dk_uB_div_dp=0,
                                                                                zB=depth,
                                                                                A_rs1=Ars1,
                                                                                A_rs2=Ars2
                                                )
                                            )
    
    df_div_df_5 = air_water.dbelow2above_div_dp(R_rs_water,
                                                water_alg.drs_rs_shallow_div_dp(r_rs_deep=rrsd,
                                                                                dr_rs_deep_div_dp=0,
                                                                                K_d=Kd,
                                                                                dK_d_div_dp=0,
                                                                                k_uW=kuW,
                                                                                dk_uW_div_dp=0,
                                                                                r_rs_b=Rrsb,
                                                                                d_r_rs_b_div_dp=bottom_reflectance.dR_rs_b_div_df_i(5, wavelengths=wavelengths, R_i_b_res=R_i_b_res),
                                                                                k_uB=kuB,
                                                                                dk_uB_div_dp=0,
                                                                                zB=depth,
                                                                                A_rs1=Ars1,
                                                                                A_rs2=Ars2
                                                )
                                            )
    
    df_div_dg_dd = sky_radiance.d_LS_div_dg_dd(E_dd)

    df_div_dg_dsa = sky_radiance.d_LS_div_dg_dsa(E_dsa)

    df_div_dg_dsr = sky_radiance.d_LS_div_dg_dsr(E_dsr)

    zeros = np.zeros(len(wavelengths))

    return np.array([
        df_div_dC_0,
        df_div_dC_1,
        df_div_dC_2,
        df_div_dC_3,
        df_div_dC_4,
        df_div_dC_5,
        df_div_dC_Y,
        df_dic_dC_X,
        df_div_dC_Mie,
        zeros,
        zeros,
        df_div_df_0,
        df_div_df_1,
        df_div_df_2,
        df_div_df_3,
        df_div_df_4,
        df_div_df_5,
        df_div_dg_dd,
        df_div_dg_dsr,
        df_div_dg_dsa
    ]).T

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
