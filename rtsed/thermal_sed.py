import math

import numpy as np
import scipy.integrate as integrate

C_k = 1.3806504E-23  # J K^-1
C_c = 299792458.0  # m s^-1


def flux_nu(freq, theta, freq_0, mu, Te, model):
    """
    @param model: model used for fitting, use one of:
            cylindrical, spherical, plshell, sphshell
    @param mu: ratio between inner and outer radius of the shell i.e. mu=Rin/Rout
    @param Te: electron temperature in K
    @param freq: frequency in GHz
    @param theta: angular diameter in arcsec
    @param freq_0: turnover frequency in GHz
    @return: flux in mJy
    """
    # correction for geometry
    corrections = {
        "cylindrical": 1,
        "spherical"  : 1,
        "plshell"    : 8. / 3. - 2 * mu,
        "sphshell"   : 1 - mu
    }

    if model not in corrections.keys():
        print("Model must be one of:{}.".format(list(corrections.keys())))
        raise ValueError

    opt_thick_corr = corrections[model]

    # tau at nu assuming critical freq = freq_0
    tau_nu = tauFromTau_c(nu=freq, nu_c=freq_0)

    tau_integral = opt_thick_model(tau_nu / opt_thick_corr, mu, model)

    sld_angl = solidAngle_ap(theta)

    res_flux = RJ_function(freq * 1E9) * Te * sld_angl * tau_integral * 1E26
    # in mJy
    return res_flux * 1E3


# models for optical thickness
def opt_thick_model(tau, mu, model):
    res = False
    if model == "cylindrical":
        res = tau_cylindrical(tau)
    elif model == "spherical":
        res = tau_spherical(tau)
    elif model == "plshell":
        res = tau_trpowerlawmod(tau, mu)
    elif model == 'sphshell':
        res = tau_sphshell(tau, mu)
    return res


# optical thickness for cylindrical density distribution model I
def tau_cylindrical(tau):
    return 1 - np.exp(-1 * tau)


# optical thickness for spherical density distribution model II
def tau_spherical(tau):
    return 1 - 2 / tau ** 2 * (1 - (tau + 1) * np.exp(-1 * tau))


def tau_sphshell(tau, mu):
    res = np.array([])
    for t in tau:
        int_res_1, err = integrate.quad(optdepth_int_3, 0, mu, args=(t, mu,))
        int_res_2, err = integrate.quad(optdepth_int_4, mu, 1, args=(t,))
        res = np.append(res, int_res_1 + int_res_2)
    # extra 2* in eq 23 (4pi instead of 2pi)
    return 2. * res


# optical thickness modified truncated power-law
def tau_trpowerlawmod(tau, mu):
    res = np.array([])
    for t in tau:
        int_res_1, err = integrate.quad(optdepth_int_6, 0, 1, args=(t, mu,))
        int_res_2, err = integrate.quad(optdepth_int_2, 1, np.inf, args=(t,))
        res = np.append(res, int_res_1 + int_res_2)
    # extra 2* in eq 23 (4pi instead of 2pi)
    return 2 * res


# opt depth integral 2 (Olnon 1975 second part Eq.23)
def optdepth_int_2(x, p):
    return x * (1. - np.exp(-3. / 16. * math.pi * p / (x ** 3.)))


def optdepth_int_3(x, p, mu):
    return x * (1. - np.exp(-p * gx_sphshell(x, mu)))


def optdepth_int_4(x, p):
    return x * (1. - np.exp(-p * gx_sphere(x)))


# opt depth integral 6 (Olnon 1975 first part Eq.23)
def optdepth_int_6(x, p, mu):
    return x * (1. - np.exp(-3. / 8. * p * gx_pwlawmod(x, mu)))


# spherical shell
# g(x) function (new)
def gx_sphshell(x, mu):
    return np.sqrt(1. - x ** 2.) - np.sqrt(mu ** 2. - x ** 2.)


def gx_sphere(x):
    return np.sqrt(1. - x ** 2.)


# g(x) function, defined in Olnon 1975, eq. 21 with correction for spherical shell
def gx_pwlawmod(x, mu):
    if x == 0:
        res = 8. / 3. - 2 * mu
    elif 0 < x <= mu:
        res = x ** -3. * (math.pi / 2 - math.atan(1. / x * np.sqrt(1. - x ** 2.)) -
                          x * np.sqrt(1. - x ** 2.)) + 2. * np.sqrt(1. - x ** 2.) - \
              2. * np.sqrt(mu ** 2. - x ** 2.)
    elif mu < x < 1.:
        res = x ** -3. * (math.pi / 2. - math.atan(1. / x * np.sqrt(1. - x ** 2.)) -
                          x * np.sqrt(1. - x ** 2.)) + 2. * np.sqrt(1. - x ** 2.)
    else:
        res = x ** -3 * math.pi / 2.
    return res


def tauFromTau_c(nu, nu_c, tau_c=1.):
    '''calculates optical thickness at freq nu
        from known opt. thick.
        Pottasch 1984
        from:
        nu - frequency
        nu_c - frequency at known opt. thick.
        tau_c - known opt. thickness'''

    tau = tau_c * np.power(nu / nu_c, -2.1)

    return tau


def solidAngle_ap(theta):
    '''solid angle from angular diameter
        from:
        theta - angular diameter [arcsec]'''
    return np.pi / 4 * theta ** 2. * 3600. ** -2. * (np.pi / 180.) ** 2.


# Rayleigh-Jeans functions at freq
def RJ_function(freq):
    return 2. * freq ** 2. * C_k * C_c ** -2.
