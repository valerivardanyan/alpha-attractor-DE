import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import odeint
from scipy import interpolate
from scipy.interpolate import interp1d

#Definition of the potential.
def pot(phi, params):

    model, M, gamma, alpha, Omega_M, Omega_R = params
    term_1 = M**2*np.exp(-gamma)

    exponent_tmp = gamma*np.tanh(phi/np.sqrt(6.*alpha))
    term_2 = np.exp(exponent_tmp)

    if model == "EXP_model_1":
        V_0 = 0.
        prefac = 1.
    elif model == "EXP_model_2":
        V_0 = -M**2*np.exp(-2.*gamma)
        prefac = 1.
    elif model == "LCDM":
        V_0 = M**2*np.exp(-2.*gamma)
        prefac = 0.

    return  prefac*term_1*term_2 + V_0


#The derivative of the potential
def pot_der(phi, params):

    model, M, gamma, alpha, Omega_M, Omega_R = params
    term_1 = M**2*gamma*np.exp(-gamma)

    exponent_tmp = gamma*np.tanh(phi/np.sqrt(6.*alpha))
    term_2 = np.exp(exponent_tmp)
    term_3 = 1./np.cosh(phi/np.sqrt(6.*alpha))**2/np.sqrt(6.*alpha)

    return  term_1*term_2*term_3

#Friedmann equation
def Hubble_sqr_E(phi, phi_prime, params, N):

    model, M, gamma, alpha, Omega_M, Omega_R = params
    tmp_num = pot(phi, params)/3. + Omega_M*np.exp(-3.*N) + Omega_R*np.exp(-4.*N)
    tmp_denom = 1. - phi_prime**2/6.

    return tmp_num/tmp_denom

#Expression for eps
def eps(phi, phi_prime, params, N):
    model, M, gamma, alpha, Omega_M, Omega_R = params

    tmp_hbl_E_sqr = Hubble_sqr_E(phi, phi_prime, params, N)

    tmp_1 = phi_prime**2/2.
    tmp_2 = (3.*Omega_M*np.exp(-3.*N) + 4.*Omega_R*np.exp(-4.*N))/tmp_hbl_E_sqr/2.

    return tmp_1 + tmp_2

#The r.h.s. of the e.o.m's.
def diff_evolve(y, N, params):
    phi, phi_prime = y

    tmp_eps = eps(phi, phi_prime, params, N)
    tmp_hbl_E_sqr = Hubble_sqr_E(phi, phi_prime, params, N)

    dydt = [phi_prime, - phi_prime*(3. - tmp_eps) - pot_der(phi, params)/tmp_hbl_E_sqr]

    return dydt

def solve(N_lst, initial_condition, params_in):
    model, gamma_scan, h_scan, M, gamma, alpha, Omega_M, Omega_R = params_in

    def E_today_gamma_scan(gamma):

        params = [model, M, gamma, alpha, Omega_M, Omega_R]

        sol_phi = odeint(diff_evolve, initial_condition, N_lst, args=(params,))
        sol_Hubble_E_sqr = Hubble_sqr_E(sol_phi[:, 0], sol_phi[:, 1], params, N_lst)

        E_interp = interpolate.interp1d(N_lst, np.sqrt(sol_Hubble_E_sqr))

        return np.abs(E_interp(0.) - 1.)

    def E_today_h_scan(hfid_over_h):

        params = [model, M*hfid_over_h, gamma, alpha, Omega_M, Omega_R]

        sol_phi = odeint(diff_evolve, initial_condition, N_lst, args=(params,))
        sol_Hubble_E_sqr = Hubble_sqr_E(sol_phi[:, 0], sol_phi[:, 1], params, N_lst)

        E_interp = interpolate.interp1d(N_lst, np.sqrt(sol_Hubble_E_sqr))

        return np.abs(E_interp(0.) - 1.)

    if gamma_scan == True and h_scan == False:
        gamma_finding = minimize_scalar(E_today_gamma_scan, bounds=(100., 150.), method='bounded')
        gamma = gamma_finding.x
        print("gamma = ", gamma, E_today_gamma_scan(gamma) + 1., "model = ", model)

        params = [model, M, gamma, alpha, Omega_M, Omega_R]

    if gamma_scan == False and h_scan == True:
        h_finding = minimize_scalar(E_today_h_scan, bounds=(0.1, 10.), method='bounded')
        hfid_over_h = h_finding.x
        print("h_fid/h = ", hfid_over_h, E_today_h_scan(hfid_over_h) + 1., "model = ", model)

        params = [model, M*hfid_over_h, gamma, alpha, Omega_M, Omega_R]



    #We solve
    sol_phi = odeint(diff_evolve, initial_condition, N_lst, args=(params,))
    sol_eps = eps(sol_phi[:, 0], sol_phi[:, 1], params, N_lst)

    sol_Hubble_E_sqr = Hubble_sqr_E(sol_phi[:, 0], sol_phi[:, 1], params, N_lst)

    return [params, sol_phi, sol_eps, sol_Hubble_E_sqr]


def f_solve(N_lst_f_evolve, N_lst, sol_Hubble_E_sqr, Omega_M):
    def f_evolve(y, N):

        f = y[0]

        E = E_interp(N)
        E_der = E_der_interp(N)

        a = np.exp(N)
        OM = Omega_M/a**3/E**2

        dfdN = [-f**2 - (2. + E_der/E)*f + 1.5*OM]
        return dfdN

    E_interp = interpolate.interp1d(N_lst, np.sqrt(sol_Hubble_E_sqr))
    E_der = np.gradient(np.sqrt(sol_Hubble_E_sqr), N_lst[1] - N_lst[0])

    E_der_interp = interpolate.interp1d(N_lst, E_der)

    a_ini = np.exp(N_lst_f_evolve[0])
    OM = Omega_M/a_ini**3/E_interp(N_lst_f_evolve[0])**2

    power = 5./9.
    initial_condition = [OM**power]

    sol_f = odeint(f_evolve, initial_condition, N_lst_f_evolve)

    return sol_f
