# Written by: Peiqi Xia
# GitHub username: edsml-px122

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from scipy.stats import norm
from functools import partial

# Zero order model
def ZO_getQ(t, k0):
    return k0 * t

def ZO_linear(t, qt):
    t1 = t.reshape(-1, 1)
    model = LinearRegression()
    model.fit(t1, qt)
    k0 = float(model.coef_[0])

    q_model = ZO_getQ(t, k0)
    r_sq = rsquared(qt, q_model)
    return k0, r_sq, q_model

# First order model
def PFO_getQ(t, qe, k1):
    return qe * (1 - np.exp(-k1 * t))

def PFO_linear(t, qt):
    qe = max(qt) * 1.01
    ln = np.log(qe - qt)
    t1 = t.reshape(-1, 1)
    model = LinearRegression()
    model.fit(t1, ln)
    c = float(model.intercept_)
    m = float(model.coef_[0])
    
    k1 = -m
    qe = np.exp(c)
    
    q_model = PFO_getQ(t, qe, k1)
    r_sq = rsquared(qt, q_model)
    return qe, k1, r_sq, q_model

def PFO_nonlinear(t, qt):
    p1, p2, _, _ = PFO_linear(t, qt)
    p0 = np.array([p1,p2])
    popt, pcov = curve_fit(PFO_getQ, t, qt, p0=p0, maxfev=1000000)
    qe, k1 = popt

    q_model = PFO_getQ(t, qe, k1)
    r_sq = rsquared(qt, q_model)
    return qe, k1, r_sq, q_model

# Second order model
def PSO_getQ(t, qe, k2):
    return t / ((t / qe) + (1 / (k2 * qe**2)))

def PSO_linear(t, qt):
    t_qt = t[1:] / qt[1:]
    t1 = t.reshape(-1,1)
    model = LinearRegression()
    model.fit(t1[1:], t_qt)
    c = float(model.intercept_)
    m = float(model.coef_[0])

    qe = 1/m
    k2 = 1/(c*qe*qe)

    q_model = PSO_getQ(t, qe, k2)
    r_sq = rsquared(qt, q_model)
    return qe, k2, r_sq, q_model

def PSO_nonlinear(t, qt):
    p1, p2, _, _ = PSO_linear(t, qt)
    p0 = np.array([p1, p2])
    popt, pcov = curve_fit(PSO_getQ, t, qt, p0=p0, maxfev=1000000)
    qe, k2 = popt

    q_model = PSO_getQ(t, qe, k2)
    r_sq = rsquared(qt, q_model)
    return qe, k2, r_sq, q_model

def rPSO_getQ(t, qe, kprime, C0, Cs):
    dt = t[-1] / 1000
    t1 = np.arange(0, t[-1]+dt, dt)
    t_model = np.array(t * (1/dt), 'i')
    q = np.zeros(len(t1), dtype='float')
    q_model = []
    for i in range(1, len(t1)):
        q[i] = q[i-1] + (dt * (kprime * (C0 - Cs * q[i-1]) * (1 - q[i-1] / qe)**2))
    for j in t_model:
        q_model.append(q[j])
    return q_model

def rPSO_nonlinear(t, qt, C0, Cs):
    p1, p2, _, _ = PSO_nonlinear(t, qt)
    p0 = np.array([p1, p2 * p1**2 / C0])

    rPSO_getQ_fixed = partial(rPSO_getQ, C0=C0, Cs=Cs)
    popt, pcov = curve_fit(rPSO_getQ_fixed, t, qt, p0=p0, maxfev=1000000)
    qe, kprime = popt

    q_model = rPSO_getQ(t, qe, kprime, C0, Cs)
    r_sq = rsquared(qt, q_model)
    return qe, kprime, r_sq, q_model

# Multiple datasets
def ini_rate(t, qt):
    r = np.zeros(len(qt))
    for i in range(len(qt)):
        r[i] = qt[i][1] / t[i][1]
    return r

def order_analysis(r, C0):
    ini_conc = C0.reshape(-1, 1)
    ini_rates = r.reshape(-1, 1)
    model = LinearRegression()
    model.fit(np.log(ini_conc), np.log(ini_rates))
    order = float(model.coef_[0][0])
    r_pred = model.predict(np.log(ini_conc))
    return order, r_pred

# Model analysis
def rsquared(qt, q_model):
    return 1 - np.sum((qt - q_model)**2) / np.sum((qt - np.mean(qt))**2)

def error_analysis(t, qt, q_model, order, func, C0=None, Cs=None):
    sse = np.sum(np.square(q_model - qt))
    std = np.sqrt(sse/(len(qt)-2))
    if order == 0:
        k_list = np.zeros(100)
        for i in range(100):
            q_sim = q_model + norm.ppf(np.random.rand(len(qt)), 0, std)
            k_list[i], _, _ = func(t, q_sim)
        k_list = sorted(k_list)
        k_err = (k_list[94] - k_list[5]) / 2
        qe_err = None
    elif func == rPSO_nonlinear:
        qe_list = np.zeros(100)
        k_list = np.zeros(100)
        for i in range(100):
            q_sim = q_model + norm.ppf(np.random.rand(len(qt)), 0, std)
            qe_list[i], k_list[i], _, _ = func(t, q_sim, C0, Cs)
        qe_list = sorted(qe_list)
        k_list = sorted(k_list)
        qe_err = (qe_list[94] - qe_list[5]) / 2
        k_err = (k_list[94] - k_list[5]) / 2
    else:
        qe_list = np.zeros(100)
        k_list = np.zeros(100)
        for i in range(100):
            q_sim = q_model + norm.ppf(np.random.rand(len(qt)), 0, std)
            qe_list[i], k_list[i], _, _ = func(t, q_sim)
        qe_list = sorted(qe_list)
        k_list = sorted(k_list)
        qe_err = (qe_list[94] - qe_list[5]) / 2
        k_err = (k_list[94] - k_list[5]) / 2
    return qe_err, k_err

# Result visualisation
def plot_single_data(t, qt, C0, Cs, params, units):
    t_model = np.linspace(0, max(t)*1.1, 100)
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs[0,0].scatter(t, qt, label = 'experimental')
    axs[0,0].plot(t_model, ZO_getQ(t_model, params[0]), label = 'model')
    axs[0,0].legend(loc='lower right')
    axs[0,0].set_xlabel(f'Time t ({units[0]})')
    axs[0,0].set_ylabel(f'Absorbate concentration qt ({units[1]})')
    axs[0,0].set_title('linear ZO')
    axs[0,1].scatter(t, qt, label = 'experimental')
    axs[0,1].plot(t_model, PFO_getQ(t_model, params[1], params[2]), label = 'model')
    axs[0,1].legend(loc='lower right')
    axs[0,1].set_xlabel(f'Time t ({units[0]})')
    axs[0,1].set_ylabel(f'Absorbate concentration qt ({units[1]})')
    axs[0,1].set_title('linear PFO')
    axs[0,2].scatter(t, qt, label = 'experimental')
    axs[0,2].plot(t_model, PFO_getQ(t_model, params[3], params[4]), label = 'model')
    axs[0,2].legend(loc='lower right')
    axs[0,2].set_xlabel(f'Time t ({units[0]})')
    axs[0,2].set_ylabel(f'Absorbate concentration qt ({units[1]})')
    axs[0,2].set_title('nonlinear PFO')
    axs[1,0].scatter(t, qt, label = 'experimental')
    axs[1,0].plot(t_model, PSO_getQ(t_model, params[5], params[6]), label = 'model')
    axs[1,0].legend(loc='lower right')
    axs[1,0].set_xlabel(f'Time t ({units[0]})')
    axs[1,0].set_ylabel(f'Absorbate concentration qt ({units[1]})')
    axs[1,0].set_title('linear PSO')
    axs[1,1].scatter(t, qt, label = 'experimental')
    axs[1,1].plot(t_model, PSO_getQ(t_model, params[7], params[8]), label = 'model')
    axs[1,1].legend(loc='lower right')
    axs[1,1].set_xlabel(f'Time t ({units[0]})')
    axs[1,1].set_ylabel(f'Absorbate concentration qt ({units[1]})')
    axs[1,1].set_title('nonlinear PSO')
    axs[1,2].scatter(t, qt, label = 'experimental')
    axs[1,2].plot(t_model, rPSO_getQ(t_model, params[9], params[10], C0, Cs), label = 'model')
    axs[1,2].legend(loc='lower right')
    axs[1,2].set_xlabel(f'Time t ({units[0]})')
    axs[1,2].set_ylabel(f'Absorbate concentration qt ({units[1]})')
    axs[1,2].set_title('nonlinear rPSO')
    plt.tight_layout()
    plt.savefig('../result/result_image.png')

def plot_multi_data(t, qt, C0, ini_rate, order, r_pred, params, units):
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    axs[0].scatter(np.log(C0), np.log(ini_rate))
    axs[0].plot(np.log(C0), r_pred)
    axs[0].set_xlabel(f'log C0 ({units[2]})')
    axs[0].set_ylabel(f'log rate ({units[4]})')
    axs[0].set_title('Initial Rate vs. Initial Concentration')
    if round(order) == 0:
        for i in range(len(qt)):
            t_model = np.linspace(0, max(t[i])*1.1, 100)
            axs[1].scatter(t[i], qt[i], label = f'experimental data_{i+1} ')
            axs[1].plot(t_model, ZO_getQ(t_model, params[i]), label = f'model data_{i+1} ')
        axs[1].legend(loc='lower right')
        axs[1].set_xlabel(f'Time t ({units[0]})')
        axs[1].set_ylabel(f'Absorbate concentration qt ({units[1]})')
        axs[1].set_title('linear ZO')
    elif round(order) == 1:
        for i in range(len(qt)):
            t_model = np.linspace(0, max(t[i])*1.1, 100)
            axs[1].scatter(t[i], qt[i], label = f'experimental data_{i+1}')
            axs[1].plot(t_model, PFO_getQ(t_model, params[2*i], params[2*i+1]), label = f'model data_{i+1}')
        axs[1].legend(loc='lower right')
        axs[1].set_xlabel(f'Time t ({units[0]})')
        axs[1].set_ylabel(f'Absorbate concentration qt ({units[1]})')
        axs[1].set_title('nonlinear PFO')
    elif round(order) == 2:
        for i in range(len(qt)):
            t_model = np.linspace(0, max(t[i])*1.1, 100)
            axs[1].scatter(t[i], qt[i], label = f'experimental data_{i+1}')
            axs[1].plot(t_model, PSO_getQ(t_model, params[2*i], params[2*i+1]), label = f'model data_{i+1}')
        axs[1].legend(loc='lower right')
        axs[1].set_xlabel(f'Time t ({units[0]})')
        axs[1].set_ylabel(f'Absorbate concentration qt ({units[1]})')
        axs[1].set_title('nonlinear PSO')
    else:
        for i in range(len(qt)):
            axs[1].scatter(t[i], qt[i], label = f'dataset{i+1}')
        axs[1].legend(loc='lower right')
        axs[1].set_xlabel(f'Time t ({units[0]})')
        axs[1].set_ylabel(f'Absorbate concentration qt ({units[1]})')
        axs[1].set_title('experimental data')
    plt.tight_layout()
    plt.savefig('../result/result_image.png')