# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 15:44:28 2020

@author: Robert
"""

import csv
import math
import numpy as np

import time

import HEmodel

import cantera as ct
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.axes_grid1 import make_axes_locatable


#######################Reactor details########################################
#Initial design case set at 823 K and 1500 kPa
#For testing against the University of Ghent packed bed test cases

#Packed bed geometry:
    # height: 0.50355 m
    # width: 0.4521 m
    # length: 0.20 m
#Catalyst pellet geometry
    # diameter: 0.005 m
    # coating thickness: 0.0004 m
    # core material: Al2O3
    # core material density: 3950 kg/m³
    # coating density: 6110 kg/m³
    # pellet density (coating+core): 3506.25 kg/m³
#Gas phase properties:
    # kinetics and thermo: Pd/CO3O4
    # density: ideal gas law
#Solid phase properties:
    # kinetics: Pd/CO3O4
    # density: 3506.25 kg/m³
    # diameter: 0.0005 m / 500 micron
    # specific heat capacity: 5 J/kgK #This is used to reach a fast thermo-equilibrium, only useful to shorten simulation time
    # Volume fraction (1-epsilon): 0.5
    # porosity: 0
#Boundary conditions:
    # inlet composition: CO2/O2/H2O/CH4 = 0.971381/0.007523/0.019216/0.001881
    # inlet temperature: 823K (gas phase only)
    # inlet velocity: 4.427545835 m/s
    # inlet pressure: 15 bar
    # adiabatic walls set at the beginning and end of the packed bed reactor

#Test case temperature and pressure
T1 = 843 #[K]
T_in = 843 #[K] Inlet temperature
P = 1000000 #[Pa]

#Reactor information
r_length = 0.20 #[m] Packed bed length
# r_i_diameter = 0.005 #[m] Packed bed diameter
r_i_height = 0.05 #[m]
r_i_width = 0.075 #[m]
velocity = 5.082990544 #[m/s] space velocity (superficial)
porosity = 0.45 #[m³space/m³reactor] Packed bed porosity
r_i_area = r_i_height * r_i_width #[m²] Packed bed cross-sectional area

cat_porosity = 0.27 #Porosity of the catalyst, this is an assumption

#Importing kinetics and properties file
#Pre-exponential in kinetics file need to be modified each time to accomodate for catalyst mass
cti_file = 'pdc - ualberta.cti'

# PBR will be simulated by a chain of CSTRs
n_reactors = 50

#######################Reactor details########################################

#######################Catalyst details#######################################
#Using 500 micron spherical particles
p_diameter = 0.005 #[m] particle diameter

#This is not necessary for this simulation, there are no surface reactions
cat_area_per_vol = 5.9e6 #the catalytic surface area per solid volume, this property is irrelevant to the kinetics used in this simulation

#Packed bed surface area, cited from https://www.sciencedirect.com/topics/engineering/ergun-equation
#Using 5000 micron spherical particles
A_p = 4*(p_diameter/2)**2*math.pi #[m²] particle surface area
V_p = 4/3 * (p_diameter/2)**3 * math.pi #[m³] particle volume
S = (1-porosity)/V_p * A_p #[m²_particle/m³_bed] specific surface area of the packed bed

#Characteristic length of catalyst particles
d_char = V_p / A_p

#Surface-volume particle diameter
d_sv = 6* V_p / A_p #Sauter mean diameter

#Catalyst pellet properties
#From test case
cp_cat = 880 #[J/kgK] Using specific heat capacity of alumina
k_cat = 25 #[W/mK] Using conductivity of alumina
rho_cat = 3950 #[kg_pellet/m³_pellet] Uses alumina density here

epsilon_r = 0.5 #[-] Pellet surface emissivity, half of all radiation is absorbed and half is refrlected


#######################Simulation set-up######################################
#Import gas model and set the initial conditions
bulk_gas = ct.Solution(cti_file, 'gas') #Phase object to represent the gas phase
surf_gas = ct.Solution(cti_file, 'gas') #Phase object to represent the solid "gas" phase

#Setting initial conditions for both phase objects
bulk_gas.TPY = T1, P, 'CH4:0.001881, O2:0.007523, H2O:0.019216, CO2:0.971381' #based on 2 vol% O2, adding equilmolar CH4 to consume half of total O2
surf_gas.TPY = T1, P, 'CH4:0.001881, O2:0.007523, H2O:0.019216, CO2:0.971381' #based on 2 vol% O2, adding equilmolar CH4 to consume half of total O2

# import the surface model
surf = ct.Interface(cti_file,'surface1', [surf_gas])
surf.TP = T1, P #Solid phase set to be same T & P as gas phase to shorten simulation time, ie assuming packed bed is preheated

#Simulated reactor information
dx = (r_length)/(n_reactors) #Length of one CSTR

#Volume of 1 finite element
r_volume = dx * r_i_area

#catalytic surface area of one reactor (finite volume)
cat_area = cat_area_per_vol * r_volume * (1-porosity) #This property is irrelevant to the kinetics used in this simulation

#######################Simulation set-up######################################

#######################Equation definition####################################

#Making list of molar weights for gas phase
MM_g = bulk_gas.molecular_weights

# MM = np.hstack((surf_gas.molecular_weights,surf.molecular_weights)) #array of molecular weights for all species

x_axis = np.arange(0, r_length+dx, dx) #x-axis variable used for plotting results

#To calculate gas phase heat capacity using cantera
def cp_eq(T, p, Y):
    # print(T)
    surf_gas.TPY = T, p, Y
    return surf_gas.cp_mass

def rho_eq_(T, p, Y):
    y_mm = np.divide(Y,MM_g)
    avg_MM = 1/np.sum(y_mm)
    rho = p * avg_MM / (8.314 * T * 1000)
    return rho

def mu_eq(T, p, Y):
    surf_gas.TPY = T, p, Y
    return surf_gas.viscosity

def k_eq(T, p, Y):
    surf_gas.TPY = T, p, Y
    return surf_gas.thermal_conductivity

def R_eq(T_g_r, T_s_r, p_r, Y_g_r, Y_s_r, Y_surf_r):
    
    #Setting up reaction phase conditions
    bulk_gas.TPY = T_g_r, p_r, Y_g_r #Setting up gas phase T, P and species fraction, T_g is used because the reaction temperature should match the gas phase temperature
    surf_gas.TPY = T_s_r, p_r, Y_surf_r #Setting up solid phase T, P and species fraction, T_s is used because the reaction temperature should match the solid phase temperature

    #Setting up solid reactor simulation
    rsurf.coverages = Y_s_r #Setting up occupied active site fraction by surface species in the solid phase
    # rsurf.area = cat_area

    r.volume = r_volume #Setting up reactor volume
    r.syncState() #Sync reactor state to that of the one set
    sim = ct.ReactorNet([r]) #Creating solid reactor simulation

    #Solid reactor info prior to reaction
    m_c = r.mass*r.Y #Array holding mass of each gas phase species before reaction
    coverage_b = rsurf.coverages #Occupied active site fraction by surface species in the solid phase before reaction
    e_b = r.thermo.cp_mass*r.thermo.T*r.mass #Total enthalpy of the reactor before reaction [J]

    # m_b = r.mass #Solid reactor mass before reaction [kg]

    #Integrating the solid reactor object by the timestep dt
    sim.atol = 1E-10 #Absolute tolerance of 1E-10 is used, this is smaller than the default tolerance to allow the integration to occur
    sim.advance(dt) #Integrating the Cantera batch reactor by one time step #Solid reactor

    #Solid reactor info post reaction
    coverage_p = rsurf.coverages #Occupied active site fraction of surface species in the solid phase after reaction
    e_p = r.thermo.cp_mass*r.thermo.T*r.mass #Total solid reactor enthalpy of the reactor after reaction [J]

    #Add in a cpT dm/dt term to the q term as a correction (applying multiplication rule for differentiation)

    #Change in solid reactor in energy, gas phase species, and surface coverage
    q = (e_p - e_b)/ (r_volume*dt)#Calculating enthalpy change in the solid reactor due to reaction per unit volume [J/m³s], this is the energy generation due to reaction
    rxn_g = (r.mass*r.Y-m_c)/(r.mass*dt) #Calculating change in gas phase species mass fraction due to reaction [-/s], mass used is from after reaction
    rxn_s = (coverage_p-coverage_b)/dt #Calculating change in occupied active site fraction by surface species in the solid phase due to reaction [-/s]

    #Setting up gas reactor simulation
    gas_r.volume = r_volume
    gas_r.syncState()
    gas_sim = ct.ReactorNet([gas_r])
    
    #Gas reactor info prior to reaction
    m_c_g = gas_r.mass*gas_r.Y #Array holding mass of each gas phase species before reaction
    e_b_g = gas_r.thermo.cp_mass*gas_r.thermo.T*gas_r.mass #Total enthalpy of the reactor before reaction [J]

    # m_b_g = gas_r.mass #Gas reactor mass before reaction [kg]

    #Integrating the gas reactor object by the timestep dt
    gas_sim.atol = 1E-10
    gas_sim.advance(dt) #Integrating the Cantera batch reactor by one time steop #Gas reactor

    #Gas reactor info post reaction
    e_p_g = gas_r.thermo.cp_mass*gas_r.thermo.T*gas_r.mass #Total gas reactor enthalpy of the reactor after reaction [J]
    
    #Change in gas reactor in energy, gas phase species
    gas_q = (e_p_g - e_b_g)/ (r_volume*dt) #Calculating enthalpy change in the gas reactor due to reaction per unit volume [J/m³s], this is the energy generation due to reaction
    rxn_gas = (gas_r.mass*gas_r.Y-m_c_g)/(gas_r.mass*dt) #Calculating change in gas phase species mass fraction due to reaction [-/s], mass used is from after reaction

    return np.hstack((q, rxn_g, rxn_s, gas_q, rxn_gas))

#Calculating for radiation heat transfer from solid phase to gas phase
def q_rad(surf_T, epsilon):
    """
    Parameters
    ----------
    surf_T : Float, units of K
        Solid phase temperature
    epsilon : float, units of [-]
        Packed bed porosity
    Returns
    -------
    q_rad : Float, units of W/m²
        Radiation energy transfer
    Equation
    ---------
    .. math::
        S_r = 1 + 1.84 * (1 - \epsilon) + 3.15 * (1 - \epsilon)^2 + 7.2 * (1 - \epsilon)^3 for \space \epsilon > 0.3\n
        \\beta = 1.5 * \epsilon_r * (1 - \epsilon) * S_r / d_{sv} \n
        q^{''}_{rad} = -16 * \sigma * T_s^3 / (3 * \\beta)
    >>> where:
        S_r is the scaling factor [m]
        epsilon is the porosity [-]
        epsilon_r is the emissivity [-]
        d_sv is the surface to volume diameter [m]
        beta is the extinction coefficient [-]
        sigma is the Stefan-Boltzmann constant [W/(m²·K^-4)]
        k_g is the gas phase thermal conductivity [W/mK]
        c_p_g is the gas phase constant pressure heat capacity [J/kg]
        
    Applicability
    -------------
    Used to calculate radiation heat transfer (without the dT/dx) [W/m²].
    """
    sigma_SB = 5.67E-8 #[W/(m^2*K^-4)] Stefan-Boltzmann constant
    #scaling factor for porosity > 0.3
    S_r = 1 + 1.84 * (1 - epsilon) + 3.15 * (1 - epsilon)**2 + 7.2 * (1 - epsilon)**3
    #extinction coefficient
    beta_e = 1.5 * epsilon_r * (1 - porosity) * S_r / d_sv
    q_rad = 16 * sigma_SB * surf_T**3 / (3 * beta_e)
    return q_rad

#Calculating diffusion coefficient of species 1 in species 2
# def D_eq(T, p, MM1, MM2, sigma, omega):
#     #(conversion from cm²/s to m²/s) #CH4 in N2, [m²/s]
#     D = 1.858E-3 * T**(3/2) * (1/MM1 + 1/MM2)**(1/2) / ( (p/101325) * sigma**2 * omega) * 1/10000
#     return D

#Calculating omega of CH4/O2 in N2 using a linear equation, this is to change the diffusivity as temperature increases
#Obtained by linearizing the collision integral table over T = 750-1500
# def omega_eq(kTe):
#     return - 0.0151 * kTe + 0.9009

#Calculating inter-phase heat transfer coefficient
def h_eq(rho, u, cp, mu, k):

    """
    Parameters
    ----------
    u : Float, units of m/s
        Superficial gas velocity
    Returns
    -------
    h_inter : Float, units of W/m²·K
        Interphase heat transfer coefficient
    Equation
    ---------
    .. math::
        Re = \\rho_g * u / (S * f * \mu_g) \n
        J_H = 0.91 * f * Re^{-0.51} for \space 0.01 < Re < 50 \n
        J_H = 0.61 * f * Re^{-0.41} for \space 50 < Re \n
        h_{inter} = J_H * \\rho_g * u * c_{p_{g}} * (\mu_g * c_{p_{g}} / k_g) ^ {-2/3}
    >>> where:
        Re is the Reynolds number
        J_H is the Colburn J-Factor
        f is the shape factor of the catalyst pellet, 0.91 for cylindrical pellets [-]
        u is the superficial gas velocity [m/s]
        S is the specific surface area of the packed bed [m²/m³]
        mu is the gas viscosity [Pa·s]
        rho_g is the gas phase density [kg/m³]
        k_g is the gas phase thermal conductivity [W/mK]
        c_p_g is the gas phase constant pressure heat capacity [J/kg]
        
    Applicability
    -------------
    For use to calculate interphase heat transfer coefficient.
    """
    #Calculating interphase heat transfer coefficient
    #Gunn analogy used for this simulation
    Re = rho * u * d_char / (mu*porosity)
    Pr = cp * mu / k
    Nu_s = (7 - 10*porosity + 5*porosity**2) * (1 + 0.7*Re**0.2*Pr**(1/3)) + (1.33 - 2.4*porosity + 1.2*porosity**2)*Re**0.7*Pr**(1/3)
    h_inter_ = Nu_s * k / d_char #[W/m²]
    return h_inter_

def u_eq(rho_g_):
    #Calculating the gas flow velocity using the conservation of mass [m/s]
    return mass_flow_rate / (r_i_area * rho_g_)

def p_eq(p, u_, mu_g):
    #Using the Ergun equation to calculate pressure drop [Pa]
    return p - (150 * mu_g/d_sv**2 * (1-porosity)**2/porosity**3*u_ + 1.75*mass_flow_rate/(d_sv*r_i_area) * (1-porosity)/porosity**3*u_)*dx

#For calculating mass conservation
def y_eq (y_g, y_g_prev, rho_g, s_mx, r_p, r_g, r_s):
    #Calculating advection term
    np.subtract(y_g, y_g_prev)/rho_g
    m_adv = mass_flow_rate/r_i_area * np.subtract(y_g, y_g_prev)/rho_g/dx
    
    # Calculating the accumulation term in the gas phase by adding up advection, reaction and interphase mass transfer terms
    dcdt_g = r_g - m_adv + s_mx / rho_g
    
    #Calculating the accumulation term in the pore phase by adding up reaction and interphase mass transfer term
    dcdt_p = r_p - s_mx/rho_g
    
    #Calculating the accumulation term in the solid phase by adding up reaction term
    dcdt_s = r_s
    return np.concatenate((dcdt_g, dcdt_p, dcdt_s))

def k_inter_eq(T, p, rho, u, cp, mu, k):
    #Calculating interphase mass transfer coefficient
    #Gunn analogy used here
    D = mu/rho #As stated in supporting information of the University of Ghent paper http://pubs.acs.org/doi/suppl/10.1021/acs.energyfuels.0c02824/suppl_file/ef0c02824_si_001.pdf
    Re = rho * u * d_char / (mu*porosity)
    Pr = cp * mu / k
    Sc = 1 #Set to 1 as stated in the supporting information
    Nu_s = (7 - 10*porosity + 5*porosity**2) * (1 + 0.7*Re**0.2*Pr**(1/3)) + (1.33 - 2.4*porosity + 1.2*porosity**2)*Re**0.7*Pr**(1/3) #Gunn heat transfer analogy
    
    Sh = Nu_s * (Sc/Pr)**(1/3)
    k_inter = Sh * D / d_char

    k_ = k_inter * np.ones(bulk_gas.n_species)
    return k_

def smx_eq(k_inter,rho_g, rho_s, y_g, y_s):
    #Calculating interphase mass transfer source term
    s_mx = k_inter * S * (rho_g*y_s - rho_g*y_g)
    return s_mx

#######################Equation definition####################################

#######################Variable declaration###################################

#Declare a cantera batch reactor object for solid phase, install a surface
r = ct.IdealGasConstPressureReactor(surf_gas, energy="on")
r.volume = r_volume
rsurf = ct.ReactorSurface(surf, r, A=cat_area)

#Declare a cantera batch reactor object for gas phase
gas_r = ct.IdealGasConstPressureReactor(bulk_gas, energy="on")
gas_r.volume = r_volume

#Inlet states
#Bulk gas properties is used as it's the bulk gas phase
Y_g_in = bulk_gas.Y #setting up inlet gas phase composition
cp_in = bulk_gas.cp_mass #set up inlet gas phase heat capacity [J/kgK]
H_in = bulk_gas.density_mass * cp_in * T_in #set up inlet gas phase enthalpy [kg/m³] * [J/kg]
rho_g_in = rho_eq_(T1, P, Y_g_in) #setting up inlet gas density

#Use these if the initial state is different than the inlet states
surf_gas.TPY = T1, P, 'CH4:0.001881, O2:0.007523, H2O:0.019216, CO2:0.971381' #based on 0.35 CH4 mole equivalence ratio to O2
# bulk_gas.TPY = T1, P, 'N2:1'

#Initial states arrays
Y_g_0 = bulk_gas.Y #initial gas phase composition
Y_surf = Y_g_0.copy() #Setting up array to track catalyst pore phase mass fractions
Y_s_0 = rsurf.coverages #mass fractions in the solid phase tracked as fraction of catalytic sites occupied by each species
Y0 = np.hstack((Y_g_0, Y_surf, Y_s_0)) #setting the initial gas, pore, and solid phase composition

#Setting up initial conditions across all cells
T_g_0 = T1 * np.ones(n_reactors) #setting up initial gas phase temperature of all cells
T_s_0 = T1 * np.ones(n_reactors) #setting up initial solid phase temperature of all cells
rho_g_0 = rho_eq_(T1, P, Y_g_0) #setting up initial gas phase density
cp_0 = bulk_gas.cp_mass #setting up initial gas phase heat capacity
k_0 = bulk_gas.thermal_conductivity #setting up initial gas phase thermal conductivity
h_0 = np.multiply(cp_0, T1) * np.ones(n_reactors) #setting up initial gas phase enthalpy
H_0 = np.multiply(rho_g_0, h_0) #setting up initial gas phase rho*H [kg/m³] * [J/kg]
mu_0 = mu_eq(T1, P, Y_g_0) #setting up initial gas phase viscosity

#Tracking states, global variables that are used to track the properties of the gas phase in each cell
u_ = velocity * np.ones(n_reactors) #setting up gas phase velocity of all cells (used for tracking)
p_ = P * np.ones(n_reactors) #setting up gas phase pressure of all cells (used for tracking)
rho_g_ = rho_g_0 * np.ones(n_reactors) #setting up gas phase density of all cells (used for tracking)
cp_g_ = cp_0 * np.ones(n_reactors) #setting up gas phase heat capacity of all cells (used for tracking)
mu_g = mu_0 * np.ones(n_reactors) #Setting up gas phase viscosity of all cells (used for tracking)
k_g = np.ones(n_reactors) #Setting up gas phase thermal conductivity of all cells (used for tracking)

rho_s_ = rho_g_0 * np.ones(n_reactors) #setting up gas phase density of all cells (used for tracking)

#Initial conditions used for solving PDE
Y = np.hstack([Y0[:] for i in range (n_reactors)]) #initial gas and solid phase composition for all reactors
y0 = np.hstack((H_0, T_s_0, Y)) #setting up array to pass into PDE

mass_flow_rate = velocity * rho_g_in * r_i_area #defining mass flowrate

#print(mass_flow_rate)

rad_cont = np.zeros(n_reactors)
cond_cont = np.zeros(n_reactors)
conv_cont = np.zeros(n_reactors)
tot_cont = np.zeros(n_reactors)
q_tracker = np.zeros(n_reactors)

#Turning off gas phase reactions in the solid "gas" phase
surf_gas.set_multiplier(0)
#######################Variable declaration###################################

def consv_eqs(dt, y_):
    
# =============================================================================
#     Radiation was turned ON at PARTICLE EMISSIVITY = 1 for this simulation
# =============================================================================
    
    solution = np.zeros((n_reactors, 2)) #Solution array for holding rho_g*H and T_s
    c_solution = np.zeros((n_reactors, surf.n_total_species)) #Solution array for holding mass fraction (both gas and solid phase)

    H_ = y_[:n_reactors] #Gas phase rho_g * H_g
    
    T_s = y_[n_reactors:n_reactors*2] #Solid phase temperature

    # c_ = np.reshape(y_[n_reactors*2:], (-1, surf.n_total_species)) #Gas phase mass fraction and solid phase coverage fraction
    
    c_ = np.reshape(y_[n_reactors*2:], (-1, (bulk_gas.n_total_species+surf.n_total_species))) #Gas phase mass fraction, catalyst pore phase mass fraction, and solid phase coverage fraction
    
    c_g = c_.T[:][0:bulk_gas.n_total_species] #Gas phase mass fraction
    c_pore = c_.T[:][bulk_gas.n_total_species:(bulk_gas.n_total_species+surf_gas.n_total_species)] #Catalyst pore phase mass fraction
    c_s = c_.T[:][(bulk_gas.n_total_species+surf_gas.n_total_species):] #Solid phase coverage fraction

    # c_g = c_.T[:][0:surf_gas.n_total_species] #Gas phase mass fraction
    # c_s = c_.T[:][surf_gas.n_total_species:] #Solid phase coverage fraction

    global rho_g_, cp_g_, mu_g, k_g, u_, p_#, rad_cont

    T_g = np.divide(H_, np.multiply(rho_g_,cp_g_)) #Calculating gas phase temperature from rho_g*H_g

    Y_g = c_g.T.copy()
    Y_pore = c_pore.T.copy()
    Y_s = c_s.T.copy()

    #Calculating gas phase properties
    cp_g_ = list(map(cp_eq, T_g, p_, Y_g)) #Calculating all species gas phase heat capacities across all cells
    mu_g = list(map(mu_eq,T_g, p_, Y_g)) #Calculating all species viscosity across all cells
    k_g = list(map(k_eq,T_g, p_, Y_g)) #Calculating all species thermal conductivity across all cells
    h_inter = list(map(h_eq, rho_g_, u_, cp_g_, mu_g, k_g)) #Calculating convective heat transfer coefficient across all cells
    rho_g_ = list(map(rho_eq_, T_g, p_, Y_g)) #Calculating bulk gas phase density across all cells

    rho_s_ = list(map(rho_eq_, T_s, p_, Y_pore))

    #Calculating interphase mass transfer
    k_inter = list(map(k_inter_eq, T_g, p_, rho_g_, u_, cp_g_, mu_g, k_g)) #Calculating interphase mass transfer coefficient across all cells
    s_mx = np.array(list(map(smx_eq, k_inter, rho_g_, rho_s_, Y_g, Y_pore))) #Mass change from interphase exchange

    #Calculating terms related to reaction
    R = np.array(list(map(R_eq, T_g, T_s, p_, Y_g, Y_s, Y_pore))) #Calculating all species reaction rates and heat of reaction across all cells
    
    #Re-arranging array to separate into heat of reaction, gas phase and solid phase reaction rate
    q = R.T[:][0] #Rate of change for energy in solid reactor
    rxn_g = R.T[1:surf_gas.n_species+1][:].T #Variable used to track gas phase species rate of change in solid reactor
    rxn_s = R.T[surf_gas.n_species+1:surf.n_total_species+1][:].T  #Variable used to track solid phase coverage fraction rate of change in solid reactor
    gas_q = R.T[surf.n_total_species+1:surf.n_total_species+2][:].T #Rate of change for energy in gas reactor
    rxn_gas = R.T[surf.n_total_species+2:][:].T #Variable used to track gas phase species rate of change in gas reactor

    #Momentum conservation
    prev_P = np.hstack((P, p_[0:-1])) #Making new array that contains inlet pressure and all pressures except for last cell
    
    u_next = np.array(list(map(u_eq, rho_g_))) #Calculating gas velocity across all cells
    p_next = np.array(list(map(p_eq, prev_P, u_, mu_g))) #Calculating pressure across all cells
    
    #Setting velocity and pressure to the calculated values
    u_ = u_next
    p_ = p_next
    
    prev_y = np.vstack((Y_g_in, Y_g[0:-1])) #Making new array that contains inlet mass fraction and all mass fractions except for last cell
    
    #Mass conservation, d/dt (c_g_i[i])
    #Evaluating mass conservation equations for gas and solid phase
    c_solution = np.array(list(map(y_eq, Y_g, prev_y, rho_g_, s_mx, rxn_g, rxn_gas, rxn_s)))

    #cell 1##########
    #gas phase
    eg_adv = (H_[0]*u_[0]-velocity*H_in)/(dx) #Calculating for advection transfer
    inter_g = h_inter[0]*(T_s[0]-T_g[0])*S #Calculating for interphase heat transfer
    dH_gdt = (-eg_adv + inter_g + gas_q[0]) / (porosity) #Calculating for accumulation

    #Solid, (1-epsilon)*rho_s * d/dt(H_s[i])
    rad_s = q_rad(T_s[0],porosity) * (T_s[1]-T_s[0])/dx**2 #Calculating for radiation transfer, Adiabatic wall BC
    es_cond = (1-porosity)*(1-cat_porosity) * k_cat * (T_s[1]-T_s[0])/dx**2 #Calculating for conduction transfer, Adiabatic wall BC
    inter_s = h_inter[0]*(T_g[0]-T_s[0])*S #Calculating for interphase heat transfer
    dT_sdt = (es_cond + rad_s + inter_s + q[0]) / ((1-porosity)*(1-cat_porosity)*rho_cat*cp_cat) #Calculating for accumulation
    solution[0][0] = dH_gdt
    solution[0][1] = dT_sdt

    #cells 1 to n-1
    for i in range (1, n_reactors-1):
        #Energy conservation
        #Gas, epsilon * d/dt(rho_g[i]*H_g[i])
        eg_adv = (H_[i]*u_[i]-H_[i-1]*u_[i-1])/dx #Calculating for advection transfer
        inter_g = h_inter[i]*(T_s[i]-T_g[i])*S #Calculating for interphase heat transfer
        dH_gdt = (-eg_adv + inter_g + gas_q[i]) / (porosity) #Calculating for accumulation
        
        #Solid, (1-epsilon)*rho_s * d/dt(H_s[i])
        rad_s = q_rad(T_s[i],porosity) * (T_s[i+1]-2*T_s[i]+T_s[i-1])/dx**2 #Calculating for radiation transfer, Adiabatic wall BC
        es_cond = (1-porosity)*(1-cat_porosity) * k_cat * (T_s[i+1]-2*T_s[i]+T_s[i-1])/dx**2 #Calculating for conduction transfer, Adiabatic wall BC
        inter_s = h_inter[i]*(T_g[i]-T_s[i])*S #Calculating for interphase heat transfer
        dT_sdt = (es_cond + rad_s + inter_s + q[i]) / ((1-porosity)*(1-cat_porosity)*rho_cat*cp_cat) #Calculating for accumulation
    
        solution[i][0] = dH_gdt
        solution[i][1] = dT_sdt

    #cell n
    #Energy conservation
    #Gas, epsilon * d/dt(rho_g[i]*H_g[i])
    eg_adv = (H_[-1]*u_[-1] - H_[-2]*u_[-2]) / dx #Calculating for advection transfer
    inter_g = h_inter[-1]*(T_s[-1]-T_g[-1])*S #Calculating for interphase heat transfer
    dH_gdt = (-eg_adv + inter_g + gas_q[-1]) / (porosity) #Calculating for accumulation
    
    #Solid, (1-epsilon)*rho_s * d/dt(H_s[i])
    rad_s = q_rad(T_s[-1],porosity) * (T_s[n_reactors-2]-T_s[-1])/dx**2 #Adiabatic wall BC
    es_cond = (1-porosity)*(1-cat_porosity) * k_cat * (T_s[n_reactors-2]-T_s[-1])/dx**2 #Adiabatic wall BC
    inter_s = h_inter[-1]*(T_g[-1]-T_s[-1])*S #Calculating for interphase heat transfer
    dT_sdt = (es_cond + rad_s + inter_s + q[-1]) / ((1-porosity)*(1-cat_porosity)*rho_cat*cp_cat) #Calculating for accumulation
    
    solution[-1][0] = dH_gdt
    solution[-1][1] = dT_sdt

    states_ = np.ravel(solution, 'F')
    mass_c = np.ravel(c_solution, 'C')
    
    y_out = np.concatenate((states_,mass_c))
    return y_out

def euler_ (fun, y0, dt):

    # #Returning d/dt
    dy_e = fun(dt, y0)
    # print(dy_e)
    
    new_y = np.add(y0, np.multiply(dy_e,dt))
    for i in range (4):
        dy_i = fun(dt, new_y)
        new_new_y = np.add(y0, np.multiply(np.add(dy_e,dy_i)/2,dt))
        new_y = new_new_y
    return new_y


#==========================Time step and solution arrays=======================
t = 0
t_span = 10

dt = 5e-6 #Time step size

t_arr = []
y_arr = []

solArr = []

counter = 0
plotcount = 0
HE_plotcount = 0
#==========================Time step and solution arrays=======================

#Number of channels
n_channel = 5

#Opening .csv output file
f = open ('results.csv', 'w')
wtr = csv.writer(f,lineterminator='\n')

f1 = open ('HE_results.csv', 'w')
HE_wtr = csv.writer(f1,lineterminator='\n')

#Header for csv output file
wtr.writerow(np.concatenate(('Time', ['T_g' for i in range (n_reactors)],['T_s' for i in range (n_reactors)],
                             [np.concatenate((bulk_gas.species_names,surf_gas.species_names,
                                              surf.species_names)) for i in range (n_reactors)], 'out_P'),axis=None))
HE_wtr.writerow(np.concatenate(('Time', ['T_U_1' for i in range (n_channel)], ['T_P_2' for i in range (n_channel)], 
                               ['T_F_3' for i in range (n_channel)], ['T_P_4' for i in range (n_channel)], 
                               ['T_U_5' for i in range (n_channel)], ['T_U_P_1' for i in range (n_channel)], 
                               ['T_P_P_2' for i in range (n_channel)], ['T_F_P_3' for i in range (n_channel)], 
                               ['T_P_P_4' for i in range (n_channel)], ['T_U_P_5' for i in range (n_channel)], 'out_P'), axis=None))
    

#=================================HE SECTION===================================
'''
    Parameters
    -----
    reactant_in : list of four elements
        0. reactant composition, dict of compositions, units of mass%
        1. mass flow rate through reactant plate, units of kg/s 
        2. reactant inlet temperature, units K        
        3. reactant inlet absolute pressure, units of Pa
    
    utility_in : list of four elements
        0. utility composition, dict of compositions, units of mass%        
        1. mass flow rate through utility plate, units of kg/s         
        2. utility inlet temperature, units K        
        3. utility inlet absolute pressure, units of Pa
        
    dims : list of five elements
        0. reactant channel diameter, units of m
        1. utility channel diameter, units of m
        2. number of reactant channels, dimensionless
        3. number of utility channels, dimensionless
        4. wall thickness between channels, units of m
        5. plate thickness, units of m
'''


reactant_inlet = [{'CO2':1}, 0.00702/n_channel, 313, 1000000]
#Utility and fuel inlet pressures subject to change
utility_inlet = [{'CO2':1}, 0.005, 313, 1000000]
fuel_inlet = [{'CH4':1}, 0.0001, 313, 1000000]
dimensions = [0.003, 0.003, 5, 5, 0.012, 0.01]

# exchanger = HEmodel.crossflow_PCHE(reactant_inlet, utility_inlet, fuel_inlet, dimensions)

#Initial plate temperature to match inlet fluid temperature, does not affect steady-state result
initial_T_reactant = reactant_inlet[2]*np.ones((dimensions[2], dimensions[3]))
initial_T_reactantPlate = reactant_inlet[2]*np.ones((dimensions[2], dimensions[3]))
initial_T_utility = utility_inlet[2]*np.ones((dimensions[2], dimensions[3]))
initial_T_utilityPlate = utility_inlet[2]*np.ones((dimensions[2], dimensions[3]))
initial_T_fuel = fuel_inlet[2]*np.ones((dimensions[2], dimensions[3]))
initial_T_fuelPlate = fuel_inlet[2]*np.ones((dimensions[2], dimensions[3]))

initial_temps = np.concatenate([initial_T_utility.ravel(), initial_T_reactant.ravel(), initial_T_fuel.ravel(), initial_T_reactant.ravel(), initial_T_utility.ravel(),
                        initial_T_utilityPlate.ravel(), initial_T_reactantPlate.ravel(), initial_T_fuelPlate.ravel(), initial_T_reactantPlate.ravel(), initial_T_utilityPlate.ravel()])

#=================================HE SECTION===================================



while t < t_span:

        #Solving reactor solution
        RXN_y = euler_(consv_eqs,y0,dt)
        #=============================================================
        #NEED TO USE EULER, SOLVE_IVP DOES NOT PRODUCE A SMOOTH RESULT
        #=============================================================
        
        #Isolating out the gas phase enthalpy string from reactor solution for calculating gas phase temperature
        H_ = RXN_y[:n_reactors] #Gas phase rho_g * H_g
        
#================================HE SECTION====================================
        T_array = []
        print(r_i_area*rho_g_[-1])
        if (counter > 0) & (counter%100 ==0):
            
            HE_plotcount+=1
            
            #Constructing input string for HE from reactor outlets
            c_ = np.reshape(RXN_y[n_reactors*2:], (-1, (bulk_gas.n_total_species+surf.n_total_species))) #Gas phase mass fraction, catalyst pore phase mass fraction, and solid phase coverage fraction
            c_g = c_[-1][0:bulk_gas.n_total_species] #Gas phase mass fraction
            HE_species = {bulk_gas.species_names[i]: c_g[i] for i in range(len(bulk_gas.species_names))}
            new_reactants = [HE_species, u_[-1]*r_i_area*rho_g_[-1]/n_channel, (np.divide(H_,np.multiply(cp_g_,rho_g_))[-1]), p_[-1]]

            #Creating heat exchanger object
            exchanger = HEmodel.crossflow_PCHE(new_reactants, utility_inlet, fuel_inlet, dimensions)
            
            #Solving heat exchanger
            HE_y = solve_ivp(exchanger.transient_solver, [0, dt*100], initial_temps, method = 'BDF', t_eval = [0, dt*100])
            initial_temps = HE_y['y'][:, -1]
            print("HE solved")
            
            #Calculating heat exchanger pressure drop
            exchanger.update_pressures()
            T_U1, T_P2, T_F3, T_P4, T_U5, T_U_P1, T_P_P2, T_F_P3, T_P_P4, T_U_P5 = HEmodel.convert_T_vector(HE_y['y'][:, -1], dimensions)
            P_array = np.lib.pad(np.hstack((exchanger.reactant2_P.min(),exchanger.utility1_P.min(),exchanger.fuel3_P.min(),
                                 exchanger.reactant4_P.min(),exchanger.utility5_P.min())), (0, n_channel-5), 'constant', constant_values=0)
            
            #Writing to file
            for i in range(n_channel):
                T_array.append(np.concatenate((T_U1[i], T_P2[i], T_F3[i], T_P4[i], T_U5[i], T_U_P1[i], T_P_P2[i], T_F_P3[i], T_P_P4[i], T_U_P5[i])))
            out_array = np.column_stack(([t for i in range(n_channel)], T_array, P_array))
            HE_wtr.writerows(out_array)
            
            
            fig = plt.figure(figsize=(25, 10))

            #Fluid temperature plots
            ax1 = fig.add_subplot(2,5,1)
            m1 = ax1.matshow(np.reshape(np.round(T_P2,1),(dimensions[2],dimensions[2])),interpolation = "nearest")
            ax1.set_title('T_process2')
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(m1, cax=cax, orientation='vertical')

            ax2 = fig.add_subplot(2,5,2)
            m2 = ax2.matshow(np.reshape(np.round(T_U1,1),(dimensions[2],dimensions[2])),interpolation = "nearest")
            ax2.set_title('T_utility1')
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(m2, cax=cax, orientation='vertical')
            
            ax3 = fig.add_subplot(2,5,3)
            m3 = ax3.matshow(np.reshape(np.round(T_F3,1),(dimensions[2],dimensions[2])),interpolation = "nearest")
            ax3.set_title('T_fuel3')
            divider = make_axes_locatable(ax3)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(m3, cax=cax, orientation='vertical')

            ax4 = fig.add_subplot(2,5,4)
            m4 = ax4.matshow(np.reshape(np.round(T_P4,1),(dimensions[2],dimensions[2])),interpolation = "nearest")
            ax4.set_title('T_process4')
            divider = make_axes_locatable(ax4)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(m4, cax=cax, orientation='vertical')
            
            ax5 = fig.add_subplot(2,5,5)
            m5 = ax5.matshow(np.reshape(np.round(T_U5,1),(dimensions[2],dimensions[2])),interpolation = "nearest")
            ax5.set_title('T_utility5')
            divider = make_axes_locatable(ax5)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(m5, cax=cax, orientation='vertical')
            
            #Plate temperature plots
            ax6 = fig.add_subplot(2,5,6)
            m6 = ax6.matshow(np.reshape(np.round(T_P_P2,1),(dimensions[2],dimensions[2])),interpolation = "nearest")
            ax6.set_title('T_process_plate2')
            divider = make_axes_locatable(ax6)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(m6, cax=cax, orientation='vertical')

            ax7 = fig.add_subplot(2,5,7)
            m7 = ax7.matshow(np.reshape(np.round(T_U_P1,1),(dimensions[2],dimensions[2])),interpolation = "nearest")
            ax7.set_title('T_utility_plate1')
            divider = make_axes_locatable(ax7)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(m7, cax=cax, orientation='vertical')
            
            ax8 = fig.add_subplot(2,5,8)
            m8 = ax8.matshow(np.reshape(np.round(T_F_P3,1),(dimensions[2],dimensions[2])),interpolation = "nearest")
            ax8.set_title('T_fuel_plate3')
            divider = make_axes_locatable(ax8)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(m8, cax=cax, orientation='vertical')

            ax9 = fig.add_subplot(2,5,9)
            m9 = ax9.matshow(np.reshape(np.round(T_P_P4,1),(dimensions[2],dimensions[2])),interpolation = "nearest")
            ax9.set_title('T_process_plate4')
            divider = make_axes_locatable(ax9)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(m9, cax=cax, orientation='vertical')
            
            ax10 = fig.add_subplot(2,5,10)
            m10 = ax10.matshow(np.reshape(np.round(T_U_P5,1),(dimensions[2],dimensions[2])),interpolation = "nearest")
            ax10.set_title('T_utility_plate5')
            divider = make_axes_locatable(ax10)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(m10, cax=cax, orientation='vertical')
            
            fileNameTemplate = r'HE plots\Plot{0:02d}.png'
            
            fig.savefig(fileNameTemplate.format(HE_plotcount), format='png')           
            
            fig.clf()
            plt.close()
            # plt.show()
            
#==============================================================================

        t = t + dt

        #This part converts gas phase enthalpy to gas phase temperature
        output = np.concatenate((np.divide(H_,np.multiply(cp_g_,rho_g_)),RXN_y[n_reactors:], p_[-1]), axis=None)
        solArr = np.insert(output, 0, t)
        
        #Write reactor info (T, T_s, P, y) into Excel sheet
        wtr.writerow(solArr)
        
        #Change input to the output of the last step
        y0 = RXN_y

        print("time: " + str(t))
        
        #Counter used to decide when to output a graph
        counter+=1
        if counter % 50 == 0:
            
            plotcount+=1
            
            H_ = y0[:n_reactors] #Gas phase rho_g * H_g
            T_s = y0[n_reactors:n_reactors*2] #Solid phase temperature
        
            # c_ = np.reshape(y0[n_reactors*2:], (-1, surf.n_total_species)) #Gas phase mass fraction and solid phase coverage fraction
            c_ = np.reshape(y0[n_reactors*2:], (-1, (bulk_gas.n_total_species+surf.n_total_species))) #Gas phase mass fraction, catalyst pore phase mass fraction, and solid phase coverage fraction
    
            c_g = c_.T[:][0:bulk_gas.n_total_species] #Gas phase mass fraction
            # c_pore = c_.T[:][bulk_gas.n_total_species:(bulk_gas.n_total_species+surf_gas.n_total_species)] #Catalyst pore phase mass fraction

            c_g = c_.T[:][0:surf_gas.n_total_species] #Gas phase mass fraction
            
            c_o2 = np.hstack((0.007523,c_g[:][0]))
            c_ch4 = np.hstack((0.001881,c_g[:][2]))
            # p_o2 = np.hstack((0.048746,c_pore[:][2]))
            # p_ch4 = np.hstack((0.0977562,c_pore[:][10]))
            # c_co2 = np.hstack((0.966242,c_g[:][15]))
            # c_h2 = np.hstack((0,c_g[:][0]))
            
            x_arr = np.arange(0,r_length+dx,dx)
            Tg_arr = np.hstack((T1,np.divide(H_,np.multiply(cp_g_,rho_g_))))
            Ts_arr = np.hstack((T_s[0],T_s))
            
            fig,ax = plt.subplots()
            ax.set_title('Axial temperature at t = ' + str(t))
            ax.set_xlabel('Axial distance [m]')
            ax.set_ylabel('Temperature [K]')
            ax.plot(x_arr, Tg_arr, c='b', label='gas phase T')
            ax.plot(x_arr, Ts_arr, c='r', label='solid phase T')
            ax.legend(loc='upper left')
            ax.set_xlim([0,r_length])
            ax.set_ylim([800,1200])
            ax2 = ax.twinx()
            ax2.set_ylabel('Mass fraction [-]')
            ax2.plot(x_arr, c_o2, c='y', label='O2 mass fraction')
            ax2.plot(x_arr, c_ch4, c='c', label='CH4 mass fraction')
            # ax2.plot(x_arr, c_co2, c='m', label='CO2 mass fraction')
            
            # ax2.plot(x_arr, c_co2, c='g', label='CO2 mass fraction')
            # ax2.plot(x_arr, c_co, c='m', label='CO mass fraction')
            # ax2.plot(x_arr, c_h2, c='k', label='H2 mass fraction')
            ax2.legend(loc='upper right')
            ax2.set_ylim([0,0.02])
            
            fileNameTemplate = r'new plots\Plot{0:02d}.png'
            
            fig.savefig(fileNameTemplate.format(plotcount), format='png')
            
            fig.clf()
            plt.close()
        

f.close()
f1.close()

# for i in range(10):
#     sol = euler_(consv_eqs,10e-5,y0,dt)
#     y0 = sol

# print(sol)

