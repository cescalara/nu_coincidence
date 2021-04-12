---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.0
  kernelspec:
    display_name: cosmic_coincidence
    language: python
    name: cosmic_coincidence
---

```python
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import ligo.skymap.plot
from astropy import units as u

from popsynth.utils.cosmology import cosmology
from icecube_tools.detector.effective_area import EffectiveArea
from icecube_tools.detector.energy_resolution import EnergyResolution
from icecube_tools.detector.angular_resolution import AngularResolution
from icecube_tools.detector.detector import IceCube
from icecube_tools.source.flux_model import PowerLawFlux
from icecube_tools.source.source_model import PointSource
from icecube_tools.neutrino_calculator import NeutrinoCalculator
from icecube_tools.simulator import Simulator
```

```python
import sys
sys.path.append("../")
```

```python
from cosmic_coincidence.blazars.bllac import BLLacLDDEModel
from cosmic_coincidence.blazars.fsrq import FSRQLDDEModel
from cosmic_coincidence.populations.aux_samplers import (VariabilityAuxSampler, 
                                                         FlareRateAuxSampler, 
                                                         FlareTimeAuxSampler,
                                                         FlareDurationAuxSampler,
                                                         FlareAmplitudeAuxSampler,)
from cosmic_coincidence.utils.plotting import SphericalCircle
```

```python
# Neutrino stuff
effective_area = EffectiveArea.from_dataset("20181018")
energy_res = EnergyResolution.from_dataset("20150820")
ang_res = AngularResolution.from_dataset("20181018", ret_ang_err_p=0.9, offset=0.4)
detector = IceCube(effective_area, energy_res, ang_res)
Emin = 5e4 # GeV
Emax = 1e8 # GeV
Emin_det = 4e5 # GeV
Enorm = 1e5 # GeV
spectral_index = 2.0
flux_factor = 0.01
```

## BL Lacs

```python
# Best fit
bllac_ldde = BLLacLDDEModel()
bllac_ldde.A = 3.39 * 1e4
bllac_ldde.gamma1 = 0.27 
bllac_ldde.Lstar = 0.28 * 1e48
bllac_ldde.gamma2 = 1.86 
bllac_ldde.zcstar = 1.34 
bllac_ldde.p1star = 2.24 
bllac_ldde.tau = 4.92 
bllac_ldde.p2 = -7.37 
bllac_ldde.alpha = 4.53 * 1e-2
bllac_ldde.mustar = 2.1 
bllac_ldde.beta = 6.46 * 1e-2
bllac_ldde.sigma = 0.26 
```

```python
# For popsynth
bllac_ldde.Lmax = 1e50
bllac_ldde.prep_pop()
bllac_popsynth = bllac_ldde.popsynth()

variability = VariabilityAuxSampler()
variability.weight = 0.05

flare_rate = FlareRateAuxSampler()
flare_rate.xmin = 1/7.5
flare_rate.xmax = 15
flare_rate.index = 1.5

flare_times = FlareTimeAuxSampler()
flare_times.obs_time = 7.5 # years

flare_durations = FlareDurationAuxSampler()

flare_amplitudes = FlareAmplitudeAuxSampler()
flare_amplitudes.index = 5
flare_amplitudes.xmin=1.2

flare_rate.set_secondary_sampler(variability)
flare_times.set_secondary_sampler(flare_rate)
flare_durations.set_secondary_sampler(flare_times)
flare_amplitudes.set_secondary_sampler(flare_times)

bllac_popsynth.add_observed_quantity(flare_durations)
bllac_popsynth.add_observed_quantity(flare_amplitudes)
```

```python
bllac_pop = bllac_popsynth.draw_survey(boundary=4e-12, hard_cut=True)
```

```python
# Loop over bllacs
N = len(bllac_pop.distances)
bllac_Nnu_ex = np.zeros(N)
bllac_Nnu = np.zeros(N)
bllac_nu_Erecos = []
bllac_nu_ras = []
bllac_nu_decs = []
bllac_nu_ang_errs = []
bllac_detected = []
for i in range(N):

    ra = np.deg2rad(bllac_pop.ra[i])
    dec = np.deg2rad(bllac_pop.dec[i])
    z = bllac_pop.distances[i]
    
    # Loop over flares
    if bllac_pop.variability[i] and bllac_pop.flare_times[i].size > 0:
        
        for time, duration, amp in zip(bllac_pop.flare_times[i],
                                       bllac_pop.flare_durations[i],
                                       bllac_pop.flare_amplitudes[i]):
            
            # Define point source 
            L = bllac_pop.luminosities_latent[i] * amp * flux_factor # erg s^-1
            L = L * 624 # GeV s^-1
            F = L / (4*np.pi * cosmology.luminosity_distance(z)**2) 
            tmp = PowerLawFlux(1, Enorm, spectral_index, 
                               lower_energy=Emin, upper_energy=Emax)
            P = tmp.total_flux_density()
            norm = F / P
            power_law = PowerLawFlux(norm, Enorm, spectral_index, 
                                     lower_energy=Emin, upper_energy=Emax)
            source = PointSource(flux_model=power_law, coord=(ra, dec))
            
            # Calulate expected neutrino number per source
            nu_calc = NeutrinoCalculator([source], effective_area)
            bllac_Nnu_ex[i] += nu_calc(time=duration, 
                                       min_energy=Emin, max_energy=Emax)[0]
            
    bllac_Nnu[i] = np.random.poisson(bllac_Nnu_ex[i]) 
    
    # Simulate neutrino observations
    if bllac_Nnu[i] > 0:
        
        tmp = PowerLawFlux(1, Enorm, spectral_index, 
                           lower_energy=Emin, upper_energy=Emax)
        source = PointSource(flux_model=tmp, coord=(ra, dec))
        sim = Simulator(source, detector)
        sim.run(N=bllac_Nnu[i], show_progress=False, seed=np.random.randint(0, 100000))
        bllac_nu_Erecos.extend(sim.reco_energy)
        bllac_nu_ras.extend(sim.ra)
        bllac_nu_decs.extend(sim.dec)
        bllac_nu_ang_errs.extend(sim.ang_err)
        bllac_detected.extend(np.repeat(bllac_pop.selection[i], bllac_Nnu[i]))    
```

```python
fig, ax = plt.subplots(subplot_kw={"projection": "astro degrees mollweide"})
fig.set_size_inches((12, 7))
for r, d, e in zip(np.rad2deg(bllac_nu_ras), np.rad2deg(bllac_nu_decs), 
                   bllac_nu_ang_errs):
    circle = SphericalCircle((r * u.deg, d * u.deg), e * 2 * u.deg,
                             transform=ax.get_transform("icrs"), alpha=0.7)
    ax.add_patch(circle)
```

## FSRQs

```python
fsrq_ldde = FSRQLDDEModel()
fsrq_ldde.A = 3.06e4
fsrq_ldde.gamma1 = 0.21
fsrq_ldde.Lstar = 0.84e48
fsrq_ldde.gamma2 = 1.58
fsrq_ldde.zcstar = 1.47
fsrq_ldde.p1star = 7.35
fsrq_ldde.tau = 0
fsrq_ldde.p2 = -6.51
fsrq_ldde.alpha = 0.21
fsrq_ldde.mustar = 2.44
fsrq_ldde.beta = 0
fsrq_ldde.sigma = 0.18

fsrq_ldde.Lmax = 1e50
fsrq_ldde.prep_pop()
fsrq_popsynth = fsrq_ldde.popsynth()

variability = VariabilityAuxSampler()
variability.weight = 0.4

flare_rate = FlareRateAuxSampler()
flare_rate.xmin = 1/7.5
flare_rate.xmax = 15
flare_rate.index = 1.5

flare_times = FlareTimeAuxSampler()
flare_times.obs_time = 7.5 # years

flare_durations = FlareDurationAuxSampler()

flare_amplitudes = FlareAmplitudeAuxSampler()
flare_amplitudes.index = 3
flare_amplitudes.xmin = 1.2

flare_rate.set_secondary_sampler(variability)
flare_times.set_secondary_sampler(flare_rate)
flare_durations.set_secondary_sampler(flare_times)
flare_amplitudes.set_secondary_sampler(flare_times)

fsrq_popsynth.add_observed_quantity(flare_durations)
fsrq_popsynth.add_observed_quantity(flare_amplitudes)
```

```python
fsrq_pop = fsrq_popsynth.draw_survey(boundary=4e-12, hard_cut=True)
```

```python
# Loop over fsrq
N = len(fsrq_pop.distances)
fsrq_Nnu_ex = np.zeros(N)
fsrq_Nnu = np.zeros(N)
fsrq_nu_Erecos = []
fsrq_nu_ras = []
fsrq_nu_decs = []
fsrq_nu_ang_errs = []
fsrq_detected = []
for i in range(N):

    ra = np.deg2rad(fsrq_pop.ra[i])
    dec = np.deg2rad(fsrq_pop.dec[i])
    z = fsrq_pop.distances[i]
    
    # Loop over flares
    if fsrq_pop.variability[i] and fsrq_pop.flare_times[i].size > 0:
        
        for time, duration, amp in zip(fsrq_pop.flare_times[i],
                                       fsrq_pop.flare_durations[i],
                                       fsrq_pop.flare_amplitudes[i]):
            
            # Define point source 
            L = fsrq_pop.luminosities_latent[i] * amp * flux_factor # erg s^-1
            L = L * 624 # GeV s^-1
            F = L / (4*np.pi * cosmology.luminosity_distance(z)**2) 
            tmp = PowerLawFlux(1, Enorm, spectral_index, 
                               lower_energy=Emin, upper_energy=Emax)
            P = tmp.total_flux_density()
            norm = F / P
            power_law = PowerLawFlux(norm, Enorm, spectral_index, 
                                     lower_energy=Emin, upper_energy=Emax)
            source = PointSource(flux_model=power_law, coord=(ra, dec))
            
            # Calulate expected neutrino number per source
            nu_calc = NeutrinoCalculator([source], effective_area)
            fsrq_Nnu_ex[i] += nu_calc(time=duration, 
                                       min_energy=Emin, max_energy=Emax)[0]
            
    fsrq_Nnu[i] = np.random.poisson(fsrq_Nnu_ex[i]) 
    
    # Simulate neutrino observations
    if fsrq_Nnu[i] > 0:
        
        tmp = PowerLawFlux(1, Enorm, spectral_index, 
                           lower_energy=Emin, upper_energy=Emax)
        source = PointSource(flux_model=tmp, coord=(ra, dec))
        sim = Simulator(source, detector)
        sim.run(N=fsrq_Nnu[i], show_progress=False, seed=np.random.randint(0, 100000))
        fsrq_nu_Erecos.extend(sim.reco_energy)
        fsrq_nu_ras.extend(sim.ra)
        fsrq_nu_decs.extend(sim.dec)
        fsrq_nu_ang_errs.extend(sim.ang_err)
        fsrq_detected.extend(np.repeat(fsrq_pop.selection[i], fsrq_Nnu[i]))    
```

```python
fig, ax = plt.subplots(subplot_kw={"projection": "astro degrees mollweide"})
fig.set_size_inches((12, 7))
for r, d, e in zip(np.rad2deg(fsrq_nu_ras), np.rad2deg(fsrq_nu_decs), 
                   fsrq_nu_ang_errs):
    circle = SphericalCircle((r * u.deg, d * u.deg), e * 2 * u.deg,
                             transform=ax.get_transform("icrs"), alpha=0.7)
    ax.add_patch(circle)
```

```python
fig, ax = plt.subplots(subplot_kw={"projection": "astro degrees mollweide"})
fig.set_size_inches((12, 7))

for r, d, e, det in zip(np.rad2deg(bllac_nu_ras), 
                        np.rad2deg(bllac_nu_decs), 
                        bllac_nu_ang_errs, bllac_detected):
    if det:
        color = "red"
        alpha = 0.3
    else:
        color="blue"
        alpha = 0.3
    circle = SphericalCircle((r * u.deg, d * u.deg), e *  u.deg,
                             transform=ax.get_transform("icrs"), 
                             alpha=alpha, color=color)
    ax.add_patch(circle)
    
for r, d, e, det in zip(np.rad2deg(fsrq_nu_ras), 
                        np.rad2deg(fsrq_nu_decs), 
                        fsrq_nu_ang_errs, fsrq_detected):
    if det:
        color = "red"
        alpha = 0.3
    else:
        color="blue"
        alpha = 0.3
    circle = SphericalCircle((r * u.deg, d * u.deg), e *  u.deg,
                             transform=ax.get_transform("icrs"), 
                             alpha=alpha, color=color)
    ax.add_patch(circle)
    
legend_patches = [SphericalCircle((0*u.deg, 0*u.deg), 1*u.deg, color="red"), 
                  SphericalCircle((0*u.deg ,0*u.deg), 1*u.deg, color="blue")]
legend_labels = ["Parent blazar is detected", "Parent blazar is undetected"]
ax.legend(legend_patches, legend_labels)
fig.suptitle("F_nu = %.2f F_gamma" % flux_factor)
fig.savefig("figures/combined_flux_factor_%.2f.pdf" % flux_factor, bbox_inches="tight", dpi=500)
```

```python

```
