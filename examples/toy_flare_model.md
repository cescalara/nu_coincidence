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
Emin = 3e4 # GeV
Emax = 1e8 # GeV
Emin_det = 4e5 # GeV
Enorm = 1e5 # GeV
spectral_index = 2.0
```

```python
# Best fit
ldde = BLLacLDDEModel()
ldde.A = 3.39 * 1e4
ldde.gamma1 = 0.27 
ldde.Lstar = 0.28 * 1e48
ldde.gamma2 = 1.86 
ldde.zcstar = 1.34 
ldde.p1star = 2.24 
ldde.tau = 4.92 
ldde.p2 = -7.37 
ldde.alpha = 4.53 * 1e-2
ldde.mustar = 2.1 
ldde.beta = 6.46 * 1e-2
ldde.sigma = 0.26 
```

```python
# For popsynth
ldde.Lmax = 1e50
ldde.prep_pop()
popsynth = ldde.popsynth()

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

popsynth.add_observed_quantity(flare_durations)
popsynth.add_observed_quantity(flare_amplitudes)
```

```python
pop = popsynth.draw_survey(boundary=4e-12, hard_cut=True)
```

```python
# Loop over sources
N = len(pop.distances)
Nnu_ex = np.zeros(N)
Nnu = np.zeros(N)
Erecos = []
ras = []
decs = []
ang_errs = []
detected = []
for i in range(N):

    ra = np.deg2rad(pop.ra[i])
    dec = np.deg2rad(pop.dec[i])
    z = pop.distances[i]
    
    # Loop over flares
    if pop.variability[i] and pop.flare_times[i].size > 0:
        
        for time, duration, amp in zip(pop.flare_times[i], pop.flare_durations[i],
                                          pop.flare_amplitudes[i]):
            
            # Define point source 
            L = pop.luminosities_latent[i] * amp # erg s^-1
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
            Nnu_ex[i] += nu_calc(time=duration, min_energy=Emin, max_energy=Emax)[0]
            
    Nnu[i] = np.random.poisson(Nnu_ex[i]) 
    
    # Simulate neutrino observations
    if Nnu[i] > 0:
        
        tmp = PowerLawFlux(1, Enorm, spectral_index, 
                           lower_energy=Emin, upper_energy=Emax)
        source = PointSource(flux_model=tmp, coord=(ra, dec))
        sim = Simulator(source, detector)
        sim.run(N=Nnu[i], show_progress=False, seed=np.random.randint(0, 100000))
        Erecos.extend(sim.reco_energy)
        ras.extend(sim.ra)
        decs.extend(sim.dec)
        ang_errs.extend(sim.ang_err)
        detected.extend(np.repeat(pop.selection[i], Nnu[i]))    
```

```python
fig, ax = plt.subplots(subplot_kw={"projection": "astro degrees mollweide"})
fig.set_size_inches((12, 7))
for r, d, e in zip(np.rad2deg(ras), np.rad2deg(decs), ang_errs):
    circle = SphericalCircle((r * u.deg, d * u.deg), e * 2 * u.deg,
                             transform=ax.get_transform("icrs"), alpha=0.7)
    ax.add_patch(circle)
```

```python
fig, ax = plt.subplots(subplot_kw={"projection": "astro degrees mollweide"})
fig.set_size_inches((12, 7))
for r, d, e, det in zip(np.rad2deg(ras), np.rad2deg(decs), ang_errs, detected):
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
```

```python

```
