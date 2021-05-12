---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: cosmic_coincidence
    language: python
    name: cosmic_coincidence
---

```python
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy import units as u
from scipy import stats
```

```python
hdul = fits.open('data/2fav_v10.fits')
hdul.info()
```

```python
#hdul[2].header
```

```python
# Get flare duration for different categories
N = len(hdul[1].data)

duration = {}
duration['bll'] = []
duration['fsrq'] = []
duration['bcu'] = []

n_flares = {}
n_flares['bll'] = []
n_flares['fsrq'] = []
n_flares['bcu'] = []

ph_count_excess = {}
ph_count_excess['bll'] = []
ph_count_excess['fsrq'] = []
ph_count_excess['bcu'] = []

# Loop over sources
for i in range(N):
    
    c = hdul[1].data['CLASS'][i]
    
    # Get flare times corresponding to this source
    selection = np.where(hdul[2].data['FAVASRC']==i+1)[0]   
    sig_le = hdul[2].data['LESIGNIF'][selection]
    sig_he = hdul[2].data['HESIGNIF'][selection]
    #condition = ((sig_le > -4) & (sig_he > -4)) |  ((sig_he > -6) | (sig_le > -6))
    condition = (sig_le > 0) | (sig_he > 0)
    #condition = sig_le > -100
    
    # Get non-negative flares
    non_neg_sel = condition #selection
    tstart = hdul[2].data['TSTART'][selection].astype(int)[condition]
    tstop = hdul[2].data['TSTOP'][selection].astype(int)[condition]     
    
    # Get Expected events vs observed
    lowE_expected = hdul[2].data['LEAVNEV'][selection]
    lowE_observed = hdul[2].data['LENEV'][selection]
    count_excess = lowE_observed[condition] / lowE_expected[condition]
    
    # Merge adjacent flare periods
    eq_ind = np.where(np.equal(tstart[1:], tstop[:-1]))[0]
    a = np.delete(tstart, eq_ind+1)
    b = np.delete(tstop, eq_ind)
    d = (b-a) / 604800 # duration in weeks
    
    if c == 'bll':
        duration['bll'].extend(d)
        ph_count_excess['bll'].extend(count_excess)
        n_flares['bll'].append(len(d))
        
    elif c == 'fsrq':
        duration['fsrq'].extend(d)
        n_flares['fsrq'].append(len(d))
        ph_count_excess['fsrq'].extend(count_excess)
        
    elif c == 'bcu':
        duration['bcu'].extend(d)
        ph_count_excess['bcu'].extend(count_excess)
        n_flares['bcu'].append(len(d))
```

```python
# Flare number
fig, ax = plt.subplots()
ax.hist(n_flares['bll'])
ax.hist(n_flares['bcu'])
ax.hist(n_flares['fsrq'])
```

```python
# Duration
fig, ax = plt.subplots()
bins=np.linspace(1, 110)
ax.hist(duration['fsrq'], label='fsrq', alpha=0.7, bins=bins, density=True)
ax.hist(duration['bll'], label='bll', alpha=0.7, bins=bins, density=True)
ax.hist(duration['bcu'], label='bcu', alpha=0.7, bins=bins, density=True);
ax.plot(bins, stats.pareto(1.5).pdf(bins), alpha=0.7, color='k', 
        label='pareto approx');
ax.plot(bins, stats.lognorm(2.5, 0, 1e-1).pdf(bins), color='k', linestyle=':')
ax.set_yscale('log')
ax.legend()
```

```python
print(sum(n_flares['bll']), sum(n_flares['fsrq']), sum(n_flares['bcu']))
```

```python
# Rates
time = (473615018 - 239557418) * u.s
time = time.to(u.year)

bins = 10**np.linspace(-1, 2)
fig, ax = plt.subplots()
for key, value in n_flares.items():
    print(key, min((value/time.value)))
    ax.hist((value/time.value), label=key, alpha=0.7, density=True, bins=bins)

ax.hist(stats.pareto(1.5).rvs(10000) * 0.1, alpha=0.5, color='k', bins=bins, 
        density=True)
#ax.plot(bins, stats.pareto(0.1).pdf(bins), color='k', label='pareto approx')
ax.plot(bins, stats.cauchy(0, 0.2).pdf(bins), color='r')
ax.plot(bins, stats.lognorm(2, 0, 1e-1).pdf(bins), color='k', linestyle=':')
ax.legend()
ax.set_yscale('log')
ax.set_xscale('log')
```

```python
# Photon count excess
fig, ax = plt.subplots()
bins = np.linspace(0, 15, 100)
ax.hist(ph_count_excess['bll'], alpha=0.7, density=True, bins=bins);
ax.hist(ph_count_excess['fsrq'], alpha=0.7, density=True, bins=bins);
ax.hist(ph_count_excess['bcu'], alpha=0.7, density=True, bins=bins);
#ax.set_yscale("log")
#ax.set_yscale("log")
#ax.hist(stats.skewnorm(a=10, loc=1, scale=2).rvs(100000), density=True, 
#        alpha=0.7, bins=bins);
#ax.hist(stats.pareto(3).rvs(10000) * 1.2, alpha=0.7, bins=bins, density=True)
ax.axvline(1, color="k")
ax.set_xlim(0, 5)
```

```python
sum(hdul[1].data['FLARES'])
```

```python

```
