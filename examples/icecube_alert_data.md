---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.9.1
  kernelspec:
    display_name: cosmic_coincidence
    language: python
    name: cosmic_coincidence
---

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from astropy.time import Time 
```

```python
df = pd.read_csv("data/catalog_of_alerts.txt", delim_whitespace=True, comment="#", 
                names=["MJD", "RA", "RA_Error", "Dec", "Dec_Error"])
```

```python
times = Time(df.MJD.values, format="mjd")
```

```python
fig, ax = plt.subplots()
ax.hist(times.value - times.value[0], bins=4);
plt.vlines(times.value - times.value[0], 0, 2, color='k')
```

```python
print("Events/yr:", len(times) / ((times.value[-1] - times.value[0])/365))
```

```python
eval(df.RA_Error.values[1])
```

```python
df.
```

```python
ra_errs = []
dec_errs = []
Ereco = []
for ra_e, dec_e in zip(df.RA_Error.values, df.Dec_Error.values):
    if ra_e != "(-)" and dec_e != "(-)":
        a, b = eval(ra_e)
        ra_errs.append(a)
        ra_errs.append(b)
        a, b = eval(dec_e)
        dec_errs.append(a)
        dec_errs.append(b)
ra_errs = abs(np.array(ra_errs))
dec_errs = abs(np.array(dec_errs))
```

```python
fig, ax = plt.subplots()
bins=np.linspace(0, 5.3, 15)
ax.hist(ra_errs, alpha=0.7, label="RA errors", bins=bins)
ax.hist(dec_errs, alpha=0.7, label="Dec errors", bins=bins)
ax.axvline(np.mean(ra_errs), color='k')
ax.axvline(np.mean(dec_errs), color='k')
ax.legend();
```

```python

```
