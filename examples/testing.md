---
jupyter:
  jupytext:
    formats: ipynb,md
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

# Testing

```python
import popsynth
import networkx as nx
```

```python
homo_pareto_synth = popsynth.populations.ParetoHomogeneousSphericalPopulation(
    Lambda=0.25, 
    Lmin=1, 
    alpha=2.0  
) 
homo_pareto_synth.display()
```

```python
# we can also display a graph of the object


options = {"node_color": "g", "node_size": 2000, "width": 0.5}

# pos = nx.spring_layout(g,k=5, iterations=300)
```

```python
pos = nx.drawing.nx_agraph.graphviz_layout(homo_pareto_synth.graph, prog="dot")

nx.draw(homo_pareto_synth.graph, with_labels=True, pos=pos, **options)
```

```python
population = homo_pareto_synth.draw_survey(boundary=1e-2, hard_cut=True, 
                                           flux_sigma=0.1)
```

```python
population.display_fluxes(obs_color="g", true_color="r")
```

```python
#population.display_obs_fluxes_sphere();
```

```python
population.writeto("output/saved_pop.h5")
```

```python

```
