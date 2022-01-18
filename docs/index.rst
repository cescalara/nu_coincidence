.. nu_coincidence documentation master file, created by
   sphinx-quickstart on Tue Jan 18 17:36:42 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to nu_coincidence's documentation!
==========================================

Investigating coincident source-neutrino detections through simulations. The examples here show how to simulate realistic source populations and neutrino emission. The source populations can either be disconnected from the neutrino emission to study chance coincidences, or alternatively neutrino emission can be connected to sources to study the implications.

This code makes use of the `popsynth <https://github.com/grburgess/popsynth.git>` and `icecube_tools <https://github.com/cescalara/icecube_tools.git>` packages, which also have their own documentation on how to simulate source populations and neutrinos. 

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   notebooks/coincidence_sim.ipynb
   notebooks/connected_sim.ipynb
   api/api
