# BLG_model_builder
Tools to build bilayer graphene total energy and tight binding models

modules include:
  - Fitting interatomic potentials for bilayer graphene
  - Fitting Tight Binding models for bilayer graphene
  - running Markov Chain Monte Carlo uncertainty quantification for said models
  - running uncertainty propagation for relaxations and band structures

Interatomic Potentials included:
  - Intralayer Tersoff potential
  - Intralayer REBO potential
  - Interlayer Kolmogorov-Crespi potential
  - Custom Interlayer Neural Network Potential

Tight Binding models included:
  - Moon-Koshino model
  - Local-environment Tight Binding model
  - Popov Van Alsenoy model
  - custom Neural Network model

Total Energy Tight Binding model:
  - any combination of inter/intralayer potentials with Popov Van Alsenoy/ Moon-Koshino / Neural Network model

GPU capabilities for Tight Binding via Cupy
