# IMODAL

IMODAL is a python library allowing to register shapes (curves, meshes, images) with **structured large deformations**. The structures are incorporated via **deformation modules** which generate vector fields of particular, chosen types. They can be defined *explicitly* (generating local scalings or rotations for instance) or *implicitly* from constraints. In addition, it is possible to combine them so that a complex structure can be easily defined as the superimposition of simple ones. Trajectories of such modular vector fields can then be integrated to build *modular large deformations*. Their parameters can be optimized to register observed shapes and analyzed.

[Link to the project documentation](https://kernel-operations.io/im/).


## IMODAL provides

* registration of points clouds, curves, meshes and images
* atlas computation with hypertemplate
* estimation of the model parameters
* tools to speed up and reduce the memory footprint (such as GPU and KeOps support)

## Authors

* Benjamin Charlier
* Barbara Gris
* Leander Lacroix
* Alain Trouvé

## Related publications

* [A sub-Riemannian modular framework for diffeomorphism based analysis of shape ensembles, B. Gris, S. Durrleman and A. Trouvé, SIAM Journal of Imaging Sciences, 2018.](https://hal.archives-ouvertes.fr/hal-01321142v2)
* [IMODAL: creating learnable user-defined deformation models, B. Charlier, L. Lacroix, B. Gris, A. Trouvé, CVPR, 2021.](https://hal.archives-ouvertes.fr/hal-03251752)

