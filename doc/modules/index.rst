=======
Modules
=======
What a module is?
-----------------

A deformation is a structure that can generate vector fields of :math:`\mathbb R^d` (with :math:`d = 1, 2, 3` fixed). Theoretically, it is defined by 2 spaces and 3 functions:

* Spaces: 
    * Space of **geometrical descriptors** :math:`\mathcal O`.
    * Space of **controls** :math:`H`.

* Functions:
    * **Field generator** :math:`\zeta : \mathcal O \times H \mapsto C^\ell (\mathbb R^d, \mathbb R^d)` (space of vector fields of :math:`\mathbb R^d`) 
        Defines how a couple of geometrical descriptor and control can generate a vector field of the ambient space.
        The geometrical descriptor determines the geometry of the generated field (e.g. location) and the control determines the intensity.

    * **Action infinitesimal** :math:`\xi : \mathcal O \times C^\ell (\mathbb R^d, \mathbb R^d) \mapsto \mathcal O`
        Defines how a vector field can give a speed to a geometrical descriptor.

    * **Cost** :math:`c : \mathcal O \times H \mapsto \mathbb R^d`
        Defines the cost associated to a couple of geometrical descriptor and control.



Implemented modules
-------------------



Silent Landmarks
^^^^^^^^^^^^^^^^

This module is **silent** in the sense that the generated fields are always null. 
Geometrical descriptors are sets of landmarks that can be moved by vector fields (via the infinitesimal action). 

Let :math:`P` be an fixed integer, formally, 
the silent deformation modules with :math:`P` landmarks is defined by:

*  :math:`\mathcal O = (\mathbb R^d)^P`

*  :math:`H = \emptyset`

*  :math:`\zeta: (q,h)  \in \mathcal O \times H \mapsto  0 \in C^\ell (\mathbb R^d, \mathbb R^d)`

*  :math:`\xi: (q,v)  \in \mathcal O \times C^\ell (\mathbb R^d, \mathbb R^d) \mapsto  (v(x_1), \dots, v(x_P)) \in T_q \mathcal O` where :math:`q = (x_1, \dots, x_P)`

*  :math:`c: (q,h)  \in \mathcal O \times H \mapsto  0 \in \mathbb R`




Sum of local translations
^^^^^^^^^^^^^^^^^^^^^^^^^

This module generates a sum of local translations, localized by a chosen kernel :math:`K`.

Let :math:`P` be an fixed integer, the deformation modules generating a sum of :math:`P` translations is defined by:


*  :math:`\mathcal O = (\mathbb R^d)^P`

*  :math:`H = (\mathbb R^d)^P`

*  :math:`\zeta: (q,h)  \in \mathcal O \times H \mapsto  \sum_{i=1}^P K(x_i, \cdot) h_i  \in C^\ell (\mathbb R^d, \mathbb R^d)` where :math:`q = (x_1, \dots, x_P)` and :math:`h = (h_1, \dots, h_P)`

*  :math:`\xi: (q,v)  \in \mathcal O \times C^\ell (\mathbb R^d, \mathbb R^d) \mapsto  (v(x_1), \dots, v(x_P)) \in T_q \mathcal O` where :math:`q = (x_1, \dots, x_P)`

*  :math:`c: (q,h)  \in \mathcal O \times H \mapsto  |\zeta_q (h)|^2 \in \mathbb R`


*Remark* : This module is a particular case of the following one, in the implementation it is incorporated in it.




Implicit modules of order 0
^^^^^^^^^^^^^^^^^^

This module is **implicit** in the sense that its field generator is defined implicitely. 


Modules of order 1
^^^^^^^^^^^^^^^^^^

:math:`\mathcal O`:


.. toctree::
   :maxdepth: 2
   :caption: Modules

