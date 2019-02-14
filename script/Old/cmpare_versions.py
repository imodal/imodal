import old.DeformationModules.Combination as comb_mod_old
import old.Forward.shooting as shoot
from old.DeformationModules.ElasticOrder0 import ElasticOrderO
from old.DeformationModules.ElasticOrder1 import ElasticOrder1
from old.DeformationModules.SilentLandmark import SilentLandmark

# %%

# %%
Sil = SilentLandmark(xs.shape[0], dim)
Model0 = ElasticOrderO(sig0, x0.shape[0], dim, coeffs[0], nu)
Model00 = ElasticOrderO(sig00, x00.shape[0], dim, 0.1, nu)
Model1 = ElasticOrder1(sig1, x1.shape[0], dim, coeffs[1], C, nu)
Mod_el_init = comb_mod_old.CompoundModules([Sil, Model00, Model0, Model1])
# Mod_el_init = comb_mod.CompoundModules([Model00, Model0])

# %%
# param = [param_sil, param_00, param_0, param_1]
# param = [param_00, param_0, param_1]
# param = [param_sil, param_1]
GD = Mod_el_init.GD.copy()
Mod_el_init.GD.fill_cot_from_param(param)
Mod_el = Mod_el_init.copy_full()

N = 5

# %%
Modlist = shoot.shooting_traj(Mod_el, N)
