$HYDRA_FULL_ERROR=1
python 4_sim_datasets.py base_distrib=normal config_name=normal n_comps=[1,2]
python 4_sim_datasets.py base_distrib=normal config_name=normal_mix n_comps=[2,6]
python 4_sim_datasets.py base_distrib=exponential config_name=expon n_comps=[1,2] loc=[0.0,0.9]
python 4_sim_datasets.py base_distrib=exponential config_name=expon_mix n_comps=[2,6] loc=[0.0,0.9]
 