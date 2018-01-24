import numpy as np
import ConfigSpace as CS

def get_config_space():
	config_space = CS.ConfigurationSpace()
	config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('action_indices', lower=5, upper=201, default_value=41))
	config_space.add_hyperparameter(CS.UniformFloatHyperparameter('learning_rate', lower=1e-8, upper=1e-3, default_value=1e-5))
	config_space.add_hyperparameter(CS.UniformFloatHyperparameter('discount_factor', lower=1e-4, upper=1, default_value=0.8))
	config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('max_time_per_episode', lower=200, upper=1000, default_value=300))

	config_space.add_hyperparameter(CS.CategoricalHyperparameter('num_fc_units_1',choices=[8,16,24,32,48], default_value=24))
	config_space.add_hyperparameter(CS.CategoricalHyperparameter('num_fc_units_2',choices=[8,16,24,32,48], default_value=24))
	config_space.add_hyperparameter(CS.CategoricalHyperparameter('num_fc_units_3',choices=[8,16,24,32,48], default_value=24))
	config_space.add_hyperparameter(CS.CategoricalHyperparameter('num_fc_units_4',choices=[8,16,24,32,48], default_value=24))

	return(config_space)



