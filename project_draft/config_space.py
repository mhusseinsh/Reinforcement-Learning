import numpy as np
import ConfigSpace as CS

def get_config_space():
	config_space = CS.ConfigurationSpace()
	config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('action_indices', lower=1000, upper=100000, default_value=20000))
	config_space.add_hyperparameter(CS.UniformFloatHyperparameter('learning_rate', lower=1e-8, upper=1e-3, default_value=1e-5))
	config_space.add_hyperparameter(CS.UniformFloatHyperparameter('discount_factor', lower=1e-4, upper=1, default_value=0.8))
	config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('num_episodes', lower=50, upper=100, default_value=75))
	config_space.add_hyperparameter(CS.CategoricalHyperparameter('batch_size',choices=[32, 64, 128], default_value=64))

	config_space.add_hyperparameter(CS.CategoricalHyperparameter('num_fc_units_1',choices=[8,16,24,32,48], default_value=24))
	config_space.add_hyperparameter(CS.CategoricalHyperparameter('num_fc_units_2',choices=[8,16,24,32,48], default_value=24))
	config_space.add_hyperparameter(CS.CategoricalHyperparameter('num_fc_units_3',choices=[8,16,24,32,48], default_value=24))
	config_space.add_hyperparameter(CS.CategoricalHyperparameter('num_fc_units_4',choices=[8,16,24,32,48], default_value=24))

	return(config_space)



