import numpy as np

from model import UCILogisticRegression, MoonClassification


"""
construct the model
and the data split: removed data, remaining data
"""

def replace_elem_in_dict(ref_dict, new_dict):
    """
    replace value in ref_dict with new_dict
    return the ref_dict with modified values
    """
    for k in ref_dict:
        if k in new_dict:
            ref_dict[k] = new_dict[k]
    
    return ref_dict


def get_experiment(experiment, config={}):
    """
    if config != None: over-write the parameter in config
        ndata, removed_idxs, seed
    """
    if experiment == "gamma_unknown_shape":
    
        nparam = 1
        dim = 1

        prior_params = {
            "weight": np.array([1.0,0.0]),
            "dist": [
                { 
                    "loc": np.ones(nparam) *  0.0,
                    "cov": np.eye(nparam)
                },
                { 
                    "loc": np.ones(nparam) * 2.0,
                    "cov": np.eye(nparam) * 4.0
                }
            ]
        }

        model = GammaUnknownShape(dim, nparam, 
                    params = {"log_shape": np.ones(dim) * 1.0, 
                            "log_rate": np.ones(dim) * 1.0}, 
                    prior_params = prior_params)

        default_config = {
            'ndata': 20,
            'removed_idxs': np.array(list(range(0,5))),
            'seed': 0,
            'sort_axis': 0,
            'xmin': None,
            'xmax': None
        }

        config = replace_elem_in_dict(default_config, config)
        config['removed_idxs'] = np.array(config['removed_idxs'])
        remain_idxs = np.array( list(set(range(config['ndata'])).difference(set(config['removed_idxs']))) )

        ndata = config['ndata']
        data = model.sample_data_np(size=ndata, xbound=[config['xmin'], config['xmax']], seed=config['seed'])
        if config['sort_axis'] >= 0:
            data = data[ data[:,config['sort_axis']].argsort() ]

        removed_data = data[config['removed_idxs'],...]
        remain_data = data[remain_idxs,...]

    elif experiment == "moon":

        model = MoonClassification(nu=20, noise_std=0.2,
                log_lengthscales=np.log(np.array([2.4232, 1.8230])),
                log_sigma=np.log(4.7432))
        
        nparam = model.nparam
        dim = model.dim

        default_config = {
            'ndata': 100,
            'removed_idxs': np.array(list(range(int(100*0.8),100))),
            'seed': 0,
            'sort_axis': 0,
            'xmin': 1.,
            'xmax': 2.
        }

        config = replace_elem_in_dict(default_config, config)
        config['removed_idxs'] = np.array(config['removed_idxs'])
        remain_idxs = np.array( list(set(range(config['ndata'])).difference(set(config['removed_idxs']))) )

        ndata = config['ndata']
        data = model.sample_data_np(size=ndata, xbound=[config['xmin'], config['xmax']], seed=config['seed'])
        if config['sort_axis'] >= 0:
            data = data[ data[:,config['sort_axis']].argsort() ]

        removed_data = data[config['removed_idxs'],...]
        remain_data = data[remain_idxs,...]

    elif experiment.startswith("moon_random"):

        model = MoonClassification(nu=20, noise_std=0.2,
                log_lengthscales=np.log(np.array([2.4232, 1.8230])),
                log_sigma=np.log(4.7432))
        
        nparam = model.nparam
        dim = model.dim

        default_config = {
            'ndata': 100,
            'removed_idxs': np.array(list(range(int(100*0.8),100))),
            'seed': 0,
            'sort_axis': -1,
            'xmin': 1.,
            'xmax': 2.
        }

        config = replace_elem_in_dict(default_config, config)
        config['removed_idxs'] = np.array(config['removed_idxs'])
        remain_idxs = np.array( list(set(range(config['ndata'])).difference(set(config['removed_idxs']))) )

        ndata = config['ndata']
        data = model.sample_data_np(size=ndata, xbound=[config['xmin'], config['xmax']], seed=config['seed'])
        if config['sort_axis'] >= 0:
            data = data[ data[:,config['sort_axis']].argsort() ]

        removed_data = data[config['removed_idxs'],...]
        remain_data = data[remain_idxs,...]

    elif experiment.startswith("moon_rm_"):

        n_rm = int(experiment[8:])
        print("Remove {} points".format(n_rm))

        model = MoonClassification(nu=20, noise_std=0.2,
                log_lengthscales=np.log(np.array([2.4232, 1.8230])),
                log_sigma=np.log(4.7432))
        
        nparam = model.nparam
        dim = model.dim

        default_config = {
            'ndata': 100,
            'removed_idxs': None,
            'seed': 0,
            'sort_axis': 0,
            'xmin': 1.,
            'xmax': 2.
        }

        config = replace_elem_in_dict(default_config, config)

        ndata = config['ndata']
        data = model.sample_data_np(size=ndata, xbound=[config['xmin'], config['xmax']], seed=config['seed'])
        if config['sort_axis'] >= 0:
            data = data[ data[:,config['sort_axis']].argsort() ]

        removed_idxs = np.where(data[:,2] == 1)[0][-n_rm:]
        remain_idxs = np.array( list(set(range(config['ndata'])).difference(set(removed_idxs))) )

        removed_data = data[removed_idxs,...]
        remain_data = data[remain_idxs,...]


    elif experiment == "banknote_authentication1":

        model = UCILogisticRegression(dataset="banknote", 
                    order=1, 
                    prior_gaussian_mean = 0, 
                    prior_gaussian_std = 100)

        nparam = model.nparam
        dim = model.dim
        
        default_config = {
            'ndata': 960+412,
            'removed_idxs': np.array(list(range(960,960+412))),
            'seed': 0,
            'sort_axis': -1,
            'xmin': 1.,
            'xmax': 2.
        }

        config = replace_elem_in_dict(default_config, config)
        config['removed_idxs'] = np.array(config['removed_idxs'])
        remain_idxs = np.array( list(set(range(config['ndata'])).difference(set(config['removed_idxs']))) )

        ndata = config['ndata']
        data = model.sample_data_np(size=ndata, xbound=[config['xmin'], config['xmax']], seed=config['seed'])
        if config['sort_axis'] >= 0:
            data = data[ data[:,config['sort_axis']].argsort() ]

        removed_data = data[config['removed_idxs'],...]
        remain_data = data[remain_idxs,...]

    else:
        raise Exception("Unknown experiment: {}".format(experiment))

    return {'dim': dim, 'nparam': nparam, 'model': model, 'data': data, 
            'removed_data': removed_data, 'remain_data': remain_data, 'model_config': config}