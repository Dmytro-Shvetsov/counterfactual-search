from .counterfactual_cgan import CounterfactualCGAN
from .lungs_cgan import LungsCGAN


def build_gan(opt, **kwargs):
    assert 'kind' in opt, 'No architecture type specified in the model configuration'
    if opt.kind == 'lungs_cgan':
        return LungsCGAN(opt=opt, **kwargs)
    elif opt.kind == 'counterfactual_lungs_cgan':
        return CounterfactualCGAN(opt=opt, **kwargs)
    else:
        raise ValueError(f'Invalid architecture type: {opt.kind}')
