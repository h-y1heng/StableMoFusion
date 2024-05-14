

from .unet import T2MUnet


__all__ = ['T2MUnet']

def build_models(opt):
    print('\nInitializing model ...' )
    model = T2MUnet(
        input_feats=opt.dim_pose, 
        text_latent_dim=opt.text_latent_dim,
        base_dim = opt.base_dim,
        dim_mults = opt.dim_mults,
        time_dim=opt.time_dim,
        adagn = not opt.no_adagn,
        zero = True,
        no_eff = opt.no_eff,
        cond_mask_prob = getattr(opt, 'cond_mask_prob', 0.)
        )

    return model

