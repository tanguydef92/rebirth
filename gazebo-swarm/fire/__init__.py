"""Thermal fire model and its construction entry point."""
def create_fire(config):
    if config.get('model') != 'lagrangian':
        raise ValueError('fire.model must be lagrangian')
    from fire.particles import ParticleFire
    return ParticleFire(config)
