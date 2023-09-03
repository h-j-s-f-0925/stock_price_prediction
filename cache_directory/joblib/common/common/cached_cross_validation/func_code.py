# first line: 65
@memory.cache
def cached_cross_validation(model, initial, period, horizon):
    return cross_validation(model, initial=initial, period=period, horizon=horizon)
