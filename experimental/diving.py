"""Handle diving domains in bab-attack.

Caution: This new version has not been tested. Refer to the stable version:
https://github.com/Verified-Intelligence/alpha-beta-CROWN/tree/1c29191bcdfb64a3fa54857eb57436e02043eb16
"""

from collections import defaultdict

import torch


def stash_diving(d, batch):
    d_diving = {
        'lower_bounds': [], 'upper_bounds': [],
    }
    for i in range(len(d['lower_bounds'])):
        d_diving['lower_bounds'].append(d['lower_bounds'][i][batch:])
        d_diving['upper_bounds'].append(d['upper_bounds'][i][batch:])
        d['lower_bounds'][i] = d['lower_bounds'][i][:batch]
        d['upper_bounds'][i] = d['upper_bounds'][i][:batch]

    if isinstance(d['slopes'], defaultdict):
        d_diving['slopes'] = defaultdict(dict)
        for k, v in d['slopes'].items():
            d_diving['slopes'][k] = {
                kk: vv[:, :, batch:] for kk, vv in v.items()
            }
            d['slopes'][k] = {
                kk: vv[:, :, :batch] for kk, vv in v.items()
            }
    else:
        d_diving['slopes'] = [{
            kk: vv[:, :, batch:] for kk, vv in v.items()
        } for v in d['slopes']]
        d['slopes'] = [{
            kk: vv[:, :, :batch] for kk, vv in v.items()
        } for v in d['slopes']]

    d_diving['betas'] = d['betas'][batch:]
    d['betas'] = d['betas'][:batch]

    return d_diving


def unstash_diving(d, d_diving):
    for i in range(len(d['lower_bounds'])):
        d['lower_bounds'][i] = torch.concat(
            [d['lower_bounds'][i], d_diving['lower_bounds'][i]], dim=0)
        d['upper_bounds'][i] = torch.concat(
            [d['upper_bounds'][i], d_diving['upper_bounds'][i]], dim=0)

    if isinstance(d['slopes'], defaultdict):
        d_diving['slopes'] = defaultdict(dict)
        for k, v in d['slopes'].items():
            d['slopes'][k] = {
                kk: torch.concat([vv[:, :], d_diving['slopes'][k][kk]], dim=2)
                for kk, vv in v.items()
            }
    else:
        d['slopes'] = [{
            kk: torch.concat([vv[:, :], d_diving['slopes'][k][kk]], dim=2)
            for kk, vv in v.items()
        } for v in d['slopes']]

    d['betas'] = d['betas'] + d_diving['betas']