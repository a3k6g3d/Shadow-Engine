- running `p0-spec-267551`: 3 arms x 3 runs
### pass 0, --spec: base 49.29 tok/s (floor 1.5%, needs +4.5%)
- `--spec 4`: 49.68 tok/s (+0.8%), paired median None
- `--spec 5`: 37.16 tok/s (-24.6%), paired median None
- running `p0-dflash-min-p-267551`: 4 arms x 3 runs
### pass 0, --dflash-min-p: base 49.59 tok/s (floor 0.2%, needs +1.0%)
- `--dflash-min-p 0.4`: 41.52 tok/s (-16.3%), paired median None
- `--dflash-min-p 0.6`: 50.68 tok/s (+2.2%), paired median None
- `--dflash-min-p 0.7`: 51.01 tok/s (+2.9%), paired median None
**adopted --dflash-min-p = 0.7** -> 51.01 tok/s; best flags now: `--ctx 8192 --ram-tier-gib 30 --placement J:/Shadow/profiles/routing-varied300.shrc --tail-skip 3 --verify-rank 2 --dflash J:/Shadow/models/dflash2 --spec 3 --dflash-min-p 0.7`
- running `p0-admit-after-887314`: 4 arms x 3 runs
### pass 0, --admit-after: base 51.02 tok/s (floor 0.2%, needs +1.0%)
- `--admit-after None`: 50.99 tok/s (-0.1%), paired median None
- `--admit-after 1`: 50.66 tok/s (-0.7%), paired median None
- `--admit-after 2`: 51.37 tok/s (+0.7%), paired median None
- running `p0-pin-margin-887314`: 4 arms x 3 runs
### pass 0, --pin-margin: base 50.95 tok/s (floor 0.2%, needs +1.0%)
- `--pin-margin None`: 50.94 tok/s (-0.0%), paired median None
- `--pin-margin 2`: 50.52 tok/s (-0.8%), paired median None
- `--pin-margin 4`: 50.57 tok/s (-0.7%), paired median None
best flags: `--ctx 8192 --ram-tier-gib 30 --placement J:/Shadow/profiles/routing-varied300.shrc --tail-skip 3 --verify-rank 2 --dflash J:/Shadow/models/dflash2 --spec 3 --dflash-min-p 0.7`
