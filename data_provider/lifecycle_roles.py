"""The one definition of the lifecycle role order.

ALFNet indexes its per-role scale vectors positionally as ("aas", "nas", "vs",
"fs") (`model/ALFNet/ALFNet.py`), and the loader emits ids in that order. A
second loader that picked its own order -- unseen=0, steady=1, join=2 -- would
silently feed Unseen where the model reads AAS and Join where it reads VS, with
no shape error anywhere. That happened, which is why this module exists.

Import these names; do not re-declare the integers.
"""

ROLE_AAS = 0      # actively-attached sensor: observed, mature history
ROLE_NAS = 1      # newly-attached sensor: the join track's revealed steps
ROLE_VS = 2       # virtual sensor: no history this window (unseen or hidden)
ROLE_FS = 3       # failed sensor: the fault track's dark steps
ROLE_CHANNEL_COUNT = 4
ROLE_ORDER = ("aas", "nas", "vs", "fs")
