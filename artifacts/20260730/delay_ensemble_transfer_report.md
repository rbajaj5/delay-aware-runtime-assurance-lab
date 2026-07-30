# Delayed Loop-Diagnostic Topology Transfer

Conceptual source: [Bourgade and Huang, *Loop Equations Characterize Random Matrix Statistics*](https://arxiv.org/abs/2607.07617). The paper's universality program asks when local spectral observables persist across different underlying ensembles; its applications include random regular graphs.

Here the same delayed-message dynamics are run on a ring, a deterministic two-block graph, and independently generated simple random 4-regular graphs. We compare the spectral-edge proxy, first loop residual, receipt age, and state disagreement. This is a finite transfer check only: no universal limit, local law, or theorem hypothesis is asserted.

| topology | delay | policy | games | spectral edge | loop residual | receipt age | disagreement |
|---|---:|---|---:|---:|---:|---:|---:|
| ring | 0 | blind_delay | 15 | 0 | 0.23529411764705888 | 0 | 0.38554013513023733 |
| ring | 0 | queue_aware | 15 | 0 | 0.23529411764705888 | 0 | 0.38554013513023733 |
| ring | 1 | blind_delay | 15 | 0 | 0.23529411764705888 | 0 | 0.38115325950794865 |
| ring | 1 | queue_aware | 15 | 0 | 0.23529411764705888 | 0 | 0.38115325950794865 |
| ring | 2 | blind_delay | 15 | 1.4503016459925282 | 0.050150259611627973 | 0.99166666666666703 | 0.28354653319707257 |
| ring | 2 | queue_aware | 15 | 0 | 0.23529411764705888 | 0 | 0.3749024300750719 |
| ring | 3 | blind_delay | 15 | 1.4503016459925282 | 0.050150259611627973 | 1.9750000000000005 | 0.27635142838879373 |
| ring | 3 | queue_aware | 15 | 0 | 0.23529411764705888 | 0 | 0.37917099372411062 |
| ring | 5 | blind_delay | 15 | 1.4503016459925282 | 0.050150259611627973 | 3.9166666666666656 | 0.28248604873249605 |
| ring | 5 | queue_aware | 15 | 0 | 0.23529411764705888 | 0 | 0.38041798812827726 |
| random_4_regular | 0 | blind_delay | 15 | 0 | 0.23529411764705888 | 0 | 0.092693194031318205 |
| random_4_regular | 0 | queue_aware | 15 | 0 | 0.23529411764705888 | 0 | 0.092693194031318205 |
| random_4_regular | 1 | blind_delay | 15 | 0 | 0.23529411764705888 | 0 | 0.082812141437372117 |
| random_4_regular | 1 | queue_aware | 15 | 0 | 0.23529411764705888 | 0 | 0.082812141437372117 |
| random_4_regular | 2 | blind_delay | 15 | 1.7269506701636845 | 0.02673891783500415 | 0.99166666666666703 | 0.086781576023524112 |
| random_4_regular | 2 | queue_aware | 15 | 0 | 0.23529411764705888 | 0 | 0.093401400557219308 |
| random_4_regular | 3 | blind_delay | 15 | 1.7138156366484962 | 0.02254785567130696 | 1.9750000000000005 | 0.07908332164369096 |
| random_4_regular | 3 | queue_aware | 15 | 0 | 0.23529411764705888 | 0 | 0.086791445965534841 |
| random_4_regular | 5 | blind_delay | 15 | 1.6735435319945327 | 0.028168040915788662 | 3.9166666666666656 | 0.078889586723262725 |
| random_4_regular | 5 | queue_aware | 15 | 0 | 0.23529411764705888 | 0 | 0.092078139895923575 |
| two_block | 0 | blind_delay | 15 | 0 | 0.23529411764705888 | 0 | 0.13840251377707738 |
| two_block | 0 | queue_aware | 15 | 0 | 0.23529411764705888 | 0 | 0.13840251377707738 |
| two_block | 1 | blind_delay | 15 | 0 | 0.23529411764705888 | 0 | 0.13861181799544048 |
| two_block | 1 | queue_aware | 15 | 0 | 0.23529411764705888 | 0 | 0.13861181799544048 |
| two_block | 2 | blind_delay | 15 | 1.9382171338603513 | 0.13008232474206036 | 0.99166666666666703 | 0.12553175103716663 |
| two_block | 2 | queue_aware | 15 | 0 | 0.23529411764705888 | 0 | 0.13861036324833173 |
| two_block | 3 | blind_delay | 15 | 1.9382171338603513 | 0.13008232474206036 | 1.9750000000000005 | 0.1137158759811825 |
| two_block | 3 | queue_aware | 15 | 0 | 0.23529411764705888 | 0 | 0.1388276901721133 |
| two_block | 5 | blind_delay | 15 | 1.9382171338603513 | 0.13008232474206036 | 3.9166666666666656 | 0.097394544855768858 |
| two_block | 5 | queue_aware | 15 | 0 | 0.23529411764705888 | 0 | 0.13822596250400881 |

The intended reading is comparative: if a delay effect survives topology changes, it is less likely to be a ring-only artifact; if it changes substantially, topology is part of the mechanism and must be treated as a registered factor in any larger study.
