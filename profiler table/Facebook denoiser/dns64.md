| Name                                                   | Self CPU % | Self CPU   | CPU total % | CPU total   | CPU time avg | # of Calls |
|--------------------------------------------------------|------------|------------|-------------|-------------|--------------|------------|
| aten::uniform_                                        | 83.33%     | 136.492ms  | 83.33%      | 136.492ms   | 2.844ms      | 48         |
| aten::copy_                                           | 8.50%      | 13.924ms   | 8.50%       | 13.924ms    | 133.885us    | 104        |
| aten::std                                             | 1.50%      | 2.461ms    | 5.63%       | 9.229ms     | 461.450us    | 20         |
| aten::mean                                            | 0.45%      | 733.000us  | 4.12%       | 6.742ms     | 306.455us    | 22         |
| aten::sum                                             | 3.19%      | 5.219ms    | 3.31%       | 5.417ms     | 270.850us    | 20         |
| enumerate(DataLoader)#_SingleProcessDataLoaderIter._...| 0.96%      | 1.570ms    | 1.08%       | 1.777ms     | 888.500us    | 2          |
| aten::div_                                            | 0.71%      | 1.164ms    | 0.90%       | 1.473ms     | 23.758us     | 62         |
| aten::pow                                             | 0.35%      | 567.000us  | 0.35%       | 569.000us   | 28.450us     | 20         |
| other:                                                 | 0.90%      | 1.480ms    | 1.43%       | 2.344ms     | 104.018us    | 23         |
Self CPU time total: 163.794ms 