# Dabba forecast backtest: last 28 days, 14 dish x meal series, predicted at 9 pm the night before

| model                                |   MAE (dabbas) |   over-cooked / day |   under-cooked / day |   exact days % |   80% band coverage |
|:-------------------------------------|---------------:|--------------------:|---------------------:|---------------:|--------------------:|
| tabpfn                               |           0.9  |                3.93 |                 8.68 |          50.51 |               96.68 |
| gradient boosting                    |           1.07 |                7.32 |                 7.71 |          39.29 |              nan    |
| same as yesterday                    |           2.56 |               18.07 |                17.71 |          23.98 |              nan    |
| last week, same weekday              |           1.79 |               12.61 |                12.5  |          31.63 |              nan    |
| 7-day average                        |           2.3  |               15.68 |                16.5  |          26.28 |              nan    |
| advance orders / usual advance share |           1.55 |               12.39 |                 9.25 |          36.22 |              nan    |

TabPFN fit+predict time total: nans (client). Over/under-cooked are summed across all dishes and meals per day.
