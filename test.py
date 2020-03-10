import pandas as pd
import matplotlib.pyplot as plt

file_path = 'input/train_sample.csv'
clicks_data = pd.read_csv(file_path, parse_dates=['click_time'])

past_events = pd.Series(clicks_data.index, index=clicks_data.click_time, name="past_events_6_hours").sort_index()
count_past_events = past_events.rolling('1h').count() - 1
count_past_events.index = past_events.values
count_past_events = count_past_events.reindex(clicks_data.index)
clicks_data = clicks_data.join(count_past_events)
print(clicks_data.columns)
print(clicks_data.tail())


# first = clicks_data['click_time'][0]
# print("first: ", first)
# last = clicks_data['click_time'][len(clicks_data) - 1]
# print("last: ", last)
#
# print("Difference in second: ", (last - first).total_seconds())


def time_diff(series):
    """ Returns a series with the time since the last timestamp in seconds """
    last = clicks_data['click_times'][len(clicks_data) - 1]
    result = pd.Series((last - clicks_data['click_times']).total_seconds(), index=clicks_data.click_time,
                       name="since_last_timestamp")
    return result


print(clicks_data['is_attributed'] == 1)


def previous_attributions(series):
    """ Returns a series with the number of times an app has been download"""
    result = pd.Series(series[''])
    pass
