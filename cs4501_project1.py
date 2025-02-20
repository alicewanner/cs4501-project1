import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import numpy as np

df = pd.read_csv("Activities - A list of Google services accessed by.csv")
df = df[df["Activity Timestamp"] < "2025-02-14 04:00:00"]
null_cols = df.columns[(df.isna() | (df == "")).all()].tolist()
df = df.drop(columns=null_cols)
df = df.drop(columns=["Gaia ID", "Is Non-routable IP Address", "Sub-Product Name", "Gmail Access Channel"])
df = df[df['Activity Type'] != "Non User Initiated"]
df = df[df['Product Name'] != "Chrome Sync"]
df = df[df['User Agent String'] != "App : APPLE_NATIVE_APP. Os : MAC_OS."]

df['Activity Timestamp'] = pd.to_datetime(df['Activity Timestamp'], errors='coerce')
df = df.sort_values(by="Activity Timestamp")
df = df.groupby('Activity Timestamp', as_index=False).first()
df.set_index('Activity Timestamp', inplace=True)

if df.index.tz is None:
    df.index = df.index.tz_localize('UTC')
df.index = df.index.tz_convert('US/Eastern')

# resample data to 5-minute intervals
df_resampled = df.resample('5min').size()
df_resampled = pd.DataFrame({'Activity Count': df_resampled})

unique_dates = pd.Series(df_resampled.index.date).unique()

activity_periods = []

for date in unique_dates:
    start_time = pd.Timestamp(f"{date} 04:00:00", tz='US/Eastern')  # Start at 4:00 AM
    end_time = start_time + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)  # End at 3:55 AM next day

    daily_data = df_resampled.loc[start_time:end_time]

    # activity threshold
    threshold = 15 if date in [pd.Timestamp("2025-02-03").date(), pd.Timestamp("2025-02-04").date()] else 10

    daily_filtered = daily_data[daily_data['Activity Count'] > threshold]

    if not daily_filtered.empty:
        first_time = daily_filtered.index[0].time()  # Extract time only
        last_time = daily_filtered.index[-1].time()

        weekday = pd.Timestamp(date).day_name()  # Get the day of the week
        activity_periods.append((date, first_time, last_time, weekday))

activity_df = pd.DataFrame(activity_periods, columns=["Date", "First Activity", "Last Activity", "Weekday"])

weekday_colors = {"Thursday": "darkorange", "Friday": "darkorange", "Saturday": "darkorange"}
default_color = "royalblue"

plt.figure(figsize=(10, 6))

time_start = pd.Timestamp("2000-01-01 04:00:00")  # Reference time
time_end = pd.Timestamp("2000-01-02 03:55:00")  # Next day, 3:55 AM

legend_labels = {}
for i, row in activity_df.iterrows():
    first_time = pd.Timestamp.combine(pd.Timestamp("2000-01-01"), row["First Activity"])
    last_time = pd.Timestamp.combine(pd.Timestamp("2000-01-01"), row["Last Activity"])

    if last_time < first_time:
        last_time += pd.Timedelta(days=1)

    color = weekday_colors.get(row["Weekday"], default_color)

    line, = plt.plot([first_time, last_time], [i, i], marker="o", markersize=6, linestyle="-", color=color)

    if color not in legend_labels:
        legend_labels[color] = "Thursday - Saturday" if color == "darkorange" else "Sunday - Wednesday"

# configure x-axis
plt.xlim(time_start, time_end)
plt.xticks(pd.date_range(start=time_start, end=time_end, freq="2h"), rotation=45)
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.yticks(range(len(activity_df)), activity_df["Date"].astype(str))
plt.xlabel("Time of Day", fontsize=14)
plt.ylabel("Date", fontsize=14)
plt.title("Google Activity Periods by Day", fontsize=16)

plt.grid(axis="x", linestyle="dotted", alpha=0.3, color="gray")

plt.legend(handles=[mlines.Line2D([0], [0], color=color, marker='o', markersize=8, linestyle='-', lw=2)
                    for color in legend_labels.keys()],
           labels=legend_labels.values(),
           loc="lower right",
           title="Days of the Week")

plt.tight_layout()
# plt.show()


def plot_activity_with_avg(activity_df):
    """
    Plots the activity periods as horizontal bars and adds vertical lines for average first and last activity times.
    """
    plt.figure(figsize=(10, 6))

    time_start = pd.Timestamp("2000-01-01 04:00:00")
    time_end = pd.Timestamp("2000-01-02 03:55:00")

    weekday_colors = {"Thursday": "darkgray", "Friday": "darkgray", "Saturday": "darkgray"}
    default_color = "darkgray"

    legend_labels = {}

    first_times = []
    last_times = []

    for i, row in activity_df.iterrows():
        first_time = pd.Timestamp.combine(pd.Timestamp("2000-01-01"), row["First Activity"])
        last_time = pd.Timestamp.combine(pd.Timestamp("2000-01-01"), row["Last Activity"])

        if last_time < first_time:
            last_time += pd.Timedelta(days=1)

        first_times.append(first_time)
        last_times.append(last_time)

        color = weekday_colors.get(row["Weekday"], default_color)
        line, = plt.plot([first_time, last_time], [i, i], marker="o", markersize=6, linestyle="-", color=color)

        if color not in legend_labels:
            legend_labels[color] = "Thursday - Saturday" if color == "darkorange" else "Sunday - Wednesday"

    avg_first_time = pd.Timestamp("2000-01-01") + pd.Timedelta(
        seconds=np.mean([(t - pd.Timestamp("2000-01-01")).total_seconds() for t in first_times]))
    avg_last_time = pd.Timestamp("2000-01-01") + pd.Timedelta(
        seconds=np.mean([(t - pd.Timestamp("2000-01-01")).total_seconds() for t in last_times]))

    print("avg_first_time", avg_first_time)
    print("avg_last_time", avg_last_time)

    median_first_time = pd.Timestamp("2000-01-01") + pd.Timedelta(
        seconds=np.median([(t - pd.Timestamp("2000-01-01")).total_seconds() for t in first_times]))
    median_last_time = pd.Timestamp("2000-01-01") + pd.Timedelta(
        seconds=np.median([(t - pd.Timestamp("2000-01-01")).total_seconds() for t in last_times]))

    print("median_first_time", median_first_time)
    print("median_last_time", median_last_time)


    plt.axvline(x=avg_first_time, color='darkorange', linestyle='--', label='Avg First Activity', linewidth=3)
    plt.axvline(x=avg_last_time, color='royalblue', linestyle='--', label='Avg Last Activity', linewidth=3)

    plt.xlim(time_start, time_end)
    plt.xticks(pd.date_range(start=time_start, end=time_end, freq="2h"), rotation=45)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    plt.yticks(range(len(activity_df)), activity_df["Date"].astype(str))
    plt.xlabel("Time of Day", fontsize=14)
    plt.ylabel("Date", fontsize=14)
    plt.title("Google Activity Trends", fontsize=16)

    plt.grid(axis="x", linestyle="dotted", alpha=0.3, color="gray")

    plt.legend(handles=[mlines.Line2D([0], [0], color='darkorange', linestyle='--', lw=2),
                        mlines.Line2D([0], [0], color='royalblue', linestyle='--', lw=2)],
               labels=["Avg First Activity", "Avg Last Activity"],
               loc="lower right",
               title="Legend")

    plt.tight_layout()
    plt.show()


plot_activity_with_avg(activity_df)


def plot_sleep_hours(activity_df):
    """
    Plots the number of hours asleep each night as a bar graph based on last activity and first activity times.
    """
    sleep_durations = []
    dates = []

    for _, row in activity_df.iterrows():
        first_time = pd.Timestamp.combine(pd.Timestamp("2000-01-01"), row["First Activity"])
        last_time = pd.Timestamp.combine(pd.Timestamp("2000-01-01"), row["Last Activity"])

        if last_time < first_time:
            last_time += pd.Timedelta(days=1)

        sleep_duration = 24 - (last_time - first_time).total_seconds() / 3600
        sleep_durations.append(sleep_duration)
        dates.append(row["Date"].strftime('%Y-%m-%d'))

    avg_sleep_duration = sum(sleep_durations) / len(sleep_durations) if sleep_durations else 0

    print("avg_sleep_duration", avg_sleep_duration)

    plt.figure(figsize=(10, 6))
    plt.bar(dates, sleep_durations, color="gray")

    # plt.axhline(y=avg_sleep_duration, color='darkorange', linestyle='dotted', linewidth=2, label='Avg Sleep Duration')

    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Hours Asleep", fontsize=14)
    plt.title("Sleep Duration Per Night", fontsize=16)

    plt.xticks(ticks=range(len(dates)), labels=dates, rotation=90)
    plt.ylim(0, 17)
    plt.grid(axis="y", linestyle="dotted", alpha=0.3, color="gray")

    plt.legend()
    plt.tight_layout()
    plt.show()


plot_sleep_hours(activity_df)

def plot_sleep_hours_average(activity_df):
    """
    Plots the number of hours asleep each night as a bar graph based on last activity and first activity times,
    excluding nights where sleep duration exceeds 12 hours.
    """
    sleep_durations = []
    dates = []

    avg_sleep_duration_tot = 0
    ct = 0

    for _, row in activity_df.iterrows():
        first_time = pd.Timestamp.combine(pd.Timestamp("2000-01-01"), row["First Activity"])
        last_time = pd.Timestamp.combine(pd.Timestamp("2000-01-01"), row["Last Activity"])

        if last_time < first_time:
            last_time += pd.Timedelta(days=1)

        sleep_duration = 24 - (last_time - first_time).total_seconds() / 3600

        if sleep_duration <= 11:  # Exclude sleep durations greater than 12 hours
            sleep_durations.append(sleep_duration)
            dates.append(row["Date"].strftime('%Y-%m-%d'))
            avg_sleep_duration_tot += sleep_duration
            ct += 1

    avg_sleep_duration = avg_sleep_duration_tot / ct if sleep_durations else 0
    print("avg_sleep_duration no >11", avg_sleep_duration)
    plt.figure(figsize=(10, 6))
    plt.bar(dates, sleep_durations, color="gray")

    plt.axhline(y=avg_sleep_duration, color='darkorange', linestyle='--', linewidth=2, label='Avg Sleep Duration')

    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Hours Asleep", fontsize=14)
    plt.title("Sleep Duration Per Night (Outliers Removed and Average Included)", fontsize=16)

    plt.xticks(ticks=range(len(dates)), labels=dates, rotation=90)
    plt.ylim(0, 12)
    plt.grid(axis="y", linestyle="dotted", alpha=0.3, color="gray")

    plt.legend()
    plt.tight_layout()
    plt.show()

plot_sleep_hours_average(activity_df)
