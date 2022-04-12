import re
import pandas as pd


def preprocess(data):
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({'USER_MESSAGE': messages, 'MESSAGE_DATE': dates})
    # converting the Message_date type
    df['MESSAGE_DATE'] = pd.to_datetime(df['MESSAGE_DATE'], format='%m/%d/%y, %H:%M - ')

    df.rename(columns={'MESSAGE_DATE': 'DATE'}, inplace=True)

    users = []
    messages = []
    for message in df['USER_MESSAGE']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:  # username
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group_notif')
            messages.append(entry[0])

    df['USERS'] = users
    df['MESSAGES'] = messages
    df.drop(columns=['USER_MESSAGE'], inplace=True)

    df['ONLY_DATE'] = df['DATE'].dt.date
    df['YEAR'] = df['DATE'].dt.year
    df['DAY_NAME'] = df['DATE'].dt.day_name()
    df['MONTH_NUM'] = df['DATE'].dt.month
    df['MONTH'] = df['DATE'].dt.month_name()
    df['DAY'] = df['DATE'].dt.day
    df['HOUR'] = df['DATE'].dt.hour
    df['MINUTE'] = df['DATE'].dt.minute

    period = []
    for hour in df[['DAY_NAME', 'HOUR']]['HOUR']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['PERIOD'] = period

    return df
