import re
import pandas as pd
from dateutil import parser
def preprocess(data):
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)
    dates = [parser.parse(d.strip(' -'), dayfirst=True) for d in dates]
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    # df['message_date'] = pd.to_datetime(df['message_date'], format="%d/%m/%Y, %H:%M - ")
    df.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages = []

    for message in df['user_message']:
        entry = re.split(r'([\w\W]+?):\s', message, maxsplit=1)
        if len(entry) > 2:
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)
    df['days']=df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.strftime('%B')
    df['month_number'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minutes'] = df['date'].dt.minute

    time_period=[]
    for hour in df[['day_name','hour']]['hour']:
        if hour==23:
            time_period.append(str(hour)+"-"+str('00'))
        elif hour==0:
            time_period.append(str('00')+"-"+str(hour+1))
        else:
            time_period.append(str(hour)+"-"+str(hour+1))

    df['period'] = time_period
    return df

