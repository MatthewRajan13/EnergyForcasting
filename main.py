import json
import requests
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import nasdaqdatalink as ndl
from datetime import datetime
from datetime import timedelta
from scipy import stats
import torch
import torch.nn as nn
import numpy as np

GAS_EIA = {
    'U.S. Ending Stocks of Total Gasoline, Weekly': 'PET.WGTSTUS1.W',
    'U.S. Ending Stocks of Gasoline Blending Components, Weekly': 'PET.WBCSTUS1.W',
    'U.S. Ending Stocks of Fuel Ethanol, Weekly': 'PET.W_EPOOXE_SAE_NUS_MBBL.W',
    'U.S. Imports of Finished Motor Gasoline, Weekly': 'PET.W_EPM0F_IM0_NUS-Z00_MBBLD.W',
    'U.S. Percent Utilization of Refinery Operable Capacity, Weekly': 'PET.WPULEUS3.W',
    'U.S. Refiner and Blender Net Input of Fuel Ethanol, Weekly': 'PET.W_EPOOXE_YIR_NUS_MBBLD.W',
    'U.S. Refiner and Blender Net Input of Gasoline Blending Components, Weekly': 'PET.WBCRI_NUS_2.W'
}
CRUDE_EIA = {
    'U.S. Ending Stocks excluding SPR of Crude Oil, Weekly': 'PET.WCESTUS1.W',
    'U.S. Field Production of Crude Oil, Weekly': 'PET.WCRFPUS2.W',
    'U.S. Imports of Crude Oil, Weekly': 'PET.WCRIMUS2.W',
    'U.S. Product Supplied of Petroleum Products, Weekly': 'PET.WRPUPUS2.W',
    'U.S. Operable Crude Oil Distillation Capacity, Weekly': 'PET.WOCLEUS2.W',
    'U.S. Gross Inputs into Refineries, Weekly': 'PET.WGIRIUS2.W',
    'U.S. Refiner Net Input of Crude Oil, Weekly': 'PET.WCRRIUS2.W',
}

START_DATE = '20120918'  # 10 year back-test (Friday)
EIA_API = 'LPJxIMy69I8faXYsn4Ggo0z93f1SMXTwBxSiqvTh'
NDL_API = 'tH1_rNukQF2kHwQxi1xH'
FRED_API = '115cf5baf63cb8d94c61290db6e49ed4'


def main():
    combined_df = get_combined_df(GAS_EIA)
    normalized_df = stats.zscore(combined_df)

    print(normalized_df)

def get_combined_df(eia_choice):
    eia_data = get_eia_data(GAS_EIA, EIA_API)
    pres_data = get_pres_appr_data()
    election_data = get_days_til_election_data()
    fed_data = get_fed_funds_rate_data(NDL_API, START_DATE)
    treas_data = get_int_adj_10yr_treas_data(NDL_API, START_DATE)

    combined_data = pd.merge(eia_data, pres_data, left_index=True, right_index=True)
    combined_data = pd.merge(combined_data, election_data, left_index=True, right_index=True)
    combined_data = pd.merge(combined_data, fed_data, left_index=True, right_index=True)
    combined_data = pd.merge(combined_data, treas_data, left_index=True, right_index=True)

    return combined_data


def fill_eia_data(eia_data, code_dict):
    new_df = eia_data.head(1)
    beg_date = int(new_df.index[0])
    index_list = [beg_date]

    for i in range(len(eia_data)):
        if i is not len(eia_data) - 1:
            new_vals = []
            for key in code_dict.keys():
                curr = eia_data[key].loc[eia_data.index[i]]
                try:
                    nxt = eia_data[key].loc[eia_data.index[i + 1]]
                except IndexError:
                    return new_df

                diff = nxt - curr
                inc = diff / 7
                new_vals.append(curr + inc)

            for j in range(7):
                # Create new rows with interpolated values
                row = pd.Series(new_vals, index=new_df.columns)
                new_df = new_df.append(row, ignore_index=True)

                # Update date and check validity
                orig_date = datetime.strptime(str(beg_date), "%Y%m%d").date()

                curr_date = orig_date + timedelta(days=1)
                beg_date = int(curr_date.strftime('%Y%m%d'))
                index_list.append(beg_date)
                new_df.index = index_list

                new_vals = [z+inc for z in new_vals]

            new_vals.clear()
    return new_df


def get_eia_data(code_dict, api_key):
    output = pd.DataFrame()
    loaded = False
    for key in code_dict.keys():
        url = 'https://api.eia.gov/series/?api_key=' + api_key + '&series_id=' + code_dict[key]
        while not loaded:
            print("ATTEMPTING TO LOAD EIA DATA")
            r = requests.get(url=url)
            try:
                data = np.array(r.json()['series'][0]['data'])
                loaded = True
            except json.JSONDecodeError:
                print("FAILED ATTEMPT")
        df = pd.DataFrame(data=data, columns=['date', key])
        df = df.iloc[::-1]
        df.index = df.date
        df = df.drop('date', axis=1)
        output[key] = df[key]

    output = output[output.index >= START_DATE]  # Remove earlier dates
    output = output.astype(float)

    output = fill_eia_data(output, code_dict)
    return output


def fill_pres_data(pres_data):
    new_df = pres_data.head(1)
    beg_date = int(new_df.index[0])
    index_list = [beg_date]
    pres_data.index = pres_data.index.astype(int).astype(str)
    pres_index = pres_data.index.tolist()
    curr_date = datetime.strptime(str(int(pres_data.index[0])), "%Y%m%d").date()

    count = 0
    i = 0

    end_date = datetime.strptime(pres_index[-1], "%Y%m%d").date()
    end_date = end_date + timedelta(days=1)
    end_date = end_date.strftime("%Y%m%d")

    while curr_date.strftime("%Y%m%d") != end_date:

        if str(curr_date.strftime("%Y%m%d")) not in pres_index:
            count += 1
            new_vals = []
            prev = int(new_df['Approval Index'].loc[new_df.index[i]])
            new_vals.append(prev)

            # Create new rows with interpolated values
            row = pd.Series(new_vals, index=new_df.columns)
            new_df = new_df.append(row, ignore_index=True)

            index_list.append(int(str(curr_date.strftime("%Y%m%d"))))
            new_df.index = index_list

            new_vals.clear()
        else:
            new_vals = []
            curr = (pres_data['Approval Index'].loc[pres_data.index[i-count]])
            new_vals.append(curr)

            # Create new rows with interpolated values
            row = pd.Series(new_vals, index=new_df.columns)
            new_df = new_df.append(row, ignore_index=True)

            index_list.append(int(str(curr_date.strftime("%Y%m%d"))))
            new_df.index = index_list

            new_vals.clear()

        curr_date = curr_date + timedelta(days=1)
        i += 1

    return new_df.iloc[1:]


def get_pres_appr_data():
    # Biden
    biden_df = pd.read_html("https://www.rasmussenreports.com/public_content/politics/biden_administration"
                            "/biden_approval_index_history")
    biden_df = biden_df[0]
    biden_df = biden_df[biden_df.columns[0:2]]
    biden_df = biden_df.iloc[::-1].reset_index(drop=True)

    # Trump
    trump_df = pd.read_html("https://www.rasmussenreports.com/public_content/politics/trump_administration"
                            "/trump_approval_index_history")
    trump_df = trump_df[0]
    trump_df = trump_df[trump_df.columns[0:2]]
    trump_df = trump_df.iloc[::-1].reset_index(drop=True)

    # Obama
    obama_df = pd.read_html("https://www.rasmussenreports.com/public_content/politics/obama_administration"
                            "/obama_approval_index_history")
    obama_df = obama_df[0]
    obama_df = obama_df[obama_df.columns[0:2]]
    obama_df = obama_df.iloc[::-1].reset_index(drop=True)

    # Combine
    combined_df = pd.concat([obama_df, trump_df, biden_df])
    # Convert date
    months = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
              'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12', }
    years = {'08': '2008', '09': '2009', '10': '2010', '11': '2011', '12': '2012', '13': '2013', '14': '2014',
             '15': '2015', '16': '2016', '17': '2017', '18': '2018', '19': '2019', '20': '2020', '21': '2021',
             '22': '2022', }

    combined_df['Date'] = combined_df['Date'].str.replace('-', '')
    month_date = combined_df['Date'].str[2:5].replace(months)
    year_date = combined_df['Date'].str[5:].replace(years)
    day_date = combined_df['Date'].str[:2]
    combined_df['Date'] = year_date + month_date + day_date
    combined_df.index = combined_df.Date
    combined_df.index = combined_df.index.astype(float)
    combined_df = combined_df.drop('Date', axis=1)
    combined_df = combined_df.dropna()
    for i in range(len(combined_df)):
        if "No Polling" in combined_df['Approval Index'].loc[combined_df.index[i]] \
                or "No polling" in combined_df['Approval Index'].loc[combined_df.index[i]]:
            dup = combined_df['Approval Index'].loc[combined_df.index[i-1]]
            combined_df['Approval Index'].loc[combined_df.index[i]] = dup

    combined_df = combined_df.astype(int)
    combined_df = combined_df[combined_df.index >= int(START_DATE)]  # Remove earlier dates
    combined_df = fill_pres_data(combined_df)

    for i in range(len(combined_df['Approval Index'])):
        if type(combined_df['Approval Index'].loc[combined_df.index[i]]) != int:
            combined_df['Approval Index'].loc[combined_df.index[i]] = combined_df['Approval Index'].loc[combined_df.index[i-1]]

    return combined_df.astype(int)


def find_election_dates(curr_date, end_date):
    # Presidential Elections: First tuesday after Nov 1st on years divisible by 4
    # Midterm Elections: Tuesday after the first Monday in November 2 in years divisible
    # by 2 that is not the presidential election

    # Find next election date
    curr_year = int(str(curr_date.strftime("%Y%m%d"))[:4])
    curr_month = int(str(curr_date.strftime("%Y%m%d"))[4:6])
    curr_day = int(str(curr_date.strftime("%Y%m%d"))[6:])

    end_year = int(str(end_date.strftime("%Y%m%d"))[:4])
    end_month = int(str(end_date.strftime("%Y%m%d"))[4:6])
    end_day = int(str(end_date.strftime("%Y%m%d"))[6:])

    if end_month >= 11 and end_day > 8:
        end_date = datetime((end_year + 4), end_month, end_day)
    else:
        end_date = datetime((end_year + 3), end_month, end_day)

    pres_dates = []
    mid_dates = []

    while curr_date <= end_date:
        if curr_year % 4 == 0:  # Presidential
            if curr_month == 11:
                if curr_day > 8:
                    curr_date = curr_date + timedelta(days=(730 - curr_day))
                elif curr_day != 1 and curr_date.weekday() == 1:  # Tuesday:
                    pres_dates.append(int(str(curr_date.strftime("%Y%m%d"))))
                    curr_date = curr_date + timedelta(days=(730 - curr_day))
                else:
                    curr_date = curr_date + timedelta(days=1)

            elif curr_month == 12:
                curr_date = datetime((curr_year + 2), 11, 1)
            else:
                curr_date = datetime(curr_year, 11, 1)

        elif curr_year % 2 == 0:  # Midterms
            if curr_month == 11:
                if curr_day > 8:
                    curr_date = curr_date + timedelta(days=(730 - curr_day))
                elif curr_day != 1 and curr_date.weekday() == 1:  # Tuesday:
                    mid_dates.append(int(str(curr_date.strftime("%Y%m%d"))))
                    curr_date = curr_date + timedelta(days=(730 - curr_day))
                else:
                    curr_date = curr_date + timedelta(days=1)

            elif curr_month == 12:
                curr_date = datetime((curr_year + 2), 11, 1)
            else:
                curr_date = datetime(curr_year, 11, 1)

        else:
            curr_date = datetime((curr_year + 1), 11, 1)

        curr_year = int(str(curr_date.strftime("%Y%m%d"))[:4])
        curr_month = int(str(curr_date.strftime("%Y%m%d"))[4:6])
        curr_day = int(str(curr_date.strftime("%Y%m%d"))[6:])

    return mid_dates, pres_dates


def get_days_til_election_data():

    curr_date = datetime.strptime(START_DATE, "%Y%m%d")
    end_date = datetime.now()

    mid_dates, pres_dates = find_election_dates(curr_date, end_date)

    index_list = []
    days_til_mid = []
    days_til_pres = []

    for election in mid_dates:
        el_date = str(election)
        el_date = datetime.strptime(el_date, "%Y%m%d")
        while curr_date <= el_date:
            date_delta = el_date - curr_date
            date_delta = str(date_delta).split(' ')
            if len(date_delta) == 1:
                days_til_mid.append(0)
            else:
                days_til_mid.append(int(date_delta[0]))

            int_curr = int(str(curr_date.strftime("%Y%m%d")))
            index_list.append(int_curr)

            curr_date = curr_date + timedelta(days=1)

    curr_date = datetime.strptime(START_DATE, "%Y%m%d")

    for election in pres_dates:
        el_date = str(election)
        el_date = datetime.strptime(el_date, "%Y%m%d")
        while curr_date <= el_date:
            date_delta = el_date - curr_date
            date_delta = str(date_delta).split(' ')
            if len(date_delta) == 1:
                days_til_pres.append(0)
            else:
                days_til_pres.append(int(date_delta[0]))

            curr_date = curr_date + timedelta(days=1)

    zipped = list(zip(days_til_mid, days_til_pres))
    combined_df = pd.DataFrame(zipped, columns=['Days til Midterms', 'Days til Presidential'])
    combined_df.index = index_list

    return combined_df


def get_fed_funds_rate_data(api_key, start_date):
    ndl.read_key(api_key)
    fed_funds = ndl.get("FED/RIFSPFF_N_B", start_date=start_date)
    fed_funds.index = fed_funds.index.strftime("%Y%m%d")
    fed_funds.index = fed_funds.index.astype(int)

    fed_funds.columns = ['Fed Funds Rate']

    return fed_funds


def get_int_adj_10yr_treas_data(api_key, start_date):
    ndl.read_key(api_key)
    t10ii = ndl.get("FRED/DFII10", start_date=start_date)
    t10ii.index = t10ii.index.strftime("%Y%m%d")
    t10ii.index = t10ii.index.astype(int)

    t10ii.columns = ['Interest Adj. Treasury Data']

    return t10ii


if __name__ == "__main__":
    main()
