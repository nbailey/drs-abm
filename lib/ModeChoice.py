import numpy as np
import pandas as pd 
import csv

from lib.Constants import INI_WAIT, INI_DETOUR
from lib.ModeChoiceVars import *

'''distance conversion'''
km2mile = 1.609344
m2mile = 1609.344

'''currency conversion'''
gbp2usd = 1.33

def taxifare_calc(autodis):
    '''
    calculates taxi fare based on the distance of the car trip
    TfL website used - a rough piecewise function was created based on their given values
    '''
    dist_brkup_1 = min(autodis/1000, 1)
    dist_brkup_2 = max(0, min(autodis/1000-1, 1))
    dist_brkup_3 = max(0, min(autodis/1000-2, 2))
    dist_brkup_4 = max(0, min(autodis/1000-4, 2))
    dist_brkup_5 = max(0, autodis/1000-6)
    taxi_fare = 2.6 + ( 5.1 * dist_brkup_1 + 4.1 * dist_brkup_2 + 3.85 * dist_brkup_3 + 4 * (dist_brkup_4 + dist_brkup_5) ) / km2mile
    return taxi_fare

def transit_utility_calc(transit_utility, MU_TRANSIT):
    if transit_utility == 0:
        b = 0
    else:
        b = np.exp(MU_TRANSIT*transit_utility)
    return b

def expsum(df,MU_TRANSIT):
    if df['av5'] == 0 and df['av6'] == 0 and df['av8'] == 0:
        value = df['car_utility'] + df['walk_utility'] + df['bike_utility'] + df['taxi_utility']
    else:
        value = np.exp(1/MU_TRANSIT * np.log(df['bus_utility'] + df['rail_utility'] + df['intermodal_utility'])) \
    + df['car_utility'] + df['walk_utility'] + df['bike_utility'] + df['taxi_utility']
    return value

def calcprob_noAVPT(df,MU_TRANSIT,identifier):
    if identifier == 0:
        df['utility'] = df['bus_utility']
    elif identifier == 1:
        df['utility'] = df['rail_utility']
    elif identifier == 2:
        df['utility'] = df['intermodal_utility']
    df["sum_noAVPT_transit_utility"]= df['bus_utility'] + df['rail_utility'] + df['intermodal_utility']
    if df['av5'] == 0 and df['av6'] == 0 and df['av8'] == 0:
        prob = 0
    else:
        prob = np.exp(1/MU_TRANSIT * np.log(df['sum_noAVPT_transit_utility']))/df['expsum_wout_AVPT'] * df['utility']/df['sum_noAVPT_transit_utility']
    return prob

def diffinprob(df):

    df['prob_car_woutAVPT'] = df['car_utility']/df['expsum_wout_AVPT']
    df['prob_walk_woutAVPT'] = df['walk_utility']/df['expsum_wout_AVPT']
    df['prob_bike_woutAVPT'] = df['bike_utility']/df['expsum_wout_AVPT']
    df['prob_taxi_woutAVPT'] = df['taxi_utility']/df['expsum_wout_AVPT']
    #pass in identifier with 1 = bus, 2 = rail, 3 = intermodal
    identifier = 0
    df['prob_bus_woutAVPT'] = df.apply(lambda value: calcprob_noAVPT(value, MU_TRANSIT, identifier), axis = 1)
    identifier = 1
    df['prob_rail_woutAVPT'] = df.apply(lambda value: calcprob_noAVPT(value, MU_TRANSIT, identifier), axis = 1)
    identifier = 2
    df['prob_intermodal_woutAVPT'] = df.apply(lambda value: calcprob_noAVPT(value, MU_TRANSIT, identifier), axis = 1)

    df['differenceinprob_car'] = df['prob_car'] - df['prob_car_woutAVPT']
    df['differenceinprob_walk'] = df['prob_walk'] - df['prob_walk_woutAVPT']
    df['differenceinprob_bike'] = df['prob_bike'] - df['prob_bike_woutAVPT']
    df['differenceinprob_taxi'] = df['prob_taxi'] - df['prob_taxi_woutAVPT']
    df['differenceinprob_bus'] = df['prob_bus'] - df['prob_bus_woutAVPT']
    df['differenceinprob_rail'] = df['prob_rail'] - df['prob_rail_woutAVPT']
    df['differenceinprob_intermodal'] = df['prob_intermodal'] - df['prob_intermodal_woutAVPT']

    return df

def main_CBD(filename, ASC_AVPT, fare, df_OD_LOS):
    df_1 = pd.read_csv(filename)
    df = pd.concat([df_1, df_OD_LOS], axis=1)

    print("----------------------------------------")
    name = str(filename)
    with open('output/OD.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([None])
        row = [name]
        writer.writerow(row)
    df.to_csv('output/OD.csv', mode='a', header=True)
    print("----------------------------------------")

    sharing_discount = fare[3]
    transit_connect_discount = fare[4]/gbp2usd
    min_cost_avpt = fare[5]/gbp2usd
    taxi_wait_time = 0.3 #???

    '''DATA PREP'''
    #calculate taxi fare
    df['taxi_fare'] = df['autodis'].apply(taxifare_calc)
    # av dist, time columns -- not necessary for cbd trips
    # add av cost column
    df['cost_temp_1'] = (fare[0] + fare[1] * df['avtime'] / 60 + fare[2] * df['avdis']/1000)*sharing_discount / gbp2usd
    df['cost_temp_2'] = df['cost_temp_1'].apply(lambda value: max(value, min_cost_avpt))
    df['AV+PT_cost'] = df['cost_temp_2'] - transit_connect_discount + df['rail_fare']
    '''end of DATA PREP'''

    '''define utility equations'''
    df['car_utility'] = ASC_CAR + B_CAR_TT * df['autotime']/600 + B_CAR_NUM * df['num_cars'] \
                        + B_COST * (df['parking_fare'] + df['autodis'] * 5.2416/(1000*76.165) + df['congcharg'] * 11.5)
    df['walk_utility'] = B_WALK_TT * df['walktime']/600
    df['bike_utility'] = ASC_BIKE + B_BIKE_TT * df['biketime']/600
    df['taxi_utility'] = ASC_TAXI + B_TAXI_TT*(df['autotime']/600+taxi_wait_time) + B_TAXI_DIS * df['autodis'] /1000 +  B_COST *  df['taxi_fare'] # assume 3 min waiting time for taxi 
    df['bus_utility'] = (ASC_BUS + B_BUS_TT * (df['bus_transittime'] - df['bus_transitwalktime'])/600 + B_COST * df['bus_fare'] \
                        + B_TRANSITNUMTRANSFERS * df['bus_transitnumtransfers'] + B_TRANSITWALK_TT * df['bus_transitwalktime']/600 ) * df['av5']
    df['rail_utility'] = (ASC_RAIL + B_RAIL_TT * (df['rail_transittime'] - df['rail_transitwalktime'])/600 + B_COST * df['rail_fare'] \
                        + B_TRANSITNUMTRANSFERS * df['rail_transitnumtransfers'] + B_TRANSITWALK_TT * df['rail_transitwalktime']/600 ) * df['av6']
    df['intermodal_utility'] = (ASC_PnK_RIDE + B_PnK_RIDE_TT * (df['pnk_transittime'] - df['pnk_transitwalktime'])/600 \
                              + B_COST * df['pnk_fare'] + B_TRANSITNUMTRANSFERS * (df['pnk_transitnumtransfers']+1) + B_TRANSITWALK_TT *df['pnk_transitwalktime']/600 ) * df['av8']
    df['AVPT_utility'] = ASC_AVPT + BETA_AVPT_CAR_TT * (df['detour_factor']/600 * df['avtime'] + df['wait_time']/600) + BETA_AVPT_PT_TT * df['avpttime']/600 \
                        + B_TRANSITNUMTRANSFERS * df['avptNumSteps'] \
                        + BETA_AVPT_COST * df['AV+PT_cost']
    # its num of steps not num of transfers, the extra step required to transfer to AV is accounted for by the num of steps

    '''exponent each utility equation'''
    df['car_utility'] = np.exp(df['car_utility'])
    df['walk_utility'] = np.exp(df['walk_utility'])
    df['bike_utility'] = np.exp(df['bike_utility'])
    df['taxi_utility'] = np.exp(df['taxi_utility'])
    df['bus_utility'] = df['bus_utility'].apply(lambda value: transit_utility_calc(value, MU_TRANSIT))
    df['rail_utility'] = df['rail_utility'].apply(lambda value: transit_utility_calc(value, MU_TRANSIT))
    df['intermodal_utility'] = df['intermodal_utility'].apply(lambda value: transit_utility_calc(value, MU_TRANSIT))
    df['AVPT_utility'] = np.exp(df['AVPT_utility']*MU_TRANSIT)

    df['exp_sum']  = np.exp(1/MU_TRANSIT * np.log(df['bus_utility'] + df['rail_utility'] + df['intermodal_utility'] + df['AVPT_utility'])) \
        + df['car_utility'] + df['walk_utility'] + df['bike_utility'] + df['taxi_utility']
    df['expsum_wout_AVPT'] = df.apply(lambda value: expsum(value, MU_TRANSIT), axis = 1)
    df['log_sum'] = np.log(df['exp_sum']) * df['expan_fac/10']
    df['log_sum_wout_AVPT'] = np.log(df['expsum_wout_AVPT']) * df['expan_fac/10']

    '''
    calc prob with AV+PT
    '''
    df['sum_transit_utility'] = df['bus_utility'] + df['rail_utility'] + df['intermodal_utility'] + df['AVPT_utility']
    df['prob_car'] = df['car_utility']/df['exp_sum']
    df['prob_walk'] = df['walk_utility']/df['exp_sum']
    df['prob_bike'] = df['bike_utility']/df['exp_sum']
    df['prob_taxi'] = df['taxi_utility']/df['exp_sum']
    df['prob_bus'] = np.exp(1/MU_TRANSIT * np.log(df['sum_transit_utility'])) \
                        /df['exp_sum'] * df['bus_utility']/df['sum_transit_utility']
    df['prob_rail'] = np.exp(1/MU_TRANSIT * np.log(df['sum_transit_utility'])) \
                        /df['exp_sum'] * df['rail_utility']/df['sum_transit_utility']
    df['prob_intermodal'] = np.exp(1/MU_TRANSIT * np.log(df['sum_transit_utility'])) \
                        /df['exp_sum'] * df['intermodal_utility']/df['sum_transit_utility']
    df['prob_AVPT'] = np.exp(1/MU_TRANSIT * np.log(df['sum_transit_utility'])) \
                        /df['exp_sum'] * df['AVPT_utility']/df['sum_transit_utility']

    for _, row in df.iterrows():
        assert np.isclose(row['prob_car'] + row['prob_walk'] + row['prob_bike'] + row['prob_taxi'] + row['prob_bus'] + \
        row['prob_rail'] + row['prob_intermodal'] + row['prob_AVPT'], 1.0)

    df1 = diffinprob(df)

    df['AVPT_choice'] = df['prob_AVPT'] * df['expan_fac/10']

    df['differenceinprob_car'] = (df['prob_car'] - df['prob_car_woutAVPT'])/df['prob_AVPT']
    df['differenceinprob_walk'] = (df['prob_walk'] - df['prob_walk_woutAVPT'])/df['prob_AVPT']
    df['differenceinprob_bike'] = (df['prob_bike'] - df['prob_bike_woutAVPT'])/df['prob_AVPT']
    df['differenceinprob_taxi'] = (df['prob_taxi'] - df['prob_taxi_woutAVPT'])/df['prob_AVPT']
    df['differenceinprob_bus'] = (df['prob_bus'] - df['prob_bus_woutAVPT'])/df['prob_AVPT']
    df['differenceinprob_rail'] = (df['prob_rail'] - df['prob_rail_woutAVPT'])/df['prob_AVPT']
    df['differenceinprob_intermodal'] = (df['prob_intermodal'] - df['prob_intermodal_woutAVPT'])/df['prob_AVPT']

    df_temp1 = df[['expan_fac','expan_fac/10','AVPT_choice','log_sum', 'log_sum_wout_AVPT','prob_AVPT','detour_factor','wait_time']]
    df_temp2 = df1[['differenceinprob_car','differenceinprob_walk','differenceinprob_bike','differenceinprob_taxi','differenceinprob_bus','differenceinprob_rail','differenceinprob_intermodal']]
    df_return = pd.concat([df_temp1, df_temp2], axis = 1)
    return df_return

def main_intrazonal(filename, ASC_AVPT, fare, df_OD_LOS):
    df_1 = pd.read_csv(filename)
    df = pd.concat([df_1, df_OD_LOS], axis=1)

    print("----------------------------------------")
    name = str(filename)
    with open('output/OD.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([None])
        row = [name]
        writer.writerow(row)
    df.to_csv('output/OD.csv', mode='a', header=True)
    print("----------------------------------------")

    sharing_discount = fare[3]
    transit_connect_discount = fare[4]/gbp2usd
    min_cost_avpt = fare[5]/gbp2usd
    taxi_wait_time = 0.3

    '''DATA PREP'''
    #calculate taxi fare
    df['taxi_fare'] = df['autodis'].apply(taxifare_calc)

    # add av dist, time, cost columns
    df['AV_dist'] = df['autodis']
    df['AV_time'] = df['autotime']
    df['temp'] = (fare[0] + fare[1] * df['AV_time'] / 60 + fare[2] * df['AV_dist']/1000)*sharing_discount/gbp2usd # divide by 1.33 to make into pounds
    df['AV+PT_cost'] = df['temp'].apply(lambda value: max(value, min_cost_avpt))
    '''end of DATA PREP'''

    '''define utility equations'''
    df['car_utility'] = ASC_CAR + B_CAR_TT * df['autotime']/600 + B_CAR_NUM * df['num_cars'] \
        + B_COST * (df['parking_fare']+ df['autodis'] * 5.2416/(1000*76.165) + df['congcharg'] * 11.5)
    df['walk_utility'] = B_WALK_TT * df['walktime']/600
    df['bike_utility'] = ASC_BIKE + B_BIKE_TT * df['biketime']/600
    df['taxi_utility'] = ASC_TAXI + B_TAXI_TT*(df['autotime']/600+taxi_wait_time) + B_TAXI_DIS * df['autodis'] /1000 +  B_COST *  df['taxi_fare'] # assume 3 min waiting time for taxi
    df['bus_utility'] = (ASC_BUS + B_BUS_TT * (df['bus_transittime'] - df['bus_transitwalktime'])/600 + B_COST * df['bus_fare'] \
                         + B_TRANSITNUMTRANSFERS * df['bus_transitnumtransfers'] + B_TRANSITWALK_TT * df['bus_transitwalktime']/600 ) * df['av5']
    df['rail_utility'] = (ASC_RAIL + B_RAIL_TT * (df['rail_transittime'] - df['rail_transitwalktime'])/600 + B_COST * df['rail_fare'] \
                       + B_TRANSITNUMTRANSFERS * df['rail_transitnumtransfers'] + B_TRANSITWALK_TT * df['rail_transitwalktime']/600 ) * df['av6']
    df['intermodal_utility'] = (ASC_PnK_RIDE + B_PnK_RIDE_TT * (df['pnk_transittime'] - df['pnk_transitwalktime'])/600 \
                             + B_COST * df['pnk_fare'] + B_TRANSITNUMTRANSFERS * (df['pnk_transitnumtransfers']+1) + B_TRANSITWALK_TT *df['pnk_transitwalktime']/600 ) * df['av8']
    df['AVPT_utility'] = ASC_AVPT + BETA_AVPT_CAR_TT * (df['detour_factor']/600 * df['AV_time'] + df['wait_time']/600) + BETA_AVPT_COST * df['AV+PT_cost']

    '''exponent each utility equation'''
    df['car_utility'] = np.exp(df['car_utility'])
    df['walk_utility'] = np.exp(df['walk_utility'])
    df['bike_utility'] = np.exp(df['bike_utility'])
    df['taxi_utility'] = np.exp(df['taxi_utility'])
    df['bus_utility'] = df['bus_utility'].apply(lambda value: transit_utility_calc(value,MU_TRANSIT))
    df['rail_utility'] = df['rail_utility'].apply(lambda value: transit_utility_calc(value,MU_TRANSIT))
    df['intermodal_utility'] = df['intermodal_utility'].apply(lambda value: transit_utility_calc(value,MU_TRANSIT))
    df['AVPT_utility'] = np.exp(df['AVPT_utility']*MU_TRANSIT)

    df['exp_sum']  = np.exp(1/MU_TRANSIT * np.log(df['bus_utility'] + df['rail_utility'] + df['intermodal_utility'] + df['AVPT_utility'])) \
        + df['car_utility'] + df['walk_utility'] + df['bike_utility'] + df['taxi_utility']
    df['expsum_wout_AVPT'] = df.apply(lambda value: expsum(value, MU_TRANSIT), axis = 1)
    df['log_sum'] = np.log(df['exp_sum']) * df['expan_fac/10']
    df['log_sum_wout_AVPT'] = np.log(df['expsum_wout_AVPT']) * df['expan_fac/10']
    '''
    calc prob
    '''
    df['sum_transit_utility'] = df['bus_utility'] + df['rail_utility'] + df['intermodal_utility'] + df['AVPT_utility']
    df['prob_car'] = df['car_utility']/df['exp_sum']
    df['prob_walk'] = df['walk_utility']/df['exp_sum']
    df['prob_bike'] = df['bike_utility']/df['exp_sum']
    df['prob_taxi'] = df['taxi_utility']/df['exp_sum']
    df['prob_bus'] = np.exp(1/MU_TRANSIT * np.log(df['sum_transit_utility'])) \
                    /df['exp_sum'] * df['bus_utility']/df['sum_transit_utility']
    df['prob_rail'] = np.exp(1/MU_TRANSIT * np.log(df['sum_transit_utility'])) \
                    /df['exp_sum'] * df['rail_utility']/df['sum_transit_utility']
    df['prob_intermodal'] = np.exp(1/MU_TRANSIT * np.log(df['sum_transit_utility'])) \
                    /df['exp_sum'] * df['intermodal_utility']/df['sum_transit_utility']
    df['prob_AVPT'] = np.exp(1/MU_TRANSIT * np.log(df['sum_transit_utility'])) \
                    /df['exp_sum'] * df['AVPT_utility']/df['sum_transit_utility']

    for _, row in df.iterrows():
        assert np.isclose(row['prob_car'] + row['prob_walk'] + row['prob_bike'] + row['prob_taxi'] + row['prob_bus'] + \
        row['prob_rail'] + row['prob_intermodal'] + row['prob_AVPT'], 1.0)

    df1 = diffinprob(df)

    df['AVPT_choice'] = df['prob_AVPT'] * df['expan_fac/10']

    df['differenceinprob_car'] = (df['prob_car'] - df['prob_car_woutAVPT'])/df['prob_AVPT']
    df['differenceinprob_walk'] = (df['prob_walk'] - df['prob_walk_woutAVPT'])/df['prob_AVPT']
    df['differenceinprob_bike'] = (df['prob_bike'] - df['prob_bike_woutAVPT'])/df['prob_AVPT']
    df['differenceinprob_taxi'] = (df['prob_taxi'] - df['prob_taxi_woutAVPT'])/df['prob_AVPT']
    df['differenceinprob_bus'] = (df['prob_bus'] - df['prob_bus_woutAVPT'])/df['prob_AVPT']
    df['differenceinprob_rail'] = (df['prob_rail'] - df['prob_rail_woutAVPT'])/df['prob_AVPT']
    df['differenceinprob_intermodal'] = (df['prob_intermodal'] - df['prob_intermodal_woutAVPT'])/df['prob_AVPT']

    df_temp1 = df[['expan_fac','expan_fac/10','AVPT_choice','log_sum', 'log_sum_wout_AVPT','prob_AVPT','detour_factor','wait_time']]
    df_temp2 = df1[['differenceinprob_car','differenceinprob_walk','differenceinprob_bike','differenceinprob_taxi','differenceinprob_bus','differenceinprob_rail','differenceinprob_intermodal']]
    df_return = pd.concat([df_temp1, df_temp2], axis = 1)

    return df_return

def set_avpt_demand(step, demand_matrix, ASC_AVPT, fare, df_OD_LOS):
    '''define file paths for each type of trip being modelled, saved in seperate csvs'''
    filepath_cbd = "lib/ModeData/data_olddata_cbd.csv"
    filepath_intrabus = "lib/ModeData/data_olddata_intrazonalbus.csv"
    filepath_intrarail = "lib/ModeData/data_olddata_intrazonalrail.csv"
    filepath_intranew = "lib/ModeData/new_intra_ridership.csv"

    with open(filepath_cbd) as cbd_file:
        reader = csv.reader(cbd_file)
        num_cbd_entries = sum(1 for row in reader) - 1 # subtract 1 for headers

    with open(filepath_intrabus) as intrabus_file:
        reader = csv.reader(intrabus_file)
        num_intrabus_entries = sum(1 for row in reader) - 1 # subtract 1 for headers

    with open(filepath_intrarail) as intrarail_file:
        reader = csv.reader(intrarail_file)
        num_intrarail_entries = sum(1 for row in reader) - 1 # subtract 1 for headers

    with open(filepath_intranew) as intranew_file:
        reader = csv.reader(intranew_file)
        num_intranew_entries = sum(1 for row in reader) - 1 # subtract 1 for headers

    '''generating seperate dataframes for each new type of trips being modelled'''
    df_OD_LOS_1 = df_OD_LOS[df_OD_LOS.index <= 173]
    assert 173 == num_cbd_entries - 1
    assert 243 == num_cbd_entries + num_intrabus_entries - 1
    assert 769 == num_cbd_entries + num_intrabus_entries + num_intrarail_entries - 1
    assert len(df_OD_LOS.index) == num_cbd_entries + num_intrabus_entries + num_intrarail_entries + num_intranew_entries

    df_OD_LOS_2 = df_OD_LOS[(df_OD_LOS.index > 173) & (df_OD_LOS.index <= 243)]
    df_OD_LOS_2 = df_OD_LOS_2.reset_index()

    df_OD_LOS_3 = df_OD_LOS[(df_OD_LOS.index > 243) & (df_OD_LOS.index <= 769)]
    df_OD_LOS_3 = df_OD_LOS_3.reset_index()

    df_OD_LOS_4 = df_OD_LOS[(df_OD_LOS.index > 769)]
    df_OD_LOS_4 = df_OD_LOS_4.reset_index()

    df1 = main_CBD(filepath_cbd, ASC_AVPT, fare, df_OD_LOS_1)
    df2 = main_intrazonal(filepath_intrabus, ASC_AVPT, fare, df_OD_LOS_2)
    df3 = main_intrazonal(filepath_intrarail, ASC_AVPT, fare, df_OD_LOS_3)
    df4 = main_intrazonal(filepath_intranew, ASC_AVPT, fare, df_OD_LOS_4)

    df = df1.append(df2.append(df3.append(df4, ignore_index=True), ignore_index=True), ignore_index=True)

    total_volume = 0.0
    for idx, row in df.iterrows():
        demand_matrix[idx][4] = 1/(step+1) * row['AVPT_choice'] / 3 + step/(step+1) * demand_matrix[idx][4]
        total_volume += demand_matrix[idx][4]

    accum_volume = 0.00
    for idx in range(len(df)):
        accum_volume += demand_matrix[idx][4]
        demand_matrix[idx][5] = accum_volume/total_volume

    logsum_w_AVPT = df['log_sum'].sum()
    logsum_wout_AVPT = df['log_sum_wout_AVPT'].sum()

    df['m_id'] = range(len(df))
    df_diffprob = df.copy(deep=True)

    return demand_matrix, total_volume, logsum_w_AVPT, logsum_wout_AVPT, df_diffprob

def initialize_OD_LOS():
    #initalize decision variables as 0 after each loop completion
    df_OD_LOS = pd.DataFrame(pd.np.empty((1057, 6)) * pd.np.nan,columns=['m_id', 'summed_wait_time', 'summed_detour_factor', 'number_of_occurances','wait_time', 'detour_factor'])
    df_OD_LOS['m_id'] = df_OD_LOS.index
    df_OD_LOS['number_of_occurances'] = 0
    df_OD_LOS['summed_wait_time'] = 0
    df_OD_LOS['summed_detour_factor'] = 0
    df_OD_LOS['wait_time'] = int(INI_WAIT)
    df_OD_LOS['detour_factor'] = int(INI_DETOUR)

    return df_OD_LOS

def set_ridesharing_demand_from_car_and_taxi(demand_matrix, df_OD_LOS):
    '''define file paths for each type of trip being modelled, saved in seperate csvs'''
    filepath_cbd = "lib/ModeData/data_olddata_cbd.csv"
    filepath_intrabus = "lib/ModeData/data_olddata_intrazonalbus.csv"
    filepath_intrarail = "lib/ModeData/data_olddata_intrazonalrail.csv"
    filepath_intranew = "lib/ModeData/new_intra_ridership.csv"

    filepaths = [
        filepath_cbd,
        filepath_intrabus,
        filepath_intrarail,
        filepath_intranew
    ]

    with open(filepath_cbd) as cbd_file:
        reader = csv.reader(cbd_file)
        num_cbd_entries = sum(1 for row in reader) - 1 # subtract 1 for headers

    with open(filepath_intrabus) as intrabus_file:
        reader = csv.reader(intrabus_file)
        num_intrabus_entries = sum(1 for row in reader) - 1 # subtract 1 for headers

    with open(filepath_intrarail) as intrarail_file:
        reader = csv.reader(intrarail_file)
        num_intrarail_entries = sum(1 for row in reader) - 1 # subtract 1 for headers

    with open(filepath_intranew) as intranew_file:
        reader = csv.reader(intranew_file)
        num_intranew_entries = sum(1 for row in reader) - 1 # subtract 1 for headers

    '''generating seperate dataframes for each new type of trips being modelled'''
    df_OD_LOS_1 = df_OD_LOS[df_OD_LOS.index <= 173]
    assert 173 == num_cbd_entries - 1
    assert 243 == num_cbd_entries + num_intrabus_entries - 1
    assert 769 == num_cbd_entries + num_intrabus_entries + num_intrarail_entries - 1
    assert len(df_OD_LOS.index) == num_cbd_entries + num_intrabus_entries + num_intrarail_entries + num_intranew_entries

    df_OD_LOS_2 = df_OD_LOS[(df_OD_LOS.index > 173) & (df_OD_LOS.index <= 243)]
    df_OD_LOS_2 = df_OD_LOS_2.reset_index()

    df_OD_LOS_3 = df_OD_LOS[(df_OD_LOS.index > 243) & (df_OD_LOS.index <= 769)]
    df_OD_LOS_3 = df_OD_LOS_3.reset_index()

    df_OD_LOS_4 = df_OD_LOS[(df_OD_LOS.index > 769)]
    df_OD_LOS_4 = df_OD_LOS_4.reset_index()

    all_sub_dfs = [df_OD_LOS_1, df_OD_LOS_2, df_OD_LOS_3, df_OD_LOS_4]

    # Constant for some reason...
    taxi_wait_time = 0.3

    return_dfs = list()
    for idx, filepath in enumerate(filepaths):
        choice_df = pd.read_csv(filepath)
        df = pd.concat([choice_df, all_sub_dfs[idx]], axis=1)

        print(idx)


        '''DATA PREP'''
        #calculate taxi fare
        df['taxi_fare'] = df['autodis'].apply(taxifare_calc)

        # print(df)
        '''define utility equations'''
        df['car_utility'] = ASC_CAR + B_CAR_TT * df['autotime']/600 + B_CAR_NUM * df['num_cars'] \
            + B_COST * (df['parking_fare']+ df['autodis'] * 5.2416/(1000*76.165) + df['congcharg'] * 11.5)
        df['walk_utility'] = B_WALK_TT * df['walktime']/600
        df['bike_utility'] = ASC_BIKE + B_BIKE_TT * df['biketime']/600
        df['taxi_utility'] = ASC_TAXI + B_TAXI_TT*(df['autotime']/600+taxi_wait_time) + B_TAXI_DIS * df['autodis'] /1000 +  B_COST *  df['taxi_fare'] # assume 3 min waiting time for taxi
        df['bus_utility'] = (ASC_BUS + B_BUS_TT * (df['bus_transittime'] - df['bus_transitwalktime'])/600 + B_COST * df['bus_fare'] \
                             + B_TRANSITNUMTRANSFERS * df['bus_transitnumtransfers'] + B_TRANSITWALK_TT * df['bus_transitwalktime']/600 ) * df['av5']
        df['rail_utility'] = (ASC_RAIL + B_RAIL_TT * (df['rail_transittime'] - df['rail_transitwalktime'])/600 + B_COST * df['rail_fare'] \
                           + B_TRANSITNUMTRANSFERS * df['rail_transitnumtransfers'] + B_TRANSITWALK_TT * df['rail_transitwalktime']/600 ) * df['av6']
        df['intermodal_utility'] = (ASC_PnK_RIDE + B_PnK_RIDE_TT * (df['pnk_transittime'] - df['pnk_transitwalktime'])/600 \
                                 + B_COST * df['pnk_fare'] + B_TRANSITNUMTRANSFERS * (df['pnk_transitnumtransfers']+1) + B_TRANSITWALK_TT *df['pnk_transitwalktime']/600 ) * df['av8']

        '''exponent each utility equation'''
        df['car_utility'] = np.exp(df['car_utility'])
        df['walk_utility'] = np.exp(df['walk_utility'])
        df['bike_utility'] = np.exp(df['bike_utility'])
        df['taxi_utility'] = np.exp(df['taxi_utility'])
        df['bus_utility'] = df['bus_utility'].apply(lambda value: transit_utility_calc(value,MU_TRANSIT))
        df['rail_utility'] = df['rail_utility'].apply(lambda value: transit_utility_calc(value,MU_TRANSIT))
        df['intermodal_utility'] = df['intermodal_utility'].apply(lambda value: transit_utility_calc(value,MU_TRANSIT))

        df['expsum_wout_AVPT'] = df.apply(lambda value: expsum(value, MU_TRANSIT), axis = 1)
        df['log_sum_wout_AVPT'] = np.log(df['expsum_wout_AVPT']) * df['expan_fac/10']

        df['prob_car_woutAVPT'] = df['car_utility']/df['expsum_wout_AVPT']
        df['prob_walk_woutAVPT'] = df['walk_utility']/df['expsum_wout_AVPT']
        df['prob_bike_woutAVPT'] = df['bike_utility']/df['expsum_wout_AVPT']
        df['prob_taxi_woutAVPT'] = df['taxi_utility']/df['expsum_wout_AVPT']
        #pass in identifier with 1 = bus, 2 = rail, 3 = intermodal
        identifier = 0
        df['prob_bus_woutAVPT'] = df.apply(lambda value: calcprob_noAVPT(value, MU_TRANSIT, identifier), axis = 1)
        identifier = 1
        df['prob_rail_woutAVPT'] = df.apply(lambda value: calcprob_noAVPT(value, MU_TRANSIT, identifier), axis = 1)
        identifier = 2
        df['prob_intermodal_woutAVPT'] = df.apply(lambda value: calcprob_noAVPT(value, MU_TRANSIT, identifier), axis = 1)

        df['prob_car'] = df['prob_car_woutAVPT']
        df['prob_walk'] = df['prob_walk_woutAVPT']
        df['prob_bike'] = df['prob_bike_woutAVPT']
        df['prob_taxi'] = df['prob_taxi_woutAVPT']
        df['prob_bus'] = df['prob_bus_woutAVPT']
        df['prob_rail'] = df['prob_rail_woutAVPT']
        df['prob_intermodal'] = df['prob_intermodal_woutAVPT']

        df['exp_sum'] = df['expsum_wout_AVPT']
        df['log_sum'] = df['log_sum_wout_AVPT']

        for sub_idx, row in df.iterrows():
            if not np.isclose(row['prob_car'] + row['prob_walk'] + row['prob_bike'] + row['prob_taxi'] + row['prob_bus'] + \
                row['prob_rail'] + row['prob_intermodal'], 1.0):
                print('{}: Row {}'.format(idx, sub_idx))
            # assert np.isclose(row['prob_car'] + row['prob_walk'] + row['prob_bike'] + row['prob_taxi'] + row['prob_bus'] + \
                # row['prob_rail'] + row['prob_intermodal'], 1.0)

           # Ridehailing trips take all trips away from cars and taxis
        df['prob_AVPT'] = df['prob_car'] + df['prob_taxi'] + df['prob_intermodal']
        df['AVPT_choice'] = df['prob_AVPT'] * df['expan_fac/10']

        df['differenceinprob_car'] = -1.0*df['prob_car']
        df['differenceinprob_walk'] = df['prob_walk'] - df['prob_walk']
        df['differenceinprob_bike'] = df['prob_bike'] - df['prob_bike']
        df['differenceinprob_taxi'] = -1.0*df['prob_taxi']
        df['differenceinprob_bus'] = df['prob_bus'] - df['prob_bus']
        df['differenceinprob_rail'] = df['prob_rail'] - df['prob_rail']
        df['differenceinprob_intermodal'] = -1.0*df['prob_intermodal']

        df_temp1 = df[['expan_fac','expan_fac/10','AVPT_choice','log_sum', 'log_sum_wout_AVPT','prob_AVPT','detour_factor','wait_time']]
        df_temp2 = df[['differenceinprob_car','differenceinprob_walk','differenceinprob_bike','differenceinprob_taxi','differenceinprob_bus','differenceinprob_rail','differenceinprob_intermodal']]
        df_return = pd.concat([df_temp1, df_temp2], axis = 1)
        return_dfs.append(df_return)

    df = return_dfs[0].append(return_dfs[1].append(return_dfs[2].append(return_dfs[3], ignore_index=True), ignore_index=True), ignore_index=True)

    total_volume = 0.0
    for idx, row in df.iterrows():
        demand_matrix[idx][4] = row['AVPT_choice']/20 # 5% of car/taxi/P&K+R demand is shifted to dynamic ridesharing
        total_volume += demand_matrix[idx][4]

    accum_volume = 0.00
    for idx in range(len(df)):
        accum_volume += demand_matrix[idx][4]
        demand_matrix[idx][5] = accum_volume/total_volume

    logsum_w_AVPT = df['log_sum'].sum()
    logsum_wout_AVPT = df['log_sum_wout_AVPT'].sum()

    df['m_id'] = range(len(df))
    df_diffprob = df.copy(deep=True)

    return demand_matrix, total_volume, logsum_w_AVPT, logsum_wout_AVPT, df_diffprob
