from .utils import Readin
import pandas as pd
import copy

def extract_unique_years(data_dict):
 
    years = []
    for key in data_dict.keys():
        year = key[1]
        years.append(year)
    
    unique_years = list(set(years))
    unique_years.sort()
    
    return unique_years

def load_data(filename, month, time_length):

    Tech_existing = Readin('technology portfolio', filename, month, time_length)  # [MW]
    Distance = Readin('distance', filename, month, time_length)  # [km]
    Distance_max = Readin('distance_max', filename, month, time_length)  # [km]

    DF = Readin('discount factor', filename, month, time_length)
    Cfix = Readin('technology fix cost', filename, month, time_length)  # [RMB/MW/y], O&M cost
    Cvar = Readin('technology variable cost', filename, month, time_length)  # [RMB/MWh]
    Capacity_factor = Readin('capacity factor', filename, month, time_length)
    Demand = Readin('demand', filename, month, time_length)  # [MWh]
    newtech_upper = Readin('new technology upper bound', filename, month, time_length)  # [MW]
    newtech_lower = Readin('new technology lower bound', filename, month, time_length)  # [MW]
    Cfix_lines = Readin('transline fix cost', filename, month, time_length)  # [RMB/MW/km/y]
    Cvar_lines = Readin('transline variable cost', filename, month, time_length)  # [RMB/MWh]

    ZV = Readin('ZV', filename, month, time_length)
    ZQ = Readin('ZQ', filename, month, time_length)
    tech_type = Readin('type', filename)

    df_fixcost_factor = DF.copy()
    df_varcost_factor = DF.copy()
    trans_invcost_factor = DF.copy()

    tech_existing = pd.read_excel(filename, sheet_name='technology portfolio', index_col=0, header=0)
    zone_windpv_sets = list(tech_existing.drop('zone_sw').index) 
    zone_wind_sets = copy.deepcopy(zone_windpv_sets)
    zone_pv_sets = copy.deepcopy(zone_windpv_sets)
    for station in zone_windpv_sets:
        if 'pv' in station:
            zone_wind_sets.remove(station)
        elif 'wind' in station:
            zone_pv_sets.remove(station)
    tech_sets = list(tech_type.keys())
    zone_all = pd.read_excel(filename, sheet_name='zone_all portfolio', index_col=0, header=0)
    zone_all_sets = list(zone_all.drop('zone_all').index)

    # hydropower
    df_static = Readin('static', filename, month, time_length)
    df_inflow = Readin('inflow', filename, month, time_length)
    df_storage_upbound = Readin('storage_upbound', filename, month, time_length)
    df_storage_downbound = Readin('storage_downbound', filename, month, time_length)
    df_storage_init = Readin('storage_init', filename, month, time_length)
    df_storage_end = Readin('storage_end', filename, month, time_length)
    df_connect = Readin('connect', filename, month, time_length)

    year_sets = extract_unique_years(Capacity_factor)
    # year_sets = list(i[1] for i in list(Demand.keys())[0:len(Demand.keys()):timestep])  # [2018, 2019]
    hour_sets = list([i[3] for i in list(df_inflow.keys())[:time_length]])
    month_sets = [list(df_inflow.keys())[i * time_length][2] for i in range(month)]
    stcd_sets = list(set([i[1] for i in df_static.keys()]))

    para = {'technology': Tech_existing,
            'distance': Distance,
            'distance_max': Distance_max,
            'discount': DF,
            'fixcost': Cfix,
            'varcost': Cvar,
            'capacity_factor': Capacity_factor,
            'demand': Demand,
            'newtech_upper': newtech_upper,
            'newtech_lower': newtech_lower,
            'fixcost_lines': Cfix_lines,
            'varcost_lines': Cvar_lines,
            'fix_factor': df_fixcost_factor,
            'var_factor': df_varcost_factor,
            'trans_inv_factor': trans_invcost_factor,
            'type': tech_type,
            'inflow': df_inflow,
            'storageup': df_storage_upbound,
            'storagedown': df_storage_downbound,
            'storageinit': df_storage_init,
            'storageend': df_storage_end,
            'static': df_static,
            'connect': df_connect,
            'ZV': ZV, 
            'ZQ': ZQ,
            'year_sets': year_sets, 
            'hour_sets': hour_sets,
            'month_sets': month_sets,
            'zone_windpv_sets': zone_windpv_sets, 
            'zone_wind_sets': zone_wind_sets, 
            'zone_pv_sets': zone_pv_sets,
            'tech_sets': tech_sets,
            'zone_hydro': stcd_sets,
            'zone_all_sets': zone_all_sets
            }

    return para



