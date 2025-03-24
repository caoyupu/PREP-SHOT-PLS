import pandas as pd
import datetime
import xarray as xr
import numpy as np
from scipy import interpolate
from pyomo.opt import SolverStatus, TerminationCondition


########################################################################
####################### 1. Read data from file #########################
########################################################################

def Readin(sheet_name, filename, month=None, time_length=None):
    if sheet_name == 'static':
        df = pd.read_excel(filename, sheet_name=sheet_name, index_col=0, header=0).unstack()
        return df.to_dict()
    elif sheet_name == 'connect':
        df = pd.read_excel(filename, sheet_name=sheet_name, index_col=None, header=0)
        return df
    elif sheet_name in ['demand', 'storage_upbound', 'storage_downbound']:
        df = pd.read_excel(filename, sheet_name=sheet_name, index_col=[0, 1], header=0)
        df.columns.name = df.index.names[0]
        df.index.names = df.index[0]
        return df.iloc[1:, :].unstack([0, 1]).to_dict()

    elif sheet_name in ['storage_init', 'storage_end']:
        df = pd.read_excel(filename, sheet_name=sheet_name, index_col=0, header=0)
        df.columns.name = df.index.name
        df.index.name = df.index[0]
        return df.iloc[1:, :].unstack().to_dict()

    if sheet_name in ['inflow', 'capacity factor']:
        df = pd.read_excel(filename, sheet_name=sheet_name, index_col=[0, 1, 2], header=0)
        df = df.unstack(level=[0, 1, 2]).dropna().to_dict()
        return df

    if sheet_name in ['technology portfolio', 'hydro_name', 'technology variable cost', 
                      'distance', 'technology fix cost', 'transline fix cost', 'transline variable cost']:
        df = pd.read_excel(filename, sheet_name=sheet_name, index_col=0, header=0)
        df.columns.name = df.index.name
        df.index.name = df.index[0]
        return df.iloc[1:, :].unstack().to_dict()

    if sheet_name in ('ZQ', 'ZV'):
        df = pd.read_excel(filename, sheet_name=sheet_name, header=0)
        return df

    if sheet_name in ['new technology upper bound', 'new technology lower bound', 
                      'distance_max']:
        df = pd.read_excel(filename, sheet_name=sheet_name, index_col=0, header=None)
        return df.to_dict()[1]

    if sheet_name in ['type', 'discount factor']:
        df = pd.read_excel(filename, sheet_name=sheet_name, index_col=0, header=0)
        return df.squeeze().to_dict()


def get_Z_by_Q(name, Q, ZQ):
    ZQ_temp = ZQ[ZQ.name == int(name)]
    f_ZQ = interpolate.interp1d(ZQ_temp.Q, ZQ_temp.Z, fill_value='extrapolate')
    try:
        Z = f_ZQ(Q)
    except:
        print(Q)
    return Z


def get_Z_by_S(name, S, ZV):
    ZV_temp = ZV[ZV.name == int(name)]
    f_ZV = interpolate.interp1d(ZV_temp.V, ZV_temp.Z, fill_value='extrapolate')
    Z = f_ZV(S)
    return Z


def write(file, message):
    print(message)
    with open(file, "a") as f:
        f.write(message)
        f.write("\n")


def run_model_iteration(model, solver, para, iteration_log, error_threshold=0.05, iteration_number=5):
    # 迭代误差小于5%即可停止迭代
    write(iteration_log, 'Starting iteration recorded at %s.' % (datetime.datetime.now()))

    Year = para['year_sets']
    Hour = para['hour_sets']
    Month = para['month_sets']
    zone_hydros = para['zone_hydro']

    # Iterative Head Modeling
    old_waterhead = pd.DataFrame(index=zone_hydros,
                                 columns=pd.MultiIndex.from_product([Year, Month, Hour],
                                                                    names=['year', 'month', 'hour']))
    new_waterhead = old_waterhead.copy(deep=True)

    for s in zone_hydros:
        old_waterhead.loc[s, :] = [para['static']['head', s]]*(len(Hour)*len(Month)*len(Year))
    # Initialization error
    error = 1
    iterations = 1
    errors = []

    idx = pd.IndexSlice
    while error > error_threshold and iterations <= iteration_number:
        alpha = 1/iterations

        for s, h, m, y in model.zone_hydro_hour_month_year_tuples:
            model.head_para[s, h, m, y] = old_waterhead.loc[s, idx[y, m, h]]

        results = solver.solve(model, tee=True)
        if (results.solver.status == SolverStatus.ok) and \
                (results.solver.termination_condition == TerminationCondition.optimal):
            # Do nothing when the solution in optimal and feasible
            pass
        elif (results.solver.termination_condition == TerminationCondition.infeasible):
            # Exit programming when model in infeasible
            write(iteration_log, "Error: Model is in infeasible!")
            print("Error: Model is in infeasible!")
            return 1
        else:
            # Something else is wrong
            write(iteration_log, "Solver Status: %s" % results.solver.status)
            print("Solver Status: ", results.solver.status)
        outflow_v = model.outflow.extract_values()
        storage_v = model.storage_hydro.extract_values()
        # Obtain the new water head after solution
        for stcd in zone_hydros:
            stcd = str(stcd)
            tail = np.array([[[outflow_v[int(stcd), h, m, y] for h in Hour] for m in Month] for y in Year])
            s = np.array([[[storage_v[int(stcd), h, m, y] for h in model.hour_p] for m in Month] for y in Year])
            # interpolation
            tail = get_Z_by_Q(stcd, tail, para['ZQ'])
            s = get_Z_by_S(stcd, s, para['ZV']) 
            fore = (s[:, :, :Hour[-1]] + s[:, :, 1:]) / 2
            H = fore - tail
            H[H <= 0] = 0
            new_waterhead.loc[int(stcd), :] = H.ravel()
        # Calculate iteration error
        new_waterhead[new_waterhead <= 0] = 1
        error = (abs(new_waterhead - old_waterhead) / new_waterhead).mean(axis='columns').mean()
        errors.append(error)
        print(error)
        write(iteration_log, 'water head error: {:.2%}'.format(error))
        # Update water head
        old_waterhead = old_waterhead + alpha * (new_waterhead - old_waterhead)

        iterations += 1
    write(iteration_log, 'Ending iteration recorded at %s.' % (datetime.datetime.now()))
    return 0


def saveresult(model, filename):
    Hour = model.hour
    Month = model.month
    Year = model.year
    Zone_windpv = model.zone_windpv
    Zone_all = model.zone_all
    Zone_hydro = model.zone_hydro
    Hour_p = model.hour_p

    gen = model.gen.extract_values()
    cap_newtech = model.cap_newtech.extract_values()
    cap_newline = model.cap_newline.extract_values()
    cost_var = model.cost_var.extract_values()[None]
    cost_fix = model.cost_fix.extract_values()[None]
    cost_newtech = model.cost_newtech.extract_values()[None]
    cost_newline = model.cost_newline.extract_values()[None]
    income = model.income.extract_values()[None]
    income_windpv = model.income_windpv.extract_values()[None]
    income_hydro = model.income_hydro.extract_values()[None]
    demand_coefficient = model.demand_coefficient.extract_values()

    gen_v = xr.DataArray(data=[[[[gen[h, m, y, z] / 1e6 for h in Hour]
                                for m in Month] for y in Year] for z in Zone_all],
                         dims=['zone_all', 'year', 'month', 'hour'],
                         coords={'month': Month,
                                 'hour': Hour,
                                 'year': Year,
                                 'zone_all': Zone_all},
                         attrs={'unit': 'TWh'})
    cap_newtech_v = xr.DataArray(data=[cap_newtech[z] for z in Zone_windpv],
                                 dims=['zone_windpv'],
                                 coords={'zone_windpv': Zone_windpv},
                                 attrs={'unit': 'MW'})
    demand_coefficient_v = xr.DataArray(data=[[demand_coefficient[m, y] for m in Month] for y in Year],
                                        dims=['year', 'month'],
                                        coords={'year': Year, 'month': Month},
                                        attrs={'unit': 'MW'})

    cost_v = xr.DataArray(data=cost_var + cost_fix + cost_newtech + cost_newline - income)
    cost_var_v = xr.DataArray(data=cost_var)
    cost_fix_v = xr.DataArray(data=cost_fix)
    cost_newtech_v = xr.DataArray(data=cost_newtech)
    cost_newline_v = xr.DataArray(data=cost_newline)
    income_v = xr.DataArray(data=income)
    income_windpv_v = xr.DataArray(data=income_windpv)
    income_hydro_v = xr.DataArray(data=income_hydro)

    # hydro
    zone_hydros = model.zone_hydro
    genflow = model.genflow.extract_values()
    spillflow = model.spillflow.extract_values()
    storage_hydro = model.storage_hydro.extract_values()
    transmission = model.transmission.extract_values()
    genflow_v = xr.DataArray(data=[[[[genflow[s, h, m, y] for h in Hour] for m in Month]
                                        for y in Year] for s in zone_hydros],
                                 dims=['zone_hydro', 'year', 'month', 'hour'],
                                 coords={'zone_hydro': zone_hydros, 'year': Year, 'month': Month, 'hour': Hour},
                                 attrs={'unit': 'm**3s**-1'})
    spillflow_v = xr.DataArray(data=[[[[spillflow[s, h, m, y] for h in Hour] for m in Month]
                                          for y in Year] for s in zone_hydros],
                                   dims=['zone_hydro', 'year', 'month', 'hour'],
                                   coords={'zone_hydro': zone_hydros, 'year': Year, 'month': Month, 'hour': Hour},
                                   attrs={'unit': 'm**3s**-1'})
    storage_hydro_v = xr.DataArray(data=[[[[storage_hydro[s, h, m, y] for h in Hour_p] for m in Month]
                                              for y in Year] for s in zone_hydros],
                                       dims=['zone_hydro', 'year', 'month', 'hour_p'],
                                       coords={'zone_hydro': zone_hydros, 'year': Year, 'month': Month, 'hour_p': Hour_p},
                                       attrs={'unit': '10**8m**3'})

    cap_newline_v = xr.DataArray(data=[[cap_newline[s, z] for z in Zone_windpv] for s in zone_hydros],
                                     dims=['zone_hydro', 'zone_windpv'],
                                     coords={'zone_hydro': zone_hydros, 'zone_windpv': Zone_windpv},
                                     attrs={'unit': 'MW'})

    transmission_v = xr.DataArray(data=[[[[[transmission[h, m, y, s, z] for h in Hour] for m in Month]
                                            for y in Year] for z in Zone_windpv] for s in zone_hydros],
                                      dims=['zone_hydro', 'zone_windpv', 'year', 'month', 'hour'],
                                      coords={'zone_hydro': zone_hydros, 'zone_windpv': Zone_windpv, 'year': Year, 'month': Month, 'hour': Hour},
                                      attrs={'unit': 'MW'})

    ds = xr.Dataset(data_vars={
                                   'gen_v': gen_v,
                                   'cap_newtech_v': cap_newtech_v,
                                   'cost_v': cost_v,
                                   'cost_var_v': cost_var_v,
                                   'cost_fix_v': cost_fix_v,
                                   'genflow_v': genflow_v,
                                   'spillflow_v': spillflow_v,
                                   'cost_newtech_v': cost_newtech_v,
                                   'cost_newline_v': cost_newline_v,
                                   'income_v': income_v,
                                   'income_windpv_v': income_windpv_v, 
                                   'income_hydro_v': income_hydro_v,
                                   'demand_coefficient_v': demand_coefficient_v,
                                   'storage_hydro_v': storage_hydro_v,
                                   'cap_newline_v': cap_newline_v,
                                   'transmission_v': transmission_v
        })

    ds.to_netcdf('%s.nc' % filename)
    return





