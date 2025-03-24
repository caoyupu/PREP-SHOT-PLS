from pyomo.environ import *
import numpy as np


def create_model(para):
    """ Create a pyomo ConcreateModel object according to the given data.
    """
    model = ConcreteModel(name='PREP-SHOT')
    model.dual = Suffix(direction=Suffix.IMPORT)

    # Define sets
    model.year = Set(initialize=para['year_sets'], ordered=True, doc='Set of planned timesteps')
    model.zone_windpv = Set(initialize=para['zone_windpv_sets'], ordered=True, doc='Set of zones_windpv')
    model.zone_wind = Set(initialize=para['zone_wind_sets'], ordered=True, doc='Set of zones_wind')
    model.zone_pv = Set(initialize=para['zone_pv_sets'], ordered=True, doc='Set of zones_pv')
    model.tech = Set(initialize=para['tech_sets'], ordered=True, doc='Set of technologies')
    model.hour = Set(initialize=para['hour_sets'], ordered=True, doc='Set of operation timesteps')
    model.hour_p = Set(initialize=[0] + para['hour_sets'], ordered=True,
                       doc='Set of operation timesteps')
    model.month = Set(initialize=para['month_sets'], ordered=True, doc='Set of plnning timesteps')
    model.zone_hydro = Set(initialize=para['zone_hydro'], ordered=True, doc='Set of hydro')
    model.zone_all = Set(initialize=para['zone_all_sets'], ordered=True, doc='Set of all zones')

    model.zone_hydro_zone_windpv_tuples = model.zone_hydro * model.zone_windpv
    model.year_zone_hydro_zone_windpv_tuples = model.year * model.zone_hydro * model.zone_windpv
    model.hour_month_year_zone_hydro_zone_windpv_tuples = model.hour * model.month * model.year * model.zone_hydro * model.zone_windpv

    model.month_year = model.month * model.year
    model.hour_month_year = model.hour * model.month * model.year
    model.year_zone_windpv_tuples = model.year * model.zone_windpv
    model.year_tech_tuples = model.year * model.tech
    model.year_zone_all_tuples = model.year * model.zone_all
    model.year_zone_hydro_tuples = model.year * model.zone_hydro
    model.hour_month_year_zone_windpv_tuples = model.hour * model.month * model.year * model.zone_windpv
    model.hour_month_year_zone_hydro = model.hour * model.month * model.year * model.zone_hydro

    model.hour_month_year_zone_windpv_nondispatchable_tuples = Set(initialize=[(h, m, y, z)
                                                            for h in model.hour for m in model.month for y in model.year
                                                            for z in model.zone_windpv])
    model.hour_month_year_zone_all = Set(initialize=[(h, m, y, z)
                                         for h in model.hour for m in model.month for y in model.year
                                         for z in model.zone_all])

    # Define decision variables and intermediate variables.
    model.cost = Var(within=NonNegativeReals, doc='total cost of system [RMB]')
    model.cost_var = Var(within=NonNegativeReals, doc='Variable O&M costs [RMB]')
    model.cost_newtech = Var(within=NonNegativeReals, doc='Investment costs of new technology [RMB]')
    model.cost_fix = Var(within=NonNegativeReals, doc='Fixed O&M costs [RMB/MW/year]')
    model.cost_newline = Var(within=NonNegativeReals, doc='Investment costs of new transmission lines [RMB]')
    model.income = Var(within=NonNegativeReals, doc='total income of gen_water water and pv and wind [RMB]')
    model.income_windpv = Var(within=NonNegativeReals, doc='total income of pv and wind [RMB]')
    model.income_hydro = Var(within=NonNegativeReals, doc='total income of gen_water water [RMB]')
    model.demand_coefficient = Var(model.month_year, within=NonNegativeReals, doc='demand_coefficient')

    model.cap_newtech = Var(model.zone_windpv,
                            within=NonNegativeReals, doc='Capacity of newbuild technology [MW]')
    model.cap_newline = Var(model.zone_hydro_zone_windpv_tuples,
                            within=NonNegativeReals, doc='Capacity of new transmission lines [MW]')
    model.gen = Var(model.hour_month_year_zone_all, within=NonNegativeReals,
                    doc='Output of each technology in each year, each zone and each time period [MWh]')
    model.transmission = Var(model.hour_month_year_zone_hydro_zone_windpv_tuples,
                             within=NonNegativeReals,
                             doc='The output of the wind and solar power station transmitted to the hydropower station through the connecting line')
    # Define objective funtion: Minimize total costs
    def cost_rule(model):
        return model.income_windpv - model.cost_var - model.cost_newtech - model.cost_fix - model.cost_newline + model.income_hydro

    model.total_cost = Objective(rule=cost_rule, sense=maximize, doc='Maximize the sum of all cost types')

    # Define constraints
    def var_cost_rule(model):
        """
        Return variable Operation and maintenance (O&M) cost of technologies and transmission lines.
        """
        var_OM_tech_cost = sum([para['varcost'][z, y] * model.gen[h, m, y, z] * para['dt'] * para['var_factor'][y]
                                for h, m, y, z in model.hour_month_year_zone_all]) / para['weight'] / para['lifetime']
        var_OM_line_cost = sum([para['varcost_lines'][s, z] * model.transmission[h, m, y, s, z] *
                                para['var_factor'][y]
                                for h, m, y, s, z in model.hour_month_year_zone_hydro_zone_windpv_tuples]) / para['weight'] / para['lifetime']
        return model.cost_var == var_OM_tech_cost + var_OM_line_cost

    def newtech_cost_rule(model):
        """Return total investment cost of new technologies.
        """
        return model.cost_newtech == sum(para['invcost_wind'] * model.cap_newtech[w] for w in model.zone_wind) + sum(para['invcost_pv'] * model.cap_newtech[p] for p in model.zone_pv)

    def newline_cost_rule(model):
        """Return total investment cost of new transmission lines.
        """
        return model.cost_newline == sum(para['invline'] * model.cap_newline[s, z] * \
                                         para['distance'][s, z] * \
                                         para['trans_inv_factor'][y]
                                         for y, s, z in model.year_zone_hydro_zone_windpv_tuples)

    def fix_cost_rule(model):
        """Return fixed O&M cost of technologies and transmission lines.
        """
        fix_cost_hydro = sum(para['fixcost'][z, y] * para['static']['N_max', z] * para['fix_factor'][y]
                            for y, z in model.year_zone_hydro_tuples) / para['lifetime']
        fix_cost_windpv = sum(para['fixcost'][z, y] * model.cap_newtech[z] * para['fix_factor'][y]
                              for y, z in model.year_zone_windpv_tuples) / para['lifetime']
        fix_cost_line = sum(para['fixcost_lines'][s, z] * model.cap_newline[s, z] * para['fix_factor'][y]
                            for y, s, z in model.year_zone_hydro_zone_windpv_tuples) / para['lifetime']
        return model.cost_fix == fix_cost_line + fix_cost_windpv + fix_cost_hydro

    def power_balance_rule(model, h, m, y):
        """Power balance constraints.
        """
        gen_z = sum(model.gen[h, m, y, z_all] for z_all in model.zone_all)
        demand_z = para['demand'][1, m, h] * model.demand_coefficient[m, y]
        return demand_z == gen_z

    model.power_balance_cons = Constraint(model.hour_month_year,
                                          rule=power_balance_rule,
                                          doc='Power balance')

    def gen_line_rule(model, h, m, y, z):
        """gen_line constraints.
        """
        return model.gen[h, m, y, z] <= sum(model.cap_newline[s, z] for s in model.zone_hydro)
    model.gen_line_cons = Constraint(model.hour_month_year_zone_windpv_tuples,
                                         rule=gen_line_rule,
                                         doc='gen_line constraints')

    def transmission_up_bound_rule(model, h, m, y, s, z):
        return model.transmission[h, m, y, s, z] <= model.cap_newline[s, z]
    model.transmission_up_bound_cons = Constraint(model.hour_month_year_zone_hydro_zone_windpv_tuples,
                                                  rule=transmission_up_bound_rule,
                                                  doc='transmission_up_bound constraints')

    def transmission_sum_rule(model, h, m, y, z):
        return sum(model.transmission[h, m, y, s, z] for s in model.zone_hydro) == model.gen[h, m, y, z]
    model.transmission_sum_cons = Constraint(model.hour_month_year_zone_windpv_tuples,
                                             rule=transmission_sum_rule,
                                             doc='transmission_sum constraints')

    def hydro_line_rule(model, h, m, y, s):
        """Hydro_line constraints.
        """
        return model.gen[h, m, y, s] + sum(model.transmission[h, m, y, s, z] for z in model.zone_windpv) <= para['static']['transline', s]
    model.hydro_line_cons = Constraint(model.hour_month_year_zone_hydro,
                                       rule=hydro_line_rule,
                                       doc='Hydro_line constraints')

    def line_up_bound_rule(model, z):
        """Line_up_bound constraints.
        """
        return sum(model.cap_newline[s, z] for s in model.zone_hydro) <= model.cap_newtech[z]
    model.line_up_bound_cons = Constraint(model.zone_windpv,
                                          rule=line_up_bound_rule,
                                          doc='Line_up_bound constraints')

    def gen_up_bound_rule(model, h, m, y, z):
        """Maximum output constraint
        """
        return model.gen[h, m, y, z] <= para['static']['N_max', z]
    model.gen_up_bound_cons = Constraint(model.hour_month_year_zone_hydro,
                                         rule=gen_up_bound_rule,
                                         doc='Maximum output constraint')

    def new_tech_up_bound_rule(model, z):
        if para['newtech_upper'][z] == np.Inf:
            return Constraint.Skip
        else:
            return model.cap_newtech[z] <= para['newtech_upper'][z]

    def new_tech_low_bound_rule(model, z):
        return model.cap_newtech[z] >= para['newtech_lower'][z]

    model.new_tech_up_bound_cons = Constraint(model.zone_windpv,
                                              rule=new_tech_up_bound_rule,
                                              doc='new technology upper bound')
    model.new_tech_low_bound_cons = Constraint(model.zone_windpv,
                                               rule=new_tech_low_bound_rule,
                                               doc='new technology lower bound')

    def renew_gen_rule(model, h, m, y, z):
        """Nondispatchable energy output
        """
        return model.gen[h, m, y, z] <= para['capacity_factor'][z, y, m, h] * model.cap_newtech[z] * para['dt']

    model.renew_gen_cons = Constraint(model.hour_month_year_zone_windpv_nondispatchable_tuples,
                                      rule=renew_gen_rule,
                                      doc='define renewable output')

    def judge_distance_rule(model, s, z):
        if para['distance'][s, z] <= para['distance_max'][s]:
            return Constraint.Skip
        else:
            return model.cap_newline[s, z] == 0

    model.judge_distance_cons = Constraint(model.zone_hydro_zone_windpv_tuples,
                                           rule=judge_distance_rule,
                                           doc='judge_distance')

    model.cost_var_cons = Constraint(rule=var_cost_rule, doc='Variable O&M cost and fuel cost')
    model.newtech_cost_cons = Constraint(rule=newtech_cost_rule, doc='Investment costs of new technology')
    model.newline_cost_cons = Constraint(rule=newline_cost_rule,
                                         doc='Investment costs of new transmission lines')
    model.fix_cost_cons = Constraint(rule=fix_cost_rule,
                                     doc='Fix O&M costs of new transmission lines')

    model = add_hydro(model, para)

    return model

def add_hydro(model, para):

    Hour = para['hour_sets']
    Hour_period = [0] + para['hour_sets']

    model.zone_hydro_hour_month_year_tuples = model.zone_hydro * model.hour * model.month * model.year
    model.zone_hydro_year_tuples = model.zone_hydro * model.year
    model.zone_hydro_month_year_tuples = model.zone_hydro * model.month * model.year
    model.zone_hydro_hour_p_month_year_tuples = model.zone_hydro * model.hour_p * model.month * model.year

    model.head_para = Param(model.zone_hydro_hour_month_year_tuples, mutable=True)

    ############################ hydropower operation start ###########################
    # Hydropower plant variables
    model.naturalinflow = Var(model.zone_hydro_hour_month_year_tuples, within=Reals,
                              doc='natural inflow of reservoir [m3/s]')
    model.inflow = Var(model.zone_hydro_hour_month_year_tuples, within=Reals, doc='inflow of reservoir [m3/s]')
    model.outflow = Var(model.zone_hydro_hour_month_year_tuples, within=NonNegativeReals, doc='outflow of reservoir [m3/s]')
    model.genflow = Var(model.zone_hydro_hour_month_year_tuples, within=NonNegativeReals,
                        doc='generation flow of reservoir [m3/s]')
    model.spillflow = Var(model.zone_hydro_hour_month_year_tuples, within=NonNegativeReals,
                          doc='water spillage flow of reservoir [m3/s]')

    model.storage_hydro = Var(model.zone_hydro_hour_p_month_year_tuples, within=NonNegativeReals,
                              doc='storage of reservoir [10^8 m3]')
    model.output = Var(model.zone_hydro_hour_month_year_tuples, within=NonNegativeReals,
                       doc='output of reservoir [MW]')

    ################################# Hydropower output ###################################
    def natural_inflow_rule(model, s, h, m, y):
        return model.naturalinflow[s, h, m, y] == para['inflow'][s, y, m, h]

    def total_inflow_rule(model, s, h, m, y):
        up_stream_outflow = 0
        for ups, delay in zip(para['connect'][para['connect']['NEXTPOWER_ID'] == s].POWER_ID, para['connect'][para['connect']['NEXTPOWER_ID'] == s].delay):
            delay = int(int(delay) / para['dt'])
            if h - delay >= Hour[0]:
                up_stream_outflow += model.outflow[ups, h - delay, m, y]
            elif h - delay < Hour[0] and (h-delay%Hour[-1]+Hour[-1])%Hour[-1] !=0 and (m-int(delay/Hour[-1]) + para['month_sets'][-1])%para['month_sets'][-1] != 0:
                # It is assumed to dispatch periodically every day to maintain water balance
                # up_stream_outflow += 0
                up_stream_outflow += model.outflow[ups, (h-delay%Hour[-1]+Hour[-1])%Hour[-1], (m-int(delay/Hour[-1]) + para['month_sets'][-1])%para['month_sets'][-1], y]
            elif h - delay < Hour[0] and (h-delay%Hour[-1]+Hour[-1])%Hour[-1] !=0 and (m-int(delay/Hour[-1]) + para['month_sets'][-1])%para['month_sets'][-1] == 0:
                up_stream_outflow += model.outflow[ups, (h-delay%Hour[-1]+Hour[-1])%Hour[-1], para['month_sets'][-1], y]
            elif h - delay < Hour[0] and (m-int(delay/Hour[-1]) + para['month_sets'][-1])%para['month_sets'][-1] != 0 and (h-delay%Hour[-1]+Hour[-1])%Hour[-1] ==0:
                up_stream_outflow += model.outflow[ups, Hour[-1], (m-int(delay/Hour[-1]) + para['month_sets'][-1])%para['month_sets'][-1], y]

        return model.inflow[s, h, m, y] == model.naturalinflow[s, h, m, y] + up_stream_outflow

    def water_balance_rule(model, s, h, m, y):
        return model.storage_hydro[s, h, m, y] == model.storage_hydro[s, h-1, m, y] + (model.inflow[s, h, m, y] -
                model.outflow[s, h, m, y])*3600*para['dt']*1e-8

    def discharge_rule(model, s, h, m, y):
        return model.outflow[s, h, m, y] == model.genflow[s, h, m, y] + model.spillflow[s, h, m, y]

    def outflow_low_bound_rule(model, s, h, m, y):
        return model.outflow[s, h, m, y] >= para['static']['outflow_min', s]

    def outflow_up_bound_rule(model, s, h, m, y):
        return model.outflow[s, h, m, y] <= para['static']['outflow_max', s]
    
    def genflow_low_bound_rule(model, s, h, m, y):
        return model.genflow[s, h, m, y] >= para['static']['GQ_min', s]

    def genflow_up_bound_rule(model, s, h, m, y):
        return model.genflow[s, h, m, y] <= para['static']['GQ_max', s]

    def storage_low_bound_rule(model, s, h, m, y):
        return model.storage_hydro[s, h, m, y] >= para['storagedown'][s, m, h]

    def storage_up_bound_rule(model, s, h, m, y):
        return model.storage_hydro[s, h, m, y] <= para['storageup'][s, m, h]

    def output_low_bound_rule(model, s, h, m, y):
        return model.output[s, h, m, y] >= para['static']['N_min', s]

    def output_up_bound_rule(model, s, h, m, y):
        return model.output[s, h, m, y] <= para['static']['N_max', s]

    def output_calc_rule(model, s, h, m, y):
        return model.output[s, h, m, y] == para['static']['coeff', s] * model.genflow[s, h, m, y] * model.head_para[s, h, m, y] * 1e-3

    model.natural_inflow_cons = Constraint(model.zone_hydro_hour_month_year_tuples,
                                           rule=natural_inflow_rule,
                                           doc='Natural flow')
    model.total_inflow_cons = Constraint(model.zone_hydro_hour_month_year_tuples,
                                         rule=total_inflow_rule,
                                         doc='Hydraulic Connection Constraints')
    model.water_balance_cons = Constraint(model.zone_hydro_hour_month_year_tuples,
                                          rule=water_balance_rule,
                                          doc='Water Balance Constraints')
    model.discharge_cons = Constraint(model.zone_hydro_hour_month_year_tuples,
                                      rule=discharge_rule,
                                      doc='Discharge Constraints')
    model.outflow_low_bound_cons = Constraint(model.zone_hydro_hour_month_year_tuples,
                                              rule=outflow_low_bound_rule,
                                              doc='Discharge lower limits')
    model.outflow_up_bound_cons = Constraint(model.zone_hydro_hour_month_year_tuples,
                                             rule=outflow_up_bound_rule,
                                             doc='Discharge upper limits')
    model.genflow_low_bound_cons = Constraint(model.zone_hydro_hour_month_year_tuples,
                                              rule=genflow_low_bound_rule,
                                              doc='Genflow_low_bound_rule')
    model.genflow_up_bound_cons = Constraint(model.zone_hydro_hour_month_year_tuples,
                                              rule=genflow_up_bound_rule,
                                              doc='Genflow_up_bound_rule')
    model.storage_low_bound_cons = Constraint(model.zone_hydro_hour_month_year_tuples,
                                              rule=storage_low_bound_rule,
                                              doc='Storage lower limits')
    model.storage_up_bound_cons = Constraint(model.zone_hydro_hour_month_year_tuples,
                                             rule=storage_up_bound_rule,
                                             doc='Storage upper limits')
    model.output_low_bound_cons = Constraint(model.zone_hydro_hour_month_year_tuples,
                                             rule=output_low_bound_rule,
                                             doc='Power Output lower limits')
    model.output_up_bound_cons = Constraint(model.zone_hydro_hour_month_year_tuples,
                                            rule=output_up_bound_rule,
                                            doc='Power Output upper limits')
    model.output_calc_cons = Constraint(model.zone_hydro_hour_month_year_tuples,
                                        rule=output_calc_rule,
                                        doc='Power Output Constraints')

    def month_water_balance_rule(model, s, m, y):
        if m != para['month_sets'][-1]:
            return model.storage_hydro[s, Hour_period[0], m+1, y] == (model.storage_hydro[s, Hour_period[-1], m, y] -
                                                                      model.storage_hydro[s, Hour_period[0], m, y]) / para['weight_month'] + model.storage_hydro[s, Hour_period[0], m, y]
        elif m == para['month_sets'][-1]:
            return model.storage_hydro[s, Hour_period[0], para['month_sets'][0], y] == (model.storage_hydro[s, Hour_period[-1], m, y] -
                                                                                        model.storage_hydro[s, Hour_period[0], m, y]) / para['weight_month'] + model.storage_hydro[s, Hour_period[0], m, y]
    model.month_water_balance_cons = Constraint(model.zone_hydro_month_year_tuples,
                                                rule=month_water_balance_rule,
                                                doc='Monthly water balance Constraints')
    
    def year_water_balance_rule(model, s, y):
        if y != para['year_sets'][-1]:
            return model.storage_hydro[s, Hour_period[0], para['month_sets'][0], y + 1] == (model.storage_hydro[s, Hour_period[-1], para['month_sets'][-1], y] - 
                                                                                            model.storage_hydro[s, Hour_period[0], para['month_sets'][-1], y]) / para['weight_month'] + model.storage_hydro[s, Hour_period[0], para['month_sets'][-1], y]
        else:
            return Constraint.Skip
    model.year_water_balance_cons = Constraint(model.zone_hydro_year_tuples, 
                                               rule=year_water_balance_rule, 
                                               doc='Yearly water balance Constraints')

    model.income_cons = Constraint(expr=model.income == sum([model.gen[h, m, y, s]*para['dt']*para['price']
                                                             for h, m, y, s in model.hour_month_year_zone_all])/para['weight'] / para['lifetime'])
    
    model.income_windpv_cons = Constraint(expr=model.income_windpv == sum([model.gen[h, m, y, z] * para['dt'] * para['price']
                                                                          for h, m, y, z in model.hour_month_year_zone_windpv_tuples]) / para['weight'] / para['lifetime'])

    model.income_hydro_cons = Constraint(expr=model.income_hydro == sum([model.gen[h, m, y, s] * para['dt'] * para['price']
                                                                        for h, m, y, s in model.hour_month_year_zone_hydro]) / para['weight'] / para['lifetime'])

    def hydro_output_rule(model, h, m, y, z):
        hydro_output = 0
        for s in model.zone_hydro:
            if para['static']['name', s] == z:
                hydro_output += model.output[s, h, m, y] * para['dt']
        return model.gen[h, m, y, z] == hydro_output

    model.hydro_output_cons = Constraint(model.hour_month_year_zone_hydro,
                                         rule=hydro_output_rule,
                                         doc='define hydropower output')

    return model
