from pyomo.environ import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition
import configparser
import time

from prepshot import load_data
from prepshot import create_model
from prepshot import utils

# default paths
inputpath = '/home/caoyupu/test/PREP-SHOT-PLS/input/'
outputpath = '/home/caoyupu/test/PREP-SHOT-PLS/output/'
logpath = '/home/caoyupu/test/PREP-SHOT-PLS/log/'
logtime = time.strftime("%Y-%m-%d-%H-%M-%S")
logfile = logpath + "main_%s.log" % logtime

utils.write(logfile, "Starting load parameters ...")
start_time = time.time()

# load global parameters
config = configparser.RawConfigParser(inline_comment_prefixes="#")
config.read('/home/caoyupu/test/PREP-SHOT-PLS/global.properties')
basic_para = dict(config.items('global parameters'))
hydro_para = dict(config.items('hydro parameters'))

# global parameters
time_length = int(basic_para['hour'])
month = int(basic_para['month'])
dt = int(basic_para['dt'])
# Fraction of One Year of Modeled Timesteps
weight = (month * time_length * dt) / 8760
weight_month = 1 / float(basic_para['weight_month'])
input_filename = inputpath + basic_para['inputfile']
output_filename = outputpath + basic_para['outputfile']
invcost_wind = int(basic_para['invcost_wind'])
invcost_pv = int(basic_para['invcost_pv'])
invline = int(basic_para['invline'])
lifetime = float(basic_para['lifetime'])

# hydro parameters
error_threshold = float(hydro_para['error_threshold'])
iteration_number = int(hydro_para['iteration_number'])
price = int(hydro_para['price'])

# solver config
solver = SolverFactory(basic_para['solver'], solver_io='python')
solver.options['LogToConsole'] = 0
solver.options['LogFile'] = logfile
solver.options['BarHomogeneous'] = 1
solver.options['NumericFocus'] = 2
solver.options['Method'] = 2

utils.write(logfile, "Set parameter solver to value %s" % basic_para['solver'])
utils.write(logfile, "Set parameter input_filename to value %s" % input_filename)
utils.write(logfile, "Set parameter time_length to value %s" % basic_para['hour'])
utils.write(logfile, "Parameter loading completed, taking %s minutes" % (round((time.time() - start_time) / 60, 2)))

utils.write(logfile, "\n=========================================================")
utils.write(logfile, "Starting load data ...")
start_time = time.time()

# load data
para = load_data(input_filename, month, time_length)
utils.write(logfile, "Data loading completed, taking %s minutes" % (round((time.time() - start_time) / 60, 2)))

para['inputpath'] = inputpath
para['time_length'] = time_length
para['month'] = month
para['dt'] = dt
para['weight'] = weight
para['weight_month'] = weight_month
para['logfile'] = logfile
para['price'] = price
para['invcost_wind'] = invcost_wind
para['invcost_pv'] = invcost_pv
para['invline'] = invline
para['lifetime'] = lifetime

# Validation data
# TODO
utils.write(logfile, "\n==============================================================")
utils.write(logfile, "\n==============================================================")
utils.write(logfile, "Start creating model ...")
model = create_model(para)
utils.write(logfile, "Model creating completed")
utils.write(logfile, "\n=========================================================")

# Solve the model
utils.write(logfile, "Start solving model ...")
start_time = time.time()
state = utils.run_model_iteration(model, solver, para, iteration_log=logfile,
                                        error_threshold=error_threshold,
                                        iteration_number=iteration_number)
utils.write(logfile, "Solving model completed, taking %s minutes" %
                    (round((time.time() - start_time) / 60, 2)))

if state == 0:
    utils.write(logfile, "\n=========================================================")
    utils.write(logfile, "Start writing results ...")
    # Update output file name by scenario settings
    utils.saveresult(model, output_filename)
    utils.write(logfile, "Results written to %s.nc" % output_filename)

utils.write(logfile, "Finish!")
