# coding: utf-8
from clearml import Task
from clearml.automation import UniformParameterRange, UniformIntegerParameterRange
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna

task = Task.init(project_name="test-hpsearch", task_name="transformers-lm", task_type=Task.TaskTypes.optimizer)

TEMPLATE_TASK_ID = '0ba3f4b6b9454e1c9bcfacb4fb70326c'

optimizer = HyperParameterOptimizer(
    base_task_id=TEMPLATE_TASK_ID,  # This is the experiment we want to optimize
    
    # here we define the hyper-parameters to optimize
    hyper_parameters=[
        UniformIntegerParameterRange('General/epochs', min_value=1, max_value=3, step_size=1),
        UniformIntegerParameterRange('General/batch_size', min_value=30, max_value=40, step_size=2),
        UniformParameterRange('General/dropout', min_value=0, max_value=0.5, step_size=0.05),
        UniformParameterRange('General/lr', min_value=0.00025, max_value=0.1, step_size=0.00025),
    ],

    # setting the objective metric we want to maximize/minimize
    objective_metric_title='ppl',
    objective_metric_series='test',
    objective_metric_sign='min',  # maximize or minimize the objective metric

    # setting optimizer - clearml supports GridSearch, RandomSearch, OptimizerBOHB and OptimizerOptuna
    optimizer_class=OptimizerOptuna,
    
    # Configuring optimization parameters
    execution_queue='default',  # queue to schedule the experiments for execution
    max_number_of_concurrent_tasks=2,  # number of concurrent experiments
    optimization_time_limit=30.,  # set the time limit for the optimization process
    compute_time_limit=30,  # set the compute time limit (sum of execution time on all machines)
    total_max_jobs=2,  # set the maximum number of experiments for the optimization.                         # Converted to total number of iteration for OptimizerBOHB
    min_iteration_per_job=10,  # minimum number of iterations per experiment, till early stopping
    max_iteration_per_job=30,  # maximum number of iterations per experiment
)

optimizer.set_report_period(0.5)  # setting the time gap between two consecutive reports
optimizer.start()  
optimizer.wait()  # wait until process is done
optimizer.stop()  # make sure background optimization stopped

# optimization is completed, print the top performing experiments id
k = 3
top_exp = optimizer.get_top_experiments(top_k=k)
print('Top {} experiments are:'.format(k))
for n, t in enumerate(top_exp, 1):
    print('Rank {}: task id={} |result={}'
          .format(n, t.id, t.get_last_scalar_metrics()))

