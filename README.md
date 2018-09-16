Dist TransE:
1. Using Horovod data parallelism: Modification to existing code are mentioned between 'Horovod added' and 'Horvod end'. For now, as a quick fix, for data parallelism just divided the #epochs(train_times) by #gpus. (This should work as a batch is being randomly sampled from full dataset).  
Main file where changes have been added is 'Config.py'(in config folder).  
To run train phase execute 'example_train_transe.py'.  
#epochs are set using con.set_train_times(100) in 'example_train_transe.py'. Change the argument accordingly. To suppress loss output at each step: con.set_log_on(0)  
