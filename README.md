Dist TransE:
1. Using Horovod data parallelism: Modification to existing code are mentioned between 'Horovod added' and 'Horvod end'. For now, as a quick fix, for data parallelism just divided the #epochs(train_times) by #gpus. (This should work as in batch we are randomly sampling a batch from full dataset.)
