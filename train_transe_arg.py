import config
import models
import tensorflow as tf
import numpy as np
import sys
# import horovod.tensorflow as hvd
# tf.logging.set_verbosity(tf.logging.INFO)
# hvd.init()

# if(hvd.rank() == 0):
# 	start = time.time()
train_times = int(sys.argv[1])
nbatches = int(sys.argv[2])
con = config.Config()
#Input training files from benchmarks/FB15K/ folder.
con.set_in_path("./benchmarks/FB15K/")

#Start testing
con.set_test_triple_classification(True)
con.set_test_link_prediction(True)
#End testing

con.set_log_on(0)
con.set_work_threads(8)
con.set_train_times(train_times)
con.set_nbatches(nbatches)
con.set_alpha(0.001)
con.set_margin(1.0)
con.set_bern(0)
con.set_dimension(100)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("SGD")

#Models will be exported via tf.Saver() automatically.
con.set_export_files("./res/model.vec.tf", 0)
#Model parameters will be exported to json files automatically.
con.set_out_files("./res/embedding.vec.json")
#Initialize experimental settings.
con.init()
#Set the knowledge embedding model
con.set_model(models.TransE)
#Train the model.
con.run()
#To test models after training needs "set_test_flag(True)".
con.test()
# if(hvd.rank() == 0):
# 	print("Time taken: {0}".format(time.time() - start))

