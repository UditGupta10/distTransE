#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import time
import datetime
import ctypes
import json
import pickle
import horovod.tensorflow as hvd

hvd.init()
class Config(object):

	def __init__(self):
		base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../release/Base.so'))
		self.lib = ctypes.cdll.LoadLibrary(base_file)
		self.lib.sampling.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
		self.lib.getHeadBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.getTailBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.testHead.argtypes = [ctypes.c_void_p]
		self.lib.testTail.argtypes = [ctypes.c_void_p]
		self.lib.getTestBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.getValidBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.getBestThreshold.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
		self.lib.test_triple_classification.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
		self.test_flag = False
		self.in_path = None
		self.out_path = None
		self.bern = 0
		self.hidden_size = 100
		self.ent_size = self.hidden_size
		self.rel_size = self.hidden_size
		self.train_times = 0
		self.margin = 1.0
		self.nbatches = 100
		self.negative_ent = 1
		self.negative_rel = 0
		self.workThreads = 1
		self.alpha = 0.001
		self.lmbda = 0.000
		self.log_on = 1
		self.exportName = None
		self.importName = None
		self.export_steps = 0
		self.opt_method = "SGD"
		self.optimizer = None
		self.test_link_prediction = False
		self.test_triple_classification = False
		########
		self.sync_after = 1
		self.plot_train_loss = False
		########/////
	def init(self):
		self.trainModel = None
		self.track_loss = []
		if self.in_path != None:
			self.lib.setInPath(ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2))
			self.lib.setBern(self.bern)
			self.lib.setWorkThreads(self.workThreads)
			self.lib.randReset(0)
			self.lib.importTrainFiles()
			self.relTotal = self.lib.getRelationTotal()
			self.entTotal = self.lib.getEntityTotal()
			self.trainTotal = self.lib.getTrainTotal()
			self.testTotal = self.lib.getTestTotal()
			self.validTotal = self.lib.getValidTotal()
			self.batch_size = int(self.lib.getTrainTotal() / (self.nbatches * hvd.size() * 1.0))
			###########
			self.allreduce_batch_size = self.batch_size * self.sync_after
			self.allreduce_nbatches = int(self.nbatches / (self.sync_after * 1.0))
			############/////
			self.batch_seq_size = self.batch_size * (1 + self.negative_ent + self.negative_rel)
			####################
			self.batch_h = np.zeros(self.allreduce_batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
			self.batch_t = np.zeros(self.allreduce_batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
			self.batch_r = np.zeros(self.allreduce_batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
			self.batch_y = np.zeros(self.allreduce_batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.float32)
			###################//////
			self.batch_h_addr = self.batch_h.__array_interface__['data'][0]
			self.batch_t_addr = self.batch_t.__array_interface__['data'][0]
			self.batch_r_addr = self.batch_r.__array_interface__['data'][0]
			self.batch_y_addr = self.batch_y.__array_interface__['data'][0]
		if self.test_link_prediction:
			self.lib.importTestFiles()
			self.lib.importTypeFiles()
			self.test_h = np.zeros(self.lib.getEntityTotal(), dtype = np.int64)
			self.test_t = np.zeros(self.lib.getEntityTotal(), dtype = np.int64)
			self.test_r = np.zeros(self.lib.getEntityTotal(), dtype = np.int64)
			self.test_h_addr = self.test_h.__array_interface__['data'][0]
			self.test_t_addr = self.test_t.__array_interface__['data'][0]
			self.test_r_addr = self.test_r.__array_interface__['data'][0]
		if self.test_triple_classification:
			self.lib.importTestFiles()
			self.lib.importTypeFiles()

			self.test_pos_h = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
			self.test_pos_t = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
			self.test_pos_r = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
			self.test_neg_h = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
			self.test_neg_t = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
			self.test_neg_r = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
			self.test_pos_h_addr = self.test_pos_h.__array_interface__['data'][0]
			self.test_pos_t_addr = self.test_pos_t.__array_interface__['data'][0]
			self.test_pos_r_addr = self.test_pos_r.__array_interface__['data'][0]
			self.test_neg_h_addr = self.test_neg_h.__array_interface__['data'][0]
			self.test_neg_t_addr = self.test_neg_t.__array_interface__['data'][0]
			self.test_neg_r_addr = self.test_neg_r.__array_interface__['data'][0]

			self.valid_pos_h = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
			self.valid_pos_t = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
			self.valid_pos_r = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
			self.valid_neg_h = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
			self.valid_neg_t = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
			self.valid_neg_r = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
			self.valid_pos_h_addr = self.valid_pos_h.__array_interface__['data'][0]
			self.valid_pos_t_addr = self.valid_pos_t.__array_interface__['data'][0]
			self.valid_pos_r_addr = self.valid_pos_r.__array_interface__['data'][0]
			self.valid_neg_h_addr = self.valid_neg_h.__array_interface__['data'][0]
			self.valid_neg_t_addr = self.valid_neg_t.__array_interface__['data'][0]
			self.valid_neg_r_addr = self.valid_neg_r.__array_interface__['data'][0]

	def get_ent_total(self):
		return self.entTotal

	def get_rel_total(self):
		return self.relTotal

	def set_sync_after(self, x):
		self.sync_after = x

	def set_plot_train_loss(self, flag):
		self.plot_train_loss = flag

	def set_lmbda(self, lmbda):
		self.lmbda = lmbda

	def set_optimizer(self, optimizer):
		self.optimizer = optimizer

	def set_opt_method(self, method):
		self.opt_method = method

	def set_test_link_prediction(self, flag):
		self.test_link_prediction = flag

	def set_test_triple_classification(self, flag):
		self.test_triple_classification = flag

	def set_log_on(self, flag):
		self.log_on = flag

	def set_alpha(self, alpha):
		self.alpha = alpha

	def set_in_path(self, path):
		self.in_path = path#[:-1] + "-" + str(hvd.rank()) + "/"

	def set_out_files(self, path):
		self.out_path = path

	def set_bern(self, bern):
		self.bern = bern

	def set_dimension(self, dim):
		self.hidden_size = dim
		self.ent_size = dim
		self.rel_size = dim

	def set_ent_dimension(self, dim):
		self.ent_size = dim

	def set_rel_dimension(self, dim):
		self.rel_size = dim

	#Horovod added: Reduced the number of epochs
	def set_train_times(self, times):
		self.train_times = times

	def set_nbatches(self, nbatches):
		self.nbatches = nbatches // hvd.size()

	def set_margin(self, margin):
		self.margin = margin

	def set_work_threads(self, threads):
		self.workThreads = threads

	def set_ent_neg_rate(self, rate):
		self.negative_ent = rate

	def set_rel_neg_rate(self, rate):
		self.negative_rel = rate

	def set_import_files(self, path):
		self.importName = path

	def set_export_files(self, path, steps = 0):
		self.exportName = path
		self.export_steps = steps

	def set_export_steps(self, steps):
		self.export_steps = steps

	########################	
	def sampling(self):
		self.lib.sampling(self.batch_h_addr, self.batch_t_addr, self.batch_r_addr, self.batch_y_addr, self.allreduce_batch_size, self.negative_ent, self.negative_rel)
	#########################//////////////

	def save_tensorflow(self):
		with self.graph.as_default():
			with self.sess.as_default():
				self.saver.save(self.sess, self.exportName)

	def restore_tensorflow(self):
		with self.graph.as_default():
			with self.sess.as_default():
				self.saver.restore(self.sess, self.importName)


	def export_variables(self, path = None):
		with self.graph.as_default():
			with self.sess.as_default():
				if path == None:
					self.saver.save(self.sess, self.exportName)
				else:
					self.saver.save(self.sess, path)

	def import_variables(self, path = None):
		with self.graph.as_default():
			with self.sess.as_default():
				if path == None:
					self.saver.restore(self.sess, self.importName)
				else:
					self.saver.restore(self.sess, path)

	def get_parameter_lists(self):
		return self.trainModel.parameter_lists

	def get_parameters_by_name(self, var_name):
		with self.graph.as_default():
			with self.sess.as_default():
				if var_name in self.trainModel.parameter_lists:
					return self.sess.run(self.trainModel.parameter_lists[var_name])
				else:
					return None

	def get_parameters(self, mode = "numpy"):
		res = {}
		lists = self.get_parameter_lists()
		for var_name in lists:
			if mode == "numpy":
				res[var_name] = self.get_parameters_by_name(var_name)
			else:
				res[var_name] = self.get_parameters_by_name(var_name).tolist()
		return res

	def save_parameters(self, path = None):
		if path == None:
			path = self.out_path
		f = open(path, "w")
		f.write(json.dumps(self.get_parameters("list")))
		f.close()

	def set_parameters_by_name(self, var_name, tensor):
		with self.graph.as_default():
			with self.sess.as_default():
				if var_name in self.trainModel.parameter_lists:
					self.trainModel.parameter_lists[var_name].assign(tensor).eval()

	def set_parameters(self, lists):
		for i in lists:
			self.set_parameters_by_name(i, lists[i])

	def set_model(self, model):
		self.model = model
		self.graph = tf.Graph()
		with self.graph.as_default():
			#Horovod added: Normal workflow
			config1 = tf.ConfigProto(log_device_placement=False)
			config1.gpu_options.allow_growth = True
			config1.gpu_options.visible_device_list = str(hvd.local_rank())
			self.sess = tf.Session(config=config1)
			#Horovod end
			with self.sess.as_default():
				initializer = tf.contrib.layers.xavier_initializer(uniform = True)
				with tf.variable_scope("model", reuse=None, initializer = initializer):
					self.trainModel = self.model(config = self)
					#Horovod added: Vary the learning rate, dist optimizer
					if self.optimizer != None:
						pass
					elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
						self.optimizer = tf.train.AdagradOptimizer(learning_rate = self.alpha * hvd.size(), initial_accumulator_value=1e-20)
					elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
						self.optimizer = tf.train.AdadeltaOptimizer(self.alpha * hvd.size())
					elif self.opt_method == "Adam" or self.opt_method == "adam":
						self.optimizer = tf.train.AdamOptimizer(self.alpha * hvd.size())
					else:
						self.optimizer = tf.train.GradientDescentOptimizer(self.alpha * hvd.size() * self.sync_after)

					################################################################
					# Fetch a list of our network's trainable parameters.
					self.trainable_vars = tf.trainable_variables()
					#print("Shape of trainable vars: {}".format(np.array(self.trainable_vars)))

					# Create variables to store accumulated gradients
					self.accumulators = [
					    tf.Variable(
					        tf.zeros_like(tv.initialized_value()),
					        trainable=False
					    ) for tv in self.trainable_vars
					]
					#print("Shape of accumulators: {}".format(np.array(self.accumulators)))

					# Create a variable for counting the number of accumulations
					self.accumulation_counter = tf.Variable(0.0, trainable=False)

					# Compute gradients; grad_pairs contains (gradient, variable) pairs
					self.grad_pairs = self.optimizer.compute_gradients(self.trainModel.loss, self.trainable_vars)
					# print("Shape of grad_pairs: {}".format(np.array(self.grad_pairs)))
					# for g, v in self.grad_pairs:
					# 	print("Shape of grad: {}".format(np.array(g)))


					# Create operations which add a variable's gradient to its accumulator.
					self.accumulate_ops = [
					    accumulator.assign_add(
					        grad
					    ) for (accumulator, (grad, var)) in zip(self.accumulators, self.grad_pairs) #if grad is not None
					]

					# The final accumulation operation is to increment the counter
					self.accumulate_ops.append(self.accumulation_counter.assign_add(1.0))

					# Update trainable variables by applying the accumulated gradients
					# divided by the counter. Note: apply_gradients takes in a list of 
					# (grad, var) pairs
					# self.apply_step = self.optimizer.apply_gradients(
					#     [(accumulator / self.accumulation_counter, var) \
					#         for (accumulator, (grad, var)) in zip(self.accumulators, self.grad_pairs)]
					# )

					# Accumulators must be zeroed once the accumulated gradient is applied.
					self.zero_ops = [
					    accumulator.assign(
					        tf.zeros_like(tv)
					    ) for (accumulator, tv) in zip(self.accumulators, self.trainable_vars)
					]

					# Add one last op for zeroing the counter
					self.zero_ops.append(self.accumulation_counter.assign(0.0))
					################################################################///////////



					# self.dist_optimizer = hvd.DistributedOptimizer(self.optimizer)
					# self.train_op = self.dist_optimizer.minimize(self.trainModel.loss)
					#Horovod end
				self.barrier = hvd.allreduce(tf.random_normal(shape=[1]))
				if(hvd.rank() == 0):
					self.saver = tf.train.Saver()
					# self.logSummary = tf.summary.scalar('Train_loss', self.trainModel.loss)
					# self.train_writer = tf.summary.FileWriter('./train', self.sess.graph)
				self.sess.run(tf.global_variables_initializer())
				
				#Horovod added: Normal workflow
				self.sess.run(hvd.broadcast_global_variables(0))
				#Horovod end

	def train_step(self, batch_h, batch_t, batch_r, batch_y, counter):
		self.sess.run(self.zero_ops)
		allreduce_loss = 0.0
		for i in range(self.sync_after):
			feed_dict = {
				self.trainModel.batch_h: np.append(batch_h[i*self.batch_size : (i+1)*self.batch_size], batch_h[self.allreduce_batch_size + i*self.batch_size : self.allreduce_batch_size + (i+1)*self.batch_size]),
				self.trainModel.batch_t: np.append(batch_t[i*self.batch_size : (i+1)*self.batch_size], batch_t[self.allreduce_batch_size + i*self.batch_size : self.allreduce_batch_size + (i+1)*self.batch_size]),
				self.trainModel.batch_r: np.append(batch_r[i*self.batch_size : (i+1)*self.batch_size], batch_r[self.allreduce_batch_size + i*self.batch_size : self.allreduce_batch_size + (i+1)*self.batch_size]),
				self.trainModel.batch_y: np.append(batch_y[i*self.batch_size : (i+1)*self.batch_size], batch_y[self.allreduce_batch_size + i*self.batch_size : self.allreduce_batch_size + (i+1)*self.batch_size])
			}
			_, c = self.sess.run([self.accumulate_ops, self.trainModel.loss], feed_dict = feed_dict)
			allreduce_loss += c
		self.track_loss.append((counter, allreduce_loss))
		self.sess.run(self.barrier)
		st1 = time.time()

		if hvd.size() > 1:
			averaged_gradients = []
			with tf.name_scope("Allreduce"):
				for (accumulator, (grad, var)) in zip(self.accumulators, self.grad_pairs):
					#if tf.equal(accumulator, 0)
					if accumulator is not None:
						avg_grad = hvd.allreduce(accumulator / self.accumulation_counter)
						averaged_gradients.append((avg_grad, var))
					else:
						averaged_gradients.append((None, var))
		else:
			averaged_gradients = []
			with tf.name_scope("Allreduce"):
				for (accumulator, (grad, var)) in zip(self.accumulators, self.grad_pairs):
					#print("Shape of accumulator: {}".format(np.array(accumulator)))
					if accumulator is not None:						
						avg_grad = accumulator / self.accumulation_counter
						averaged_gradients.append((avg_grad, var))
					else:
						averaged_gradients.append((None, var))


		if(counter%200 == 0):
			print('Averaging gradients for 200 batches took: {} secs'.format(time.time() - st1))

		st2 = time.time()
		self.sess.run(self.optimizer.apply_gradients(averaged_gradients))

		if(counter%200 == 0):
			print('Applying gradients for 200 batches took: {} secs'.format(time.time() - st2))

		return allreduce_loss











	def test_step(self, test_h, test_t, test_r):
		feed_dict = {
			self.trainModel.predict_h: test_h,
			self.trainModel.predict_t: test_t,
			self.trainModel.predict_r: test_r,
		}
		predict = self.sess.run(self.trainModel.predict, feed_dict)
		return predict

	def run(self):
		with self.graph.as_default():
			with self.sess.as_default():
				if self.importName != None:
					self.restore_tensorflow()
				if(hvd.rank() == 0):
					start = time.time()
				counter = 0
				for times in range(self.train_times):
					res = 0.0
					for batch in range(self.allreduce_nbatches):
						counter += 1
						self.sampling()
						res += self.train_step(self.batch_h, self.batch_t, self.batch_r, self.batch_y, counter)
					if(hvd.rank() == 0):
						print("Time taken: {0}".format(time.time() - start))
					if self.log_on:
						#print(times)
						print("Epoch: {} , nbatches = {}".format(times, self.allreduce_nbatches))
						print(res/self.allreduce_nbatches)
					if self.exportName != None and (self.export_steps!=0 and times % self.export_steps == 0):
						self.save_tensorflow()
				if(hvd.rank() == 0):
					print("Time taken: {0}".format(time.time() - start))
					if self.plot_train_loss:
						with open('./plot_loss/acc_1024/' + str(hvd.size()) + 'p-' + str(self.sync_after) + 'sa' + '.pkl', 'wb') as f:
							pickle.dump(self.track_loss, f)
					# plt.plot(*zip(*self.track_loss))
					# plt.show()
				if self.exportName != None:
					if(hvd.rank() == 0):
						self.save_tensorflow()
				if self.out_path != None:
					if(hvd.rank() == 0):
						self.save_parameters(self.out_path)

	def test(self):
		with self.graph.as_default():
			with self.sess.as_default():
				if self.importName != None:
					self.restore_tensorflow()
				if self.test_link_prediction:
					total = self.lib.getTestTotal()
					x = 0.0
					for times in range(total):
						self.lib.getHeadBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
						res = self.test_step(self.test_h, self.test_t, self.test_r)
						self.lib.testHead(res.__array_interface__['data'][0])

						self.lib.getTailBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
						res = self.test_step(self.test_h, self.test_t, self.test_r)
						self.lib.testTail(res.__array_interface__['data'][0])
						if self.log_on and hvd.rank() == 0:
							if(times == int(total*x)):
								print('Testing progress: ' + str(x*100) + '%')
								x+=0.1
					if(hvd.rank() == 0):
						self.lib.test_link_prediction()
				if (self.test_triple_classification and hvd.rank() == 0):
					self.lib.getValidBatch(self.valid_pos_h_addr, self.valid_pos_t_addr, self.valid_pos_r_addr, self.valid_neg_h_addr, self.valid_neg_t_addr, self.valid_neg_r_addr)
					res_pos = self.test_step(self.valid_pos_h, self.valid_pos_t, self.valid_pos_r)
					res_neg = self.test_step(self.valid_neg_h, self.valid_neg_t, self.valid_neg_r)
					self.lib.getBestThreshold(res_pos.__array_interface__['data'][0], res_neg.__array_interface__['data'][0])

					self.lib.getTestBatch(self.test_pos_h_addr, self.test_pos_t_addr, self.test_pos_r_addr, self.test_neg_h_addr, self.test_neg_t_addr, self.test_neg_r_addr)

					res_pos = self.test_step(self.test_pos_h, self.test_pos_t, self.test_pos_r)
					res_neg = self.test_step(self.test_neg_h, self.test_neg_t, self.test_neg_r)
					self.lib.test_triple_classification(res_pos.__array_interface__['data'][0], res_neg.__array_interface__['data'][0])



