import numpy as np
import pygad.nn
from tensorflow import keras
from tensorflow.keras.layers import Dense,Dropout
from keras.datasets import mnist
import pygad
import pygad.kerasga
import tensorflow.keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
import tensorflow as tf
import pandas
import tensorflow_datasets as tfds




(ds_train,ds_test),ds_info=tfds.load(
	'mnist',
	split=['train','test'],
	shuffle_files=True,
	as_supervised=True,
	with_info=True,
)








def normalization(image,label):
	return tf.cast(image, tf.float32)/255., label
	
ds_train = ds_train.map(normalization,num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train=ds_train.cache()
ds_train=ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train=ds_train.batch(128)
ds_train=ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_train_numpy=tfds.as_numpy(ds_train)




ds_test = ds_test.map(normalization,num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test=ds_test.batch(128)
ds_test=ds_test.cache()
ds_test=ds_test.prefetch(tf.data.experimental.AUTOTUNE)

ds_test_numpy=tfds.as_numpy(ds_test)


def fitness_function(solution,sol_idx):
	global data_inputs,data_outputs,keras_ga, model
	
	model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model,
									weights_vector=solution)
	model.set_weights(weights=model_weights_matrix)
	
	predictions=model.predict(data_inputs)
	
	cce=tensorflow.keras.losses.CategoricalCrossentropy()
	solutions_fitness=1.0/ (cce(data_outputs,predictions).numpy() + 0.00000001)
	
	return solution_fitness

def parent_selection_function(fitness,num_parents,ga_instance):
	
	
	
	parents=numpy.empty((num_parents,ga_instance.population.shape[1]))
	parents_indices=[]
	
	
	
	for parent_num in range(num_parents):
		rand_indices=numpy.random.randint(low=0.0,high=len(fitness))
		selected_parent_idx=numpy.where(max(fitness))
		parent_indices.append(selected_parent_idx)
		parents[parent_num,:]=ga_instance.population[rand_indices[selecred_parent_idx],:].copy()
		
	return parents,parents_indices
	


	
def mutation_function(offspring,ga_instance):
	for chromosome_idx in range(offspring.shape[0]):
		for genes_idx in range(offspring.shape[1]):
			if np.random.uniform(0,1)< (mutation_percent_genes/100):
				offspring[chromosome_idx][genes_idx]=offspring[chromosome_idx][genes_idx]^1
		#random_gene_idx=numpy.random.choice(range(offspring.shape[0]))
		
		
		#offspring[chromosome_idx,random_gene_idx] +=numpy.random.random()
	return offspring

	
	
def callback_generation(ga_instance):
	print("Generation={generation}".format(generation=ga_instance.generation_completed))
	print("Fitness={fitness}".format(fitness=ga_instance.best_solution(ga_instance.last_generation_fitness[1])))	



data_inputs=ds_train_numpy

data_outputs=ds_test_numpy




model=tensorflow.keras.Sequential([
	tensorflow.keras.layers.Flatten(),
	tensorflow.keras.layers.Dense(576,input_dim=5760,activation="relu"),
	tensorflow.keras.layers.Dense(10,activation="relu"),
	tensorflow.keras.layers.Dense(20,activation="relu"),
	tensorflow.keras.layers.Dense(10,activation="softmax",kernel_regularizer=tensorflow.keras.regularizers.l2(0.1))
	
	])

opt=SGD(lr=0.1,momentum=0.6)
'''
model.compile(
	optimizer='SGD',
	loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
	metrics=['accuracy']
	)
	

model.fit(
	ds_train,
	epochs=10,
	validation_data=ds_test,
	verbose=0)



model.summary
'''
keras_ga=pygad.kerasga.KerasGA(model=model,
				num_solutions=10)



#run pygad
num_generations=20
num_parents_mating=5
initial_population=20
mutation_percent_genes=5
parent_selection_type="sss"
#crossover_type="single_point"
#mutation_type="random"
crossover_probability=0.6

ga_main = pygad.GA(num_generations=num_generations,
			num_parents_mating=num_parents_mating,
			initial_population=initial_population,
			num_genes=768,
			fitness_func=fitness_function,
			mutation_percent_genes=mutation_percent_genes,
			parent_selection_type=parent_selection_function,
			crossover_type="uniform",
			mutation_type=mutation_function,
			on_generation=callback_generation)
			
ga_main.run()
#ga_isntance.plot_fitness()
#fitness/generation
ga_main.plot_fitness(title="Fitness per generation")





