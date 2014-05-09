#import matplotlib.pyplot as plt

import random
from math import sqrt

class Chromosome(object):
	
	def __init__(self, chromosome=None, shuff = False):
		self.chromosome = chromosome
		if shuff == True:
			random.shuffle(chromosome)
		self.fitness = None
	
	def evaluate(self):
		self._calculate_fitness()
	
	def mutate(self):
		self._swap_genes()
	
	def crossover(self, other):
		return self._PMX_crossover(other)    

		
	def _swap_genes(self):
		allel_one = random.randrange(0,len(self.chromosome))
		allel_two = random.randrange(0,len(self.chromosome))
		self.chromosome[allel_one], self.chromosome[allel_two] = self.chromosome[allel_two], self.chromosome[allel_one]
	
	def _calculate_fitness(self):
		self.fitness = 0
		for i in xrange(0, len(self.chromosome)):
			self.fitness += sqrt((self.chromosome[i][1] - self.chromosome[i-1][1])**2 + ((self.chromosome[i][2] - self.chromosome[i-1][2])**2))
		
	def fenotype(self):
		return [vert[0] for vert in self.chromosome]

	def _PMX_crossover(self, other):
		size = min(len(self.chromosome), len(other.chromosome))
		p1, p2 = [0]*size, [0]*size

		# Initialize the position of each indices in the individuals
		for i in xrange(size):
			p1[self.chromosome[i][0]] = i
			p2[other.chromosome[i][0]] = i
		# Choose crossover points
		cxpoint1 = random.randint(0, size)
		cxpoint2 = random.randint(0, size - 1)
		if cxpoint2 >= cxpoint1:
			cxpoint2 += 1
		else: # Swap the two cx points
			cxpoint1, cxpoint2 = cxpoint2, cxpoint1
	
		# Apply crossover between cx points
		for i in xrange(cxpoint1, cxpoint2):
			# Keep track of the selected values
			temp1 = self.chromosome[i]
			temp2 = other.chromosome[i]
		
			# Swap the matched value
			self.chromosome[i], self.chromosome[p1[temp2[0]]] = temp2, temp1
			other.chromosome[i], other.chromosome[p2[temp1[0]]] = temp1, temp2
			# Position bookkeeping
			p1[temp1[0]], p1[temp2[0]] = p1[temp2[0]], p1[temp1[0]]
			p2[temp1[0]], p2[temp2[0]] = p2[temp2[0]], p2[temp1[0]]
					
		return self, other
	
	def __repr__(self):
		return '<%s fenotype=%s ... , fitness=%s>' % \
			   (self.__class__.__name__, 
				 self.fenotype()[:5], self.fitness)

	def __cmp__(self, other):
		return cmp(self.fitness, other.fitness)
	
	def copy(self):
		twin = self.__class__(self.chromosome[:])
		twin.fitness = self.fitness
		return twin

class Population(object):
	
	def __init__(self, data_src = None, pop_size = 40, mutation_prob = 0.00, crossover_prob = 1.0):
		self.pop_size = pop_size
		self.mutation_prob = mutation_prob
		self.crossover_prob = crossover_prob
		
		self.population = [Chromosome(chromosome = data_src, shuff = True) for i in xrange(pop_size)]
		for one in self.population:
			one.evaluate()
	
	def evolve(self):
		self.population.sort()
		self._crossover()
		self._mutate()

	def _mutate(self):
		for chromosome in self.population:
			for gene in chromosome.chromosome:
				if random.random() < self.mutation_prob:
					chromosome.mutate()
		
	def _crossover(self):
		next_population = [self.best.copy()]
		while len(next_population) < self.pop_size:
			mate1 = self._select()
			if random.random() < self.crossover_prob:
				#print "crossing over!!"
				mate2 = self._select()
				offspring = mate1.crossover(mate2)
			else:
				offspring = [mate1.copy()]
			for individual in offspring:
				individual.evaluate()
				next_population.append(individual)
		self.population = next_population[:self.pop_size]
	
	def _select(self):
		"""preferred selection method"""
		return self._tournament()
	
	def _tournament(self, size=8, choosebest=0.80):
		competitors = [random.choice(self.population) for i in range(size)]
		competitors.sort()
		if random.random() < choosebest:
			return competitors[0]
		else:
			return random.choice(competitors[1:])
	
	@property
	def best(self):
		"""individual with best fitness score in population."""
		for one in self.population:
			one.evaluate()
			self.population.sort()
		return self.population[0]
	
	@property    
	def average(self):
		return sum([one.fitness for one in self.population])/float(self.pop_size)

if __name__ == '__main__':
	f = open("../data/tsp_data_29.dat")
	lines = f.readlines()
	f.close()

	vertices = []
	for line in lines:
		line = line.split(' ')
		vertices.append( [int(line[0])-1, float(line[1]), float(line[2])] )

	random.seed(None)

	population = Population(data_src = vertices, pop_size=20, mutation_prob=0.02, crossover_prob=0.35)
	bests = []
	averages = []
	generations = 10000

	for i in xrange(generations):
		population.evolve()
		bests.append(population.best.fitness)
		averages.append(population.average)

	"""plt.xlabel("pokolenie")
	plt.ylabel("dlugosc trasy")
	plt.plot(bests)
	plt.plot(averages)
	optimum = [27603]*generations
	plt.plot(optimum)"""
	print "Best fitness:",bests[-1]
	print population.best