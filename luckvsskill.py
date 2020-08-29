import numpy as np
import matplotlib.pyplot as plt

# Author: Michael Liang

def run_multiple_experiments(m = 10):
	'''
	Runs multiple experiments and aggregates the data.
	Shows histogram of the "Lucked-out-Rate", which is a function of the number of people
	who wouldn't have been selected if skill was the only factor.
	'''
	pop, cut, score = run_experiment()
	scores = [score]

	for _ in range(m-1):
		a, b, c = run_experiment()
		pop = np.vstack((pop, a))
		cut = np.vstack((cut, b))
		scores.append(c)
	
	print("\n------------------------------------------------------")
	print("For {} experiments:".format(m))
	print("------------------------------------------------------")
	print("  Whole Population:")
	print("    Skill: {}".format(np.mean(pop[:,1])))
	print("    Luck:  {}".format(np.mean(pop[:,2])))
	print("  Selected Top Population:")
	print("    Skill: {}".format(np.mean(cut[:,1])))
	print("    Luck:  {}".format(np.mean(cut[:,2])))
	print("------------------------------------------------------\n")
	_, plt.hist(scores, bins = 'auto')
	plt.title("Histogram of Lucked Out Rate")
	plt.show()

def run_experiment(n = 500000, 
	mu = 0.5, 
	sigma = 0.1,
	weight_skill = 0.95,
	threshold = 0.0001,
	tolerance = 0.01,
	verbose = False,
	report = False):
	'''
	Runs a luck and skill experiment.
	Generates 'n' people with a score for both their skill and luck separately pulled
	from a normal (Gaussian) distribution with mean 'mu' and standard deviation 'sigma'.
	Then, an aggregate score for each person is calculated from their skill and luck scores
	using a weight vector according to the 'weight_skill' attribute.
	Aggregates are sorted, and then the highest scores according to 'threshold' are identified.
	Both the values for the whole population and the threshold group are returned.

	Parameters (keyword):
		mu (float) - Mean of the normal distribution
		sigma (float) - Standard deviation of the normal distribution
		weight_skill (float) - The percentage weight of skill towards the aggregate score
		threshold (float) - The percentage of the population accepted at the top
		tolerance (float) - Testing normal distribution according to tolerance
		verbose (bool) - Set verbose to True to print additional details.
		report (bool) - Set report to True to generate a brief report for each experiment.

	Returns:
		all_means (ndarray) - Means of the features for the whole population
		cutoff_means (ndarray) - Means of the features for the cutoff group
		lucked_out_rate (float) - Percentage of selected who wouldn't have been if only skill-based
	'''

	# Generate n people with normal distribution of skill
	skill = np.random.normal(mu+0.4, sigma, n)
	if verbose:
		print("Check skill average close to mu:", np.allclose(mu, np.mean(skill), rtol=tolerance))
		print("Check skill deviation close to sigma:", np.allclose(sigma,np.std(skill, ddof=1), rtol=tolerance))

	# Generate n people with normal distribution of luck
	luck = np.random.normal(mu, sigma, n)
	if verbose:
		print("Check luck average close to mu:", np.allclose(mu, np.mean(luck), rtol=tolerance))
		print("Check luck deviation close to sigma:", np.allclose(sigma,np.std(luck, ddof=1), rtol=tolerance))


	# Aggregate their skill/luck scores with weight_skilling
	aggregate = weight_skill*skill + (1.0-weight_skill)*luck

	# Stack the array for easy sorting and transpose into columns
	all_data = np.stack((aggregate, skill, luck), axis = 0).T

	# Sort the data along the aggregate column
	sorted_data = all_data[all_data[:,0].argsort()]

	# Extract and calculated mu for extracted group
	cutoff = int(threshold*n)
	cutoff_data = sorted_data[-cutoff:,:]
	all_means = np.mean(all_data, axis = 0)
	cutoff_means = np.mean(cutoff_data, axis = 0)

	# Extract comparison with skill isolated
	skill_data = np.sort(all_data[:,1])[-cutoff:]
	intersect_data = np.intersect1d(skill_data, cutoff_data[:,1])
	lucked_out_rate = 1.0 - intersect_data.shape[0]/cutoff_data.shape[0]

	# Print report to console if verbose:
	if report:
		print("\n------------------------------------------------------")
		print("Experiment:")
		print("------------------------------------------------------")
		print("Weight: {}".format(weight_skill))
		print("Threshold: {}".format(threshold))
		print("------------------------------------------------------")
		print("Means (all):    score {}; skill {}; luck {}".format(*np.around(all_means*100, 2)))
		print("Means (cutoff): score {}; skill {}; luck {}".format(*np.around(cutoff_means*100, 2)))
		print("------------------------------------------------------")
		print("Participants selected:", cutoff_data.shape[0])
		print("Participants overlooked:", cutoff_data.shape[0]-intersect_data.shape[0])
		print("------------------------------------------------------\n")

	# Returns results
	return all_means, cutoff_means, lucked_out_rate


if __name__ == "__main__":
	run_multiple_experiments(m = 10)
	