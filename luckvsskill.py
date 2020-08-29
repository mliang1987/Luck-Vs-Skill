import numpy as np

def run_experiment(n = 100000, 
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
	'''

	# Generate n people with normal distribution of skill
	skill = np.random.normal(mu, sigma, n)
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
		print("------------------------------------------------------\n")

	# Returns results
	return all_means, cutoff_means


if __name__ == "__main__":
	run_experiment(verbose = True, report = True)
