from scipy.io import loadmat


def load_topos(path):
	topos = loadmat(path)
	return topos['topos']
