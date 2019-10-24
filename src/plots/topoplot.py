import matplotlib.patches as patches


def topoplot(x, ax, label='', cmap='rainbow'):
	im = ax.imshow(x, origin='lower', vmin=-1, vmax=1, cmap=cmap)

	patch = patches.Circle((33, 33), radius=32.5, transform=ax.transData)
	im.set_clip_path(patch)

	ax.grid(False)

	ax.set_xticks([])
	ax.set_xlabel(label)

	ax.get_yaxis().set_visible(False)
