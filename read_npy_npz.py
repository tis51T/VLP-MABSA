import sys
import os
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np


def process_file(fp: str, shapes: List[Optional[Tuple[int, ...]]]) -> None:
	"""Load a .npy or .npz file and append array shapes to shapes list.

	Individual file errors are ignored so a batch keeps going.
	"""
	lower = fp.lower()
	try:
		if lower.endswith('.npy'):
			arr = np.load(fp, allow_pickle=True)
			shapes.append(getattr(arr, 'shape', None))
		elif lower.endswith('.npz'):
			z = np.load(fp, allow_pickle=True)
			for name in z.files:
				arr = z[name]
				shapes.append(getattr(arr, 'shape', None))
		else:
			obj = np.load(fp, allow_pickle=True)
			shapes.append(getattr(obj, 'shape', None))
	except Exception:
		# ignore files we can't read
		return


def plot_shape_stats(shapes: List[Tuple[int, ...]], out_path: str) -> None:
	clean = [s for s in shapes if s is not None]
	if not clean:
		raise SystemExit('No array shapes found to plot')

	counts = Counter(clean)
	TOP_N = 30
	most = counts.most_common(TOP_N)

	labels = [str(s) for s, _ in most]
	values = [c for _, c in most]

	# prepare scatter of first two dims when available
	xs, ys, sizes = [], [], []
	for s, c in counts.items():
		if isinstance(s, tuple) and len(s) >= 2 and all(isinstance(x, int) for x in s[:2]):
			xs.append(s[0])
			ys.append(s[1])
			sizes.append(c)
		elif isinstance(s, tuple) and len(s) == 1 and isinstance(s[0], int):
			xs.append(s[0])
			ys.append(0)
			sizes.append(c)
		elif isinstance(s, int):
			xs.append(s)
			ys.append(0)
			sizes.append(c)

	try:
		import matplotlib.pyplot as plt
	except Exception as exc:
		raise SystemExit('matplotlib is required to produce the plot: ' + str(exc))

	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

	ax1.barh(range(len(labels)), values, color='C0')
	ax1.set_yticks(range(len(labels)))
	ax1.set_yticklabels(labels)
	ax1.invert_yaxis()
	ax1.set_title('Top shape counts')

	if xs and ys:
		scaled = [max(10, c * 15) for c in sizes]
		ax2.scatter(xs, ys, s=scaled, alpha=0.6)
		ax2.set_xlabel('dim0')
		ax2.set_ylabel('dim1')
		ax2.set_title('Scatter of (dim0, dim1)')
	else:
		ax2.text(0.5, 0.5, 'No 2D shapes to plot', ha='center', va='center')

	plt.tight_layout()
	plt.savefig(out_path)


def main() -> None:
	if len(sys.argv) >= 2:
		path = sys.argv[1]
	else:
		path = input('Path to .npy/.npz file or directory: ').strip()

	if not path:
		raise SystemExit('No path provided')
	if not os.path.exists(path):
		raise SystemExit(f'Path not found: {path}')

	shapes: List[Optional[Tuple[int, ...]]] = []

	if os.path.isdir(path):
		for entry in sorted(os.listdir(path)):
			full = os.path.join(path, entry)
			if os.path.isfile(full) and entry.lower().endswith(('.npy', '.npz')):
				process_file(full, shapes)
	else:
		process_file(path, shapes)

	out = os.path.join(os.getcwd(), 'shape_stats.png')
	plot_shape_stats([s for s in shapes if s is not None], out)
	print(out)


if __name__ == '__main__':
	main()


