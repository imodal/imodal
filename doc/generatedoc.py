import sys
import conf
import pathlib
import subprocess


gallery_paths = [pathlib.Path(path) for path in conf.sphinx_gallery_conf['examples_dirs']]

example_paths = []
for path in gallery_paths:
    example_paths.extend(list(path.rglob("plot_*.py")))

if len(sys.argv) > 1:
    selected_paths = [pathlib.Path(path) for path in sys.argv[1:]]

    for selected_path in selected_paths:
        if selected_path not in example_paths:
            print("Could not find example {}.".format(selected_path))
            exit(-1)

    example_paths = selected_paths


print("Found {} examples files.".format(len(example_paths)))

def make_doc_with_example(example_path):
    subprocess.run(["make", "html", "SPHINXOPTS=-D sphinx_gallery_conf.run_stale_examples=True -D sphinx_gallery_conf.filename_pattern=\'{}\'".format(example_path)])

for example_path in example_paths:
    print("Processing example {}".format(example_path))
    make_doc_with_example(example_path)

