#!/usr/bin/env python

"""
This script will tell git to ignore prompt numbers and cell output during commit, diff, status etc.
The notebooks themselves are not changed, only the representation for git.

Inspired by the this blog post: http://pascalbugnion.net/blog/ipython-notebooks-and-git.html.

To tell git to not ignore the outputs and prompts of a notebook,
open the notebook's metadata (Edit > Edit Notebook Metadata). A
panel should open containing the lines:

    {
        "name" : "",
        "signature" : "some very long hash"
    }

Add an extra line so that the metadata now looks like:

    {
        "name" : "",
        "signature" : "don't change the hash, but add a comma at the end of the line",
        "git" : { "suppress_outputs" : false }
    }

********************************************************************************
WARNING: If you change this file, all notebooks in the repository will be reported as
modified by git, as git only sees the representation created by this file.
So when you change the file format of this, (1) you need update all other notebook files,
and (2) notify the other developers, as changing all notebooks could cause issues.
********************************************************************************
"""
import sys
import json

nb = sys.stdin.read()

try:
    json_in = json.loads(nb)
except Exception as e:
    sys.stderr.write("Invalid JSON file in notebook\n")
    sys.stderr.write(str(e))
    sys.stderr.write("\n")
    sys.stderr.write("Leaving notebook unchanged\n")

    sys.stdout.write(nb)
    exit()
nb_metadata = json_in["metadata"]
suppress_output = True

if "git" in nb_metadata:
    if "suppress_outputs" in nb_metadata["git"]:
        suppress_output = nb_metadata["git"]["suppress_outputs"]
if not suppress_output:
    # write notebook unchanged, with outputs etc.
    sys.stdout.write(nb)
    exit()


def strip_output_from_cell(cell):
    if "outputs" in cell:
        cell["outputs"] = []
    if "execution_count" in cell:
        cell["execution_count"] = None
    if "metadata" not in cell:
        cell["metadata"] = {}
    if "ExecuteTime" in cell["metadata"]:
        del cell["metadata"]["ExecuteTime"]

    for metadata_option, default in [("deletable", True), ("editable", True), ("collapsed", False)]:
        if metadata_option not in cell["metadata"]:
            cell["metadata"][metadata_option] = default

    for metadata_option, default in [("heading_collapsed", False), ("hidden", False)]:
        cell["metadata"][metadata_option] = default

    cell["metadata"]["deletable"] = True
    cell["metadata"]["editable"] = True


for cell in json_in["cells"]:
    strip_output_from_cell(cell)

output = json.dumps(json_in, sort_keys=True, indent=1, ensure_ascii=False, separators=(",", ": "))
if sys.version_info >= (3, 0):
    print(output)
else:
    print(output.encode("utf-8"))
