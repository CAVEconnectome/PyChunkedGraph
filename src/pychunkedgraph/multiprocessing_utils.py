import numpy as np
from multiprocessing import cpu_count, Process
from multiprocessing.pool import Pool
import time
import os
import shutil
import pickle as pkl
import sys
import subprocess

python_path = sys.executable
home = os.path.expanduser("~")
file_path = os.path.dirname(os.path.abspath(__file__))
src_folder_path = file_path.split("src")[0] + "src/"
subp_work_folder = home + "/pychg_subp_workdir/"


def multiprocess_func(func, params, debug=False, verbose=False, nb_cpus=None):

    if nb_cpus is None:
        nb_cpus = max(cpu_count(), 1)

    if debug:
        nb_cpus = 1

    if verbose:
        print("Computing %d parameters with %d cpus." % (len(params), nb_cpus))

    start = time.time()
    if not debug:
        pool = Pool(nb_cpus)
        result = pool.map(func, params)
        pool.close()
        pool.join()
    else:
        result = []
        for p in params:
            result.append(func(p))

    if verbose:
        print("\nTime to compute grid: %.3fs" % (time.time() - start))

    return result


def multisubprocess_func(func, params, wait_delay_s=5, n_subprocesses=1):
    name = func.__name__.strip("_")

    if os.path.exists(subp_work_folder + "/%s_folder/" % (name)):
        shutil.rmtree(subp_work_folder + "/%s_folder/" % (name))

    path_to_storage = subp_work_folder + "/%s_folder/storage/" % name
    path_to_out = subp_work_folder + "/%s_folder/out/" % name
    path_to_src = subp_work_folder + "/%s_folder/src/" % name
    path_to_script = subp_work_folder + "/%s_folder/main.py" % name

    os.makedirs(subp_work_folder + "/%s_folder/" % (name))
    os.makedirs(path_to_storage)
    os.makedirs(path_to_out)

    shutil.copytree(src_folder_path, path_to_src)
    write_multisubprocess_script(func, path_to_script, path_to_src)

    processes = []
    for ii in range(len(params)):
        while len(processes) >= n_subprocesses:
            for i_p, p in enumerate(processes):
                poll = p.poll()
                # print("Poll", p)
                if poll is not None:
                    del(processes[i_p])
                    break

            if len(processes) >= n_subprocesses:
                time.sleep(wait_delay_s)

        this_storage_path = path_to_storage + "job_%d.pkl" % ii
        this_out_path = path_to_out + "job_%d.pkl" % ii

        with open(this_storage_path, "wb") as f:
            pkl.dump(params[ii], f)

        p = subprocess.Popen("cd %s; %s -W ignore %s  %s %s" % (path_to_src,
                                                     python_path,
                                                     path_to_script,
                                                     this_storage_path,
                                                     this_out_path), shell=True)
        processes.append(p)
        time.sleep(.01)  # Avoid OS hickups

    for p in processes:
        p.wait()

    return path_to_out


def write_multisubprocess_script(func, path_to_script, path_to_src):
    module = sys.modules.get(func.__module__)
    module_path = path_to_src + module.__file__.split("src")[1]
    module_h = module.__file__.split("src")[1].strip("/").split("/")
    module_h[-1] = module_h[-1][:-3]

    lines = []
    lines.append("".join(["from src."] + [f for f in module_h[:-1]] +
                         [" import %s" % module_h[-1]] + ["\n"]))
    lines.extend(["import pickle as pkl\n",
                  "import sys\n\n\n"
                  "def main(p_params, p_out):\n\n",
                  "\twith open(p_params, 'rb') as f:\n",
                  "\t\tparams = pkl.load(f)\n\n",
                  "\tr = %s.%s(params)\n\n" % (module_h[-1], func.__name__),
                  "\twith open(p_out, 'wb') as f:\n",
                  "\t\tpkl.dump(r, f)\n\n",
                  "if __name__ == '__main__':\n",
                  "\tmain(sys.argv[1], sys.argv[2])\n"])

    with open(path_to_script, "w") as f:
        f.writelines(lines)



