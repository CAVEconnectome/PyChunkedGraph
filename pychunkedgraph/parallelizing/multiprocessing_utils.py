import numpy as np
from multiprocessing import cpu_count, Process
from multiprocessing.pool import Pool, ThreadPool
import time
import os
import shutil
import pickle as pkl
import sys
import subprocess
import glob
import logging

python_path = sys.executable
home = os.path.expanduser("~")
file_path = os.path.dirname(os.path.abspath(__file__))
src_folder_path = file_path.split("pychunkedgraph")[0] + "pychunkedgraph/"
subp_work_folder = home + "/pychg_subp_workdir/"

n_cpus = cpu_count()

def _load_multifunc_out_thread(out_file_path):
    """ Reads the outputs from any multi func from disk

    :param results_folder: str
    :return: list of returns
    """
    # results = []
    # for out_fp in out_file_path:
    with open(out_file_path, "rb") as f:
        return pkl.load(f)

    # return results


def multiprocess_func(func, params, debug=False, verbose=False, n_threads=None):
    """ Processes data independent functions in parallel using parallelizing

    :param func: function
    :param params: list
        list of arguments to function
    :param debug: bool
    :param verbose: bool
    :param n_threads: int
    :return: list of returns of function
    """

    if n_threads is None:
        n_threads = max(cpu_count(), 1)

    if debug:
        n_threads = 1

    if verbose:
        print("Computing %d parameters with %d cpus." % (len(params), n_threads))

    start = time.time()
    if not debug:
        pool = Pool(n_threads)
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


def multithread_func(func, params, debug=False, verbose=False, n_threads=None):
    """ Processes data independent functions in parallel using multithreading

    :param func: function
    :param params: list
        list of arguments to function
    :param debug: bool
    :param verbose: bool
    :param n_threads: int
    :return: list of returns of function
    """

    if n_threads is None:
        n_threads = max(cpu_count(), 1)

    if debug:
        n_threads = 1

    if verbose:
        print("Computing %d parameters with %d cpus." % (len(params), n_threads))

    start = time.time()
    if not debug:
        pool = ThreadPool(n_threads)
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


def _start_subprocess(ii, path_to_script, path_to_storage, path_to_src,
                      path_to_out, params=None):

    this_storage_path = path_to_storage + "job_%d.pkl" % ii
    this_out_path = path_to_out + "job_%d.pkl" % ii

    if params is not None:
        with open(this_storage_path, "wb") as f:
            pkl.dump(params[ii], f)
    else:
        assert os.path.exists(this_storage_path)

    p = subprocess.Popen("cd %s; %s -W ignore %s  %s %s" %
                         (path_to_src, python_path, path_to_script,
                          this_storage_path, this_out_path), shell=True,
                         stderr=subprocess.PIPE)
    return p


def _poll_running_subprocesses(processes, runtimes, path_to_script,
                               path_to_storage, path_to_src,  path_to_out,
                               path_to_err, min_n_meas, kill_tol_factor,
                               n_retries, logger):
    if len(runtimes) > min_n_meas:
        median_runtime = np.median(runtimes)
    else:
        median_runtime = 0

    for i_p, p in enumerate(processes):
        poll = p[0].poll()

        if poll == 0 and os.path.exists(path_to_out):
            runtimes.append(time.time() - processes[i_p][3])
            del (processes[i_p])

            logger.info("process %d finished after %.3fs" % (p[1], runtimes[-1]))

            break
        elif poll == 1:
            _, err_msg = p[0].communicate()
            err_msg = err_msg.decode()

            _write_error_to_file(err_msg, path_to_err + "job_%d_%d.pkl" %
                                 (p[1], p[2]))

            logger.warning("process %d failed" % p[1])

            if p[2] >= n_retries:
                del (processes[i_p])

                logger.error("no retries left for process %d" % p[1])

                break
            else:
                p = _start_subprocess(i_p, path_to_script, path_to_storage,
                                      path_to_src,  path_to_out, params=None)

                processes[i_p][0] = p
                processes[i_p][2] += 1
                processes[i_p][3] = time.time()

                logger.info("restarted process %d -- n(restarts) = %d" %
                            (i_p, processes[i_p][2]))

                time.sleep(.01)  # Avoid OS hickups

        elif kill_tol_factor is not None and median_runtime > 0:
            if processes[i_p][2] == 0:
                this_kill_tol_factor = 5
            elif processes[i_p][2] == n_retries - 1:
                this_kill_tol_factor = kill_tol_factor * 10
            else:
                this_kill_tol_factor = kill_tol_factor

            p_run_time = time.time() - processes[i_p][3]
            if p_run_time > median_runtime * this_kill_tol_factor:
                processes[i_p][0].kill()
                print("\n\nKILLED PROCESS %d -- it was simply too slow... "
                      "(.%3fs), median is %.3fs -- %s\n\n" %
                      (p_run_time, median_runtime, i_p, path_to_storage))

                logger.warning("killed process %d -- (.%3fs), "
                               "median is %.3fs" % (i_p, p_run_time,
                                                    median_runtime))

                p = _start_subprocess(i_p, path_to_script, path_to_storage,
                                      path_to_src,  path_to_out, params=None)

                processes[i_p][0] = p
                processes[i_p][2] += 1
                processes[i_p][3] = time.time()

                logger.info("restarted process %d -- n(restarts) = %d" %
                            (i_p, processes[i_p][2]))

                time.sleep(.01)  # Avoid OS hickups
    return processes, runtimes


def multisubprocess_func(func, params, wait_delay_s=5, n_threads=1,
                         n_retries=10, kill_tol_factor=10,
                         suffix="", package_name="pychunkedgraph"):
    """ Processes data independent functions in parallel using multithreading

    :param func: function
    :param params: list
        list of arguments to function
    :param wait_delay_s: float
    :param n_threads: int
    :param n_retries: int
    :param kill_tol_factor: int or None
        kill_tol_factor x mean_run_time sets a threshold after which a process
        is restarted. If None: Processes are not restarted.
    :param min_n_meas: int
    :param suffix: str
    :param package_name: str
    :return: list of returns of function
    """

    name = func.__name__.strip("_")

    if len(suffix) > 0 and not suffix.startswith("_"):
        suffix = "_" + suffix

    subp_job_folder = subp_work_folder + "/%s%s_folder/" % (name, suffix)

    if os.path.exists(subp_job_folder):
        shutil.rmtree(subp_job_folder)

    path_to_storage = subp_job_folder + "/storage/"
    path_to_out = subp_job_folder + "/out/"
    path_to_err = subp_job_folder + "/err/"
    path_to_src = subp_job_folder + "/%s/" % package_name
    path_to_script = subp_job_folder + "/main.py"

    os.makedirs(subp_job_folder)
    os.makedirs(path_to_storage)
    os.makedirs(path_to_out)
    os.makedirs(path_to_err)

    # shutil.copytree(src_folder_path, path_to_src)
    _write_multisubprocess_script(func, path_to_script,
                                  package_name=package_name)

    min_n_meas = int(len(params) * .9)

    # Add logging
    # https: // docs.python.org / 3 / howto / logging - cookbook.html

    logger = logging.Logger(name)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('%s/logger.log' % subp_job_folder)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - '
                                  '%(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("Parameters: wait_delay_s={}, n_threads={},  n_retries={},  "
                "kill_tol_factor={},  suffix={}, package_name={}, "
                "min_n_meas={}".format(wait_delay_s, n_threads, n_retries,
                                       kill_tol_factor, suffix, package_name,
                                       min_n_meas))

    processes = []
    runtimes = []
    for ii in range(len(params)):
        while len(processes) >= n_threads:
            processes, runtimes = _poll_running_subprocesses(processes,
                                                             runtimes,
                                                             path_to_script,
                                                             path_to_storage,
                                                             path_to_src,
                                                             path_to_out,
                                                             path_to_err,
                                                             min_n_meas,
                                                             kill_tol_factor,
                                                             n_retries,
                                                             logger)

            if len(processes) >= n_threads:
                time.sleep(wait_delay_s)

        p = _start_subprocess(ii, path_to_script, path_to_storage, path_to_src,
                              path_to_out, params=params)

        logger.info("started process %d" % ii)

        processes.append([p, ii, 0, time.time()])
        time.sleep(.01)  # Avoid OS hickups

    while len(processes) > 0:
        processes, runtimes = _poll_running_subprocesses(processes,
                                                         runtimes,
                                                         path_to_script,
                                                         path_to_storage,
                                                         path_to_src,
                                                         path_to_out,
                                                         path_to_err,
                                                         min_n_meas,
                                                         kill_tol_factor,
                                                         n_retries,
                                                         logger)

        if len(processes) >= n_threads:
            time.sleep(wait_delay_s)

    out_file_paths = glob.glob(path_to_out + "/*")

    if len(out_file_paths) > 0:
        result = multithread_func(_load_multifunc_out_thread, out_file_paths,
                                  n_threads=n_threads, verbose=False)
    else:
        result = None

    return result


def _write_multisubprocess_script(func, path_to_script,
                                  package_name="pychunkedgraph"):
    """ Helper script to write python file called by subprocess """

    module = sys.modules.get(func.__module__)
    module_h = module.__file__.split(package_name)[1].strip("/").split("/")
    module_h[-1] = module_h[-1][:-3]

    lines = []
    lines.append("".join(["from %s" % package_name] +
                         [".%s" % f for f in module_h[:-1]] +
                         [" import %s" % module_h[-1]] +
                         ["\n"]))
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


def _write_error_to_file(err_msg, path_to_err):
    """ Helper script to write error message to file """

    with open(path_to_err, "w") as f:
        f.write(err_msg)



