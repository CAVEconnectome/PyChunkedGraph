import os
import subprocess
import glob

path_to_scripts = os.path.dirname(__file__) + "/meshlabserver_scripts/"


def run_script(script_name, arg_dict):
    """ Runs meshlabserver script --headless

    No X-Server required

    :param script_name: str
    :param arg_dict: dict [str: str]
    """
    arg_string = "".join(["-{0} {1} ".format(k, arg_dict[k])
                          for k in arg_dict.keys()])
    command = "xvfb-run meshlabserver -s {0} {1}".\
        format(script_name, arg_string)

    p = subprocess.Popen(command, shell=True, stderr=subprocess.PIPE)
    p.wait()


def run_script_on_dir(script_name, dir, suffix, arg_dict={}):
    paths = glob.glob(dir + "/*.obj")

    for path in paths:
        out_path = "{}_{}.obj".format("".join(path.split(".")[:-1]), suffix)

        this_arg_dict = {"i": path, "o": out_path}
        this_arg_dict.update(arg_dict)

        run_script(script_name, this_arg_dict)
