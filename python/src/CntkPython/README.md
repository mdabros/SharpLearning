# Installation

## Python 3.6 in Visual Studio 2017
Use **Visual Studio Installer** -> select **Modify**.

Workload:
 * Install **Data science and analytical applications**  workload

Components:
 * Install **Python language support**
 * Install **Python 3 64-bit** (e.g. 3.6.3)

Install cntk packages:
 * Start Visual Studio, open **Python Environments**
 * Select **Python 3.6 (64-bit)**
 * On the **Overview** combo box select **Packages**
 * For GPU/CPU (v. 2.3.1): Search for `https://cntk.ai/PythonWheel/GPU/cntk-2.3.1-cp36-cp36m-win_amd64.whl`
 * For CPU-Only (v. 2.3.1): Search for `https://cntk.ai/PythonWheel/CPU-Only/cntk-2.3.1-cp36-cp36m-win_amd64.whl`

The above will install cntk, and the scripts will work with cntk from the python or anaconda command line.
However, visual studio does not recognize cntk for some reason, even though the environment seems to be the same.
Some discussion is going on here: [cntk python from visual studio](https://github.com/Microsoft/CNTK/issues/1587). The conclusion seems to be that you have to launch
visual studio from the python enviroment prompt...

So currently the examples can only run from the prompt and not visual studio. Which means no debugging.

 # Running
 Right click a `py` file and select **Start with Debugging** for example, or set as startup file and press F5.