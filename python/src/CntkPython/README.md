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

Add python environment path to the `PATH` environment variable
 * Locate the path to the selected python environment (CNTK must be installed)
   * For instance `C:\Users\User\AppData\Local\Continuum\anaconda3`
 * Open System environment variables and add the location to `PATH`

 # Running
 Right click a `py` file and select **Start with Debugging** for example, or set as startup file and press F5.