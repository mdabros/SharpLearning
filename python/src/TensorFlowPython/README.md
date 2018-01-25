# Installation

## Python 3.6 in Visual Studio 2017
Use **Visual Studio Installer** -> select **Modify**.

Workload:
 * Install **Data science and analytical applications**  workload

Components:
 * Install **Python language support**
 * Install **Python 3 64-bit** (e.g. 3.6.3)

Install Tensor Flow packages:
 * Start Visual Studio, open **Python Environments**
 * Select **Python 3.6 (64-bit)**
 * On the **Overview** combo box select **Packages**
 * Search for `tensorflow` and install.

 # Running
 Right click a `py` file and select **Start with Debugging** for example, or set as startup file and press F5.