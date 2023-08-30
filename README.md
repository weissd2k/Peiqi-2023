## File Structure

- [`data`](./data): contains the collected data for validation and analysed data spreadsheet
- [`result`](./result): contains the raw calculation results, the path of saving results from the software which is fixed
- [`tool`](./tool): contains all the developed codes:
    - `base.py`: design of user interface (generated code)
    - `base.ui`: design of user interface (ui file for linux system)
    - `funcs.py`: useful functions involved in the software
    - `main.py`: combination of the functions in the user interface
    - `run_model.ipynb`: a notebook version of the software, which can be used online

## User Instruction

To run this program locally, you need to install the software in the command line,

1. Clone this repository:

    `git clone https://github.com/weissd2k/Peiqi-2023.git`

2. Navigate to this repository folder:

    `cd Peiqi-2023`

3. Create the required environment named 'px122':

    `conda env create -f environmental.yml`

3. Activate this environment:

    `conda activate px122`

4. Navigate to the `tool` folder:

    `cd tool`

5. Run the `main.py` file:

    `python main.py`

If you want to run this program online, there's no need to install this environment. You can just upload the `run_model.ipynb` file to Google Colab and play around with the codes.