# Written by: Peiqi Xia
# GitHub username: edsml-px122

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtGui import QPixmap
import numpy as np
import funcs as fc
import base

class GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.load_ui()
    
    def load_ui(self):
        """
        Open the default window of user interface and connect the buttons with functions.
        """
        self.ui = base.Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.start.clicked.connect(self.run)
        self.ui.save.clicked.connect(self.save_results)
        self.ui.clean.clicked.connect(self.clean_input)
        self.ui.help.clicked.connect(self.helpbox)
    
    def run(self):
        """
        Complete workflow of this software.
        """
        # receive inputs from the user interface and convert into lists or arrays
        t = [np.array(list(map(float, x.split(',')))) for x in self.ui.time.toPlainText().split('\n')]
        qt = [np.array(list(map(float, x.split(',')))) for x in self.ui.amount_of_absorbate.toPlainText().split('\n')]
        C0 = np.array(list(map(float,self.ui.concentration_of_absorbate.text().split(","))))
        Cs = np.array(list(map(float,self.ui.concentration_of_absorbent.text().split(","))))
        unit_input = [self.ui.t_unit.text(), self.ui.qt_unit.text(), self.ui.C0_unit.text(), self.ui.Cs_unit.text()]
        # set units if they are given
        units = ['min', 'mg/g', 'mg/L', 'g/L']
        for i in range(4):
            if unit_input[i]:
                units[i] = unit_input[i]

        # check if the size of inputs matches
        if len(t) == len(qt) and all(t[i].shape == qt[i].shape for i in range(len(qt))) and len(qt) == len(C0):
            # for single dataset
            if len(qt) == 1:
                # fit the data with six models
                k0, r_sq0, q_model0 = fc.ZO_linear(t[0], qt[0])
                qe1_l, k1_l, r_sq1_l, q_model1_l = fc.PFO_linear(t[0], qt[0])
                qe1_n, k1_n, r_sq1_n, q_model1_n = fc.PFO_nonlinear(t[0], qt[0])
                qe2_l, k2_l, r_sq2_l, q_model2_l = fc.PSO_linear(t[0], qt[0])
                qe2_n, k2_n, r_sq2_n, q_model2_n = fc.PSO_nonlinear(t[0], qt[0])
                qe2_r, k2_r, r_sq2_r, q_model2_r = fc.rPSO_nonlinear(t[0], qt[0], C0[0], Cs[0])

                # calculate uncertainties of all the parameters
                _, k0_err = fc.error_analysis(t[0], qt[0], q_model0, 0, fc.ZO_linear)
                qe1_err_l, k1_err_l = fc.error_analysis(t[0], qt[0], q_model1_l, 1, fc.PFO_linear)
                qe1_err_n, k1_err_n = fc.error_analysis(t[0], qt[0], q_model1_n, 1, fc.PFO_nonlinear)
                qe2_err_l, k2_err_l = fc.error_analysis(t[0], qt[0], q_model2_l, 2, fc.PSO_linear)
                qe2_err_n, k2_err_n = fc.error_analysis(t[0], qt[0], q_model2_n, 2, fc.PSO_nonlinear)
                qe2_err_r, k2_err_r = fc.error_analysis(t[0], qt[0], q_model2_r, 2, fc.rPSO_nonlinear, C0[0], Cs[0])

                params = [k0, qe1_l, k1_l, qe1_n, k1_n, qe2_l, k2_l, qe2_n, k2_n, qe2_r, k2_r]
                r_sq = [r_sq0, r_sq1_l, r_sq1_n, r_sq2_l, r_sq2_n, r_sq2_r]
                # compare the value of R^2 to determine the best simulated model
                if max(r_sq) == r_sq[0]:
                    result = f'This reaction is best simulated with linear ZO model - zero order with respect to adsorbate.\nModel parameters:\nk0: {k0:.6f} +- {k0_err:.6f} {units[1]}{units[0]}-1\nR^2: {r_sq0:.6f}\n'
                elif max(r_sq) == r_sq[1]:
                    result = f'This reaction is best simulated with linear PFO model - first order with respect to adsorbate.\nModel parameters:\nk1: {k1_l:.6f} +- {k1_err_l:.6f} {units[0]}-1\nqe: {qe1_l:.6f} +- {qe1_err_l:.6f} {units[1]}\nR^2: {r_sq1_l:.6f}\n'
                elif max(r_sq) == r_sq[2]:
                    result = f'This reaction is best simulated with nonlinear PFO model - first order with respect to adsorbate.\nModel parameters:\nk1: {k1_n:.6f} +- {k1_err_n:.6f} {units[0]}-1\nqe: {qe1_n:.6f} +- {qe1_err_n:.6f} {units[1]}\nR^2: {r_sq1_n:.6f}\n'
                elif max(r_sq) == r_sq[3]:
                    result = f'This reaction is best simulated with linear PSO model - second order with respect to adsorbate.\nModel parameters:\nk2: {k2_l:.6f} +- {k2_err_l:.6f} ({units[1]})-1{units[0]}-1\nqe: {qe2_l:.6f} +- {qe2_err_l:.6f} {units[1]}\nR^2: {r_sq2_l:.6f}\n'
                elif max(r_sq) == r_sq[4]:
                    result = f'This reaction is best simulated with nonlinear PSO model - second order with respect to adsorbate.\nModel parameters:\nk2: {k2_n:.6f} +- {k2_err_n:.6f} ({units[1]})-1{units[0]}-1\nqe: {qe2_n:.6f} +- {qe2_err_n:.6f} {units[1]}\nR^2: {r_sq2_n:.6f}\n'
                else:
                    result = f'This reaction is best simulated with nonlinear rPSO model - second order with respect to adsorbate.(initial concentrations taken into account)\nModel parameters:\nk2: {k2_r:.6f} +- {k2_err_r:.6f} ({units[1]})-1{units[0]}-1\nqe: {qe2_r:.6f} +- {qe2_err_r:.6f} {units[1]}\nR^2: {r_sq2_r:.6f}\n'

                # display all the parameters in a table
                table = [
                    ['Model            ', 'k      ', 'qe     ', 'R^2'], 
                    ['Linear ZO      ', f'{k0:.6f} +- {k0_err:.6f}', '   /   ', f'{r_sq0:.6f}'], 
                    ['Linear PFO     ', f'{k1_l:.6f} +- {k1_err_l:.6f}', f'{qe1_l:.6f} +- {qe1_err_l:.6f}', f'{r_sq1_l:.6f}'], 
                    ['Nonlinear PFO', f'{k1_n:.6f} +- {k1_err_n:.6f}', f'{qe1_n:.6f} +- {qe1_err_n:.6f}', f'{r_sq1_n:.6f}'], 
                    ['Linear PSO     ', f'{k2_l:.6f} +- {k2_err_l:.6f}', f'{qe2_l:.6f} +- {qe2_err_l:.6f}', f'{r_sq2_l:.6f}'], 
                    ['Nonlinear PSO', f'{k2_n:.6f} +- {k2_err_n:.6f}', f'{qe2_n:.6f} +- {qe2_err_n:.6f}', f'{r_sq2_n:.6f}'], 
                    ['Nonlinear rPSO', f'{k2_r:.6f} +- {k2_err_r:.6f}', f'{qe2_r:.6f} +- {qe2_err_r:.6f}', f'{r_sq2_r:.6f}']
                    ]
                data = "\n".join(" | ".join(row) for row in table)
                fc.plot_single_data(t[0], qt[0], C0[0], Cs[0], params, units)

            # for multiple datasets
            else:
                # calculate initial rates and reaction order
                ini_rates = fc.ini_rate(t, qt)
                order, r_pred = fc.order_analysis(ini_rates, C0)

                params = []
                # determine the model to fit using the reaction order number
                # fit each dataset with this model and calculate parameter uncertainties
                if round(order) == 0:
                    result = f'rate = k[adsorbate]^{order:.2f}\nZero order model is applied to simulate the data.\n'
                    table = [['dataset', f'k0 ({units[1]}{units[0]}-1)', 'R^2']]
                    for i in range(len(qt)):
                        k0, r_sq, q_model = fc.ZO_linear(t[i], qt[i])
                        params.append(k0)
                        _, k0_err = fc.error_analysis(t[i], qt[i], q_model, 0, fc.ZO_linear)
                        table.append([f'{i+1}', f'{k0:.6f} +- {k0_err:.6f}', f'{r_sq:.6f}'])
                    data = "\n".join(" | ".join(row) for row in table)
                elif round(order) == 1:
                    result = f'rate = k[adsorbate]^{order:.2f}\nNonlinear PFO model is applied to simulate the data.\n'
                    table = [['dataset', f'k1 ({units[0]}-1)', f'qe ({units[1]})', 'R^2']]
                    for i in range(len(qt)):
                        qe, k1, r_sq, q_model = fc.PFO_nonlinear(t[i], qt[i])
                        params.append(qe)
                        params.append(k1)
                        qe_err, k1_err = fc.error_analysis(t[i], qt[i], q_model, 1, fc.PFO_nonlinear)
                        table.append([f'{i+1}', f'{k1:.6f} +- {k1_err:.6f}', f'{qe:.6f} +- {qe_err:.6f}', f'{r_sq:.6f}'])
                    data = "\n".join(" | ".join(row) for row in table)
                elif round(order) == 2:
                    result = f'rate = k[adsorbate]^{order:.2f}\nNonlinear PSO model is applied to simulate the data.\n'
                    table = [['dataset', f'k2 (({units[1]})-1{units[0]}-1)', f'qe ({units[1]})', 'R^2']]
                    for i in range(len(qt)):
                        qe, k2, r_sq, q_model = fc.PSO_nonlinear(t[i], qt[i])
                        params.append(qe)
                        params.append(k2)
                        qe_err, k2_err = fc.error_analysis(t[i], qt[i], q_model, 2, fc.PSO_nonlinear)
                        table.append([f'{i+1}', f'{k2:.6f} +- {k2_err:.6f}', f'{qe:.6f} +- {qe_err:.6f}', f'{r_sq:.6f}'])
                    data = "\n".join(" | ".join(row) for row in table)
                else:
                    result = f'rate = k[adsorbate]^{order:.2f}\nBeyond our availability.'
                    data = None
                
                units.append(f'{units[1]} {units[0]}-1')
                fc.plot_multi_data(t, qt, C0, ini_rates, order, r_pred, params, units)
            
            # show the text and plot results on the user interface
            self.ui.result_data.append(result)
            self.ui.result_data.append(data)
            self.ui.result_data.append('\n')
            self.ui.result_plot.setPixmap(QPixmap('../result/result_image.png'))

        # the pop-up window if the size of inputs does not match
        else:
            msg = QMessageBox()
            msg.setWindowTitle('Warning')
            msg.setText('Invalid input! Please enter again.')
            msg.setIcon(QMessageBox.Critical)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

    def save_results(self):
        """
        Save the text results and pop up a window when it's finished.
        """
        with open('../result/result_data.txt', 'w') as file:
            results = self.ui.result_data.toPlainText()
            file.write(results)
        msg = QMessageBox()
        msg.setWindowTitle('Save Results')
        msg.setText('Saved successfully in the "result" folder!\n\nPlease rename it or move to other folders, or it will be covered in the next processes.')
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def clean_input(self):
        """
        Clean all the inputs and outputs on the user interface.
        """
        self.ui.time.clear()
        self.ui.amount_of_absorbate.clear()
        self.ui.concentration_of_absorbate.clear()
        self.ui.concentration_of_absorbent.clear()
        self.ui.t_unit.clear()
        self.ui.qt_unit.clear()
        self.ui.C0_unit.clear()
        self.ui.Cs_unit.clear()
        self.ui.result_data.clear()
        self.ui.result_plot.clear()

    def helpbox(self):
        """
        A pop-up window to explain the contents of graph.
        """
        msg = QMessageBox()
        msg.setWindowTitle('Help')
        msg.setText('Single dataset: the image shows plots of all the six fitted models.\n\nMultiple datasets: the image shows the order of reaction from initial-rate vs initial-concentration plot and all the datasets fitted with the chosen best model.\n\nThe image will be saved automatically in the "result" folder. Please rename it or move to other folders, or it will be covered in the next processes.')
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Close)
        msg.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = GUI()
    main.show()
    sys.exit(app.exec_())