# Written by: Peiqi Xia
# GitHub username: edsml-px122

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5 import uic
import numpy as np
import funcs as fc

class GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.load_ui()
    
    def load_ui(self):
        uic.loadUi('base.ui', self)
        self.start.clicked.connect(self.run)
        self.save.clicked.connect(self.save_results)
        self.clean.clicked.connect(self.clean_input)
        self.help.clicked.connect(self.helpbox)
    
    def run(self):
        t = [np.array(list(map(float, x.split(',')))) for x in self.time.toPlainText().split('\n')]
        qt = [np.array(list(map(float, x.split(',')))) for x in self.amount_of_absorbate.toPlainText().split('\n')]
        C0 = np.array(list(map(float,self.concentration_of_absorbate.text().split(","))))
        Cs = np.array(list(map(float,self.concentration_of_absorbent.text().split(","))))

        if len(t) == len(qt) and all(t[i].shape == qt[i].shape for i in range(len(qt))) and len(qt) == len(C0):
            if len(qt) == 1:
                k0, r_sq0 = fc.ZO_linear(t[0], qt[0])
                qe1_l, k1_l, r_sq1_l = fc.PFO_linear(t[0], qt[0])
                qe1_n, k1_n, r_sq1_n = fc.PFO_nonlinear(t[0], qt[0])
                qe2_l, k2_l, r_sq2_l = fc.PSO_linear(t[0], qt[0])
                qe2_n, k2_n, r_sq2_n = fc.PSO_nonlinear(t[0], qt[0])
                qe2_r, k2_r, r_sq2_r = fc.rPSO_nonlinear(t[0], qt[0], C0[0], Cs[0])

                params = [k0, qe1_l, k1_l, qe1_n, k1_n, qe2_l, k2_l, qe2_n, k2_n, qe2_r, k2_r]
                r_sq = [r_sq0, r_sq1_l, r_sq1_n, r_sq2_l, r_sq2_n, r_sq2_r]
                if max(r_sq) == r_sq[0]:
                    q_model = fc.ZO_getQ(t[0], k0)
                    _, k0_err = fc.error_analysis(t[0], qt[0], q_model, 0, fc.ZO_linear)
                    result = f'This reaction is best simulated with linear ZO model - zero order with respect to adsorbate.\nModel parameters:\nk0: {k0:.6f} +- {k0_err:.6f}\nR^2: {r_sq0:.6f}\n'
                elif max(r_sq) == r_sq[1]:
                    q_model = fc.PFO_getQ(t[0], qe1_l, k1_l)
                    qe_err, k1_err = fc.error_analysis(t[0], qt[0], q_model, 1, fc.PFO_linear)
                    result = f'This reaction is best simulated with linear PFO model - first order with respect to adsorbate.\nModel parameters:\nk1: {k1_l:.6f} +- {k1_err:.6f}\nqe: {qe1_l:.6f} +- {qe_err:.6f}\nR^2: {r_sq1_l:.6f}\n'
                elif max(r_sq) == r_sq[2]:
                    q_model = fc.PFO_getQ(t[0], qe1_n, k1_n)
                    qe_err, k1_err = fc.error_analysis(t[0], qt[0], q_model, 1, fc.PFO_nonlinear)
                    result = f'This reaction is best simulated with nonlinear PFO model - first order with respect to adsorbate.\nModel parameters:\nk1: {k1_n:.6f} +- {k1_err:.6f}\nqe: {qe1_n:.6f} +- {qe_err:.6f}\nR^2: {r_sq1_n:.6f}\n'
                elif max(r_sq) == r_sq[3]:
                    q_model = fc.PSO_getQ(t[0], qe2_l, k2_l)
                    qe_err, k2_err = fc.error_analysis(t[0], qt[0], q_model, 2, fc.PSO_linear)
                    result = f'This reaction is best simulated with linear PSO model - second order with respect to adsorbate.\nModel parameters:\nk2: {k2_l:.6f} +- {k2_err:.6f}\nqe: {qe2_l:.6f} +- {qe_err:.6f}\nR^2: {r_sq2_l:.6f}\n'
                elif max(r_sq) == r_sq[4]:
                    q_model = fc.PSO_getQ(t[0], qe2_n, k2_n)
                    qe_err, k2_err = fc.error_analysis(t[0], qt[0], q_model, 2, fc.PSO_nonlinear)
                    result = f'This reaction is best simulated with nonlinear PSO model - second order with respect to adsorbate.\nModel parameters:\nk2: {k2_n:.6f} +- {k2_err:.6f}\nqe: {qe2_n:.6f} +- {qe_err:.6f}\nR^2: {r_sq2_n:.6f}\n'
                else:
                    q_model = fc.rPSO_getQ(t[0], qe2_r, k2_r, C0[0], Cs[0])
                    qe_err, k2_err = fc.error_analysis(t[0], qt[0], q_model, 2, fc.rPSO_nonlinear, C0[0], Cs[0])
                    result = f'This reaction is best simulated with nonlinear rPSO model - second order with respect to adsorbate.(initial concentrations taken into account)\nModel parameters:\nk2: {k2_r:.6f} +- {k2_err:.6f}\nqe: {qe2_r:.6f} +- {qe_err:.6f}\nR^2: {r_sq2_r:.6f}\n'

                table = [
                    ['Model            ', 'k      ', 'qe     ', 'R^2'], 
                    ['Linear ZO      ', f'{k0:.6f}', '   /   ', f'{r_sq0:.6f}'], 
                    ['Linear PFO     ', f'{k1_l:.6f}', f'{qe1_l:.6f}', f'{r_sq1_l:.6f}'], 
                    ['Nonlinear PFO', f'{k1_n:.6f}', f'{qe1_n:.6f}', f'{r_sq1_n:.6f}'], 
                    ['Linear PSO     ', f'{k2_l:.6f}', f'{qe2_l:.6f}', f'{r_sq2_l:.6f}'], 
                    ['Nonlinear PSO', f'{k2_n:.6f}', f'{qe2_n:.6f}', f'{r_sq2_n:.6f}'], 
                    ['Nonlinear rPSO', f'{k2_r:.6f}', f'{qe2_r:.6f}', f'{r_sq2_r:.6f}']
                    ]
                data = "\n".join(" | ".join(row) for row in table)
                fc.plot_single_data(t[0], qt[0], C0[0], Cs[0], params)

            else:
                ini_rates = fc.ini_rate(t, qt)
                order, r_pred = fc.order_analysis(ini_rates, C0)

                params = []
                if round(order) == 0:
                    result = f'rate = k[adsorbate]^{order:.2f}\nZero order model is applied to simulate the data.\n'
                    table = [['dataset', 'k0', 'R^2']]
                    for i in range(len(qt)):
                        k0, r_sq = fc.ZO_linear(t[i], qt[i])
                        params.append(k0)
                        q_model = fc.ZO_getQ(t[i], k0)
                        _, k0_err = fc.error_analysis(t[i], qt[i], q_model, 0, fc.ZO_linear)
                        table.append([f'{i+1}', f'{k0:.6f} +- {k0_err:.6f}', f'{r_sq:.6f}'])
                    data = "\n".join(" | ".join(row) for row in table)
                elif round(order) == 1:
                    result = f'rate = k[adsorbate]^{order:.2f}\nNonlinear PFO model is applied to simulate the data.\n'
                    table = [['dataset', 'k1', 'qe', 'R^2']]
                    for i in range(len(qt)):
                        qe, k1, r_sq = fc.PFO_nonlinear(t[i], qt[i])
                        params.append(qe)
                        params.append(k1)
                        q_model = fc.PFO_getQ(t[i], qe, k1)
                        qe_err, k1_err = fc.error_analysis(t[i], qt[i], q_model, 1, fc.PFO_nonlinear)
                        table.append([f'{i+1}', f'{k1:.6f} +- {k1_err:.6f}', f'{qe:.6f} +- {qe_err:.6f}', f'{r_sq:.6f}'])
                    data = "\n".join(" | ".join(row) for row in table)
                elif round(order) == 2:
                    result = f'rate = k[adsorbate]^{order:.2f}\nNonlinear PSO model is applied to simulate the data.\n'
                    table = [['dataset', 'k2', 'qe', 'R^2']]
                    for i in range(len(qt)):
                        qe, k2, r_sq = fc.PSO_nonlinear(t[i], qt[i])
                        params.append(qe)
                        params.append(k2)
                        q_model = fc.PSO_getQ(t[i], qe, k2)
                        qe_err, k2_err = fc.error_analysis(t[i], qt[i], q_model, 2, fc.PSO_nonlinear)
                        table.append([f'{i+1}', f'{k2:.6f} +- {k2_err:.6f}', f'{qe:.6f} +- {qe_err:.6f}', f'{r_sq:.6f}'])
                    data = "\n".join(" | ".join(row) for row in table)
                else:
                    result = f'rate = k[adsorbate]^{order:.2f}\nBeyond our availability.'
                    data = None

                fc.plot_multi_data(t, qt, C0, ini_rates, order, r_pred, params)
            
            self.result_data.append(result)
            self.result_data.append(data)
            self.result_plot.setPixmap(QPixmap('../result/result_image.png'))

        else:
            msg = QMessageBox()
            msg.setWindowTitle('Warning')
            msg.setText('Invalid input! Please enter again.')
            msg.setIcon(QMessageBox.Critical)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

    def save_results(self):
        with open('../result/result_data.txt', 'w') as file:
            results = self.result_data.toPlainText()
            file.write(results)
        msg = QMessageBox()
        msg.setWindowTitle('Save Results')
        msg.setText('Saved successfully in the "result" folder!\n\nPlease rename it or move to other folders, or it will be covered in the next processes.')
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def clean_input(self):
        self.time.clear()
        self.amount_of_absorbate.clear()
        self.concentration_of_absorbate.clear()
        self.concentration_of_absorbent.clear()
        self.result_data.clear()
        self.result_plot.clear()

    def helpbox(self):
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