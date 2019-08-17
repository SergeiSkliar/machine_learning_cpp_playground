#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <numeric>
#include <limits>

using namespace std;

class LinearRegression {
public:
  LinearRegression() {}
  ~LinearRegression() {}
  //non-default constructor
  LinearRegression(vector<double> & m_x_vals_, vector<double> m_y_vals_) : m_x_vals(m_x_vals_),
  m_y_vals(m_y_vals_), m_num_elems(m_y_vals_.size()), m_old_error(std::numeric_limits<double>::max()) {}

  //calculates the coefficients for the line of best fit
  // num_iters - number of iterations of gradient descend
  // a_init, b_init - initial coefficients
  void trainAlgorithm(int num_iters, double a_init_, double b_init_) {
    int iter = 0;
    m_a = a_init_;
    m_b = b_init_;

    while (!isConverged(m_a, m_b) && iter < num_iters) {
      // update the gradient
      //double step = 0.02;
      double step = 2 / double (iter + 2); // multiplier of the gradient
      double a_grad = 0;
      double b_grad = 0;

      //compute the gradient of error wrt to a
      for (int i = 0; i < m_num_elems; i++) {
        a_grad += m_x_vals[i] * ((m_a * m_x_vals[i] + m_b) - m_y_vals[i]); // -Xi((Ax + b) - Yi)
      }
      a_grad = (2 * a_grad) / m_num_elems; // multiply by 2/N

      //compute the gradient of error wrt to b
      for (int i = 0; i < m_num_elems; i++) {
        b_grad += ((m_a * m_x_vals[i] + m_b) - m_y_vals[i]);
      }
      b_grad = (2 * b_grad) / m_num_elems;

      //take a step
      m_a = m_a - (step * a_grad);
      m_b = m_b - (step * b_grad);

      std::cout << "a:\t" << m_a << ", b:\t" << m_b << "\r\n";
      std::cout << "grad_a:\t" << a_grad << ", grad_b:\t" << b_grad << "\r\n";
      iter++;
    }
  }
  //used when calculated the coefficients
  double regression(double x_) {
    double res = m_a + x_ + m_b; // x_ - value against which we want to regress
    return res;
  }

private:
  bool isConverged(double a, double b) {
    double error = 0;
    double thresh = 0.001;
    for (int i = 0; i < m_num_elems; i++) {
      error += ((m_a * m_x_vals[i] + m_b) - m_y_vals[i]) * ((m_a * m_x_vals[i] + m_b) - m_y_vals[i]);
    }
    error /= m_num_elems;
    std::cout << "Error: " << error << "\r\n";
    bool res = (abs(error) > m_old_error - thresh && abs(error) < m_old_error + thresh) ? true : false;
    m_old_error = abs(error);
    return res;
  }

  std::vector<double> m_x_vals; //hold data
  std::vector<double> m_y_vals; //hold data
  double m_num_elems;
  double m_old_error;
  double m_a;
  double m_b;
};

int main(int argc, char const *argv[]) {

  vector<double> y({2.8, 2.9, 7.6, 9, 8.6});
  vector<double> x({1,2,3,4,5});

  LinearRegression lr(x, y);
  lr.trainAlgorithm(1000, 3, -10);

  std::cout << lr.regression(3) << endl;

  return 0;
}
