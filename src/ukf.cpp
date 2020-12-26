#include <iostream>
#include <limits>
#include <stdexcept>

#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

// helper function to normalize angle in the range (-PI, +PI]
double normalizeAngle(double angle) {
  while (angle > M_PI) angle -= 2 * M_PI;
  while (angle <= -M_PI) angle += 2 * M_PI;
  return angle;
};

} // namespace

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2.0; // TODO: change to a more approriate value
  double var_a = std_a_ * std_a_;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1.0; // TODO: change to a more approriate value
  double var_yawdd = std_yawdd_ * std_yawdd_;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * DONE: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

  // the UKF will be initialized on first sensor measurement
  is_initialized_ = false;

  // initialize state covariance matrix as identity matrix
  P_.setIdentity();

  // initialize process noise matrix
  int n_noise = 2; // noise vector size (longitudinal acceleration and yaw acceleration)
  Q_ = MatrixXd(n_noise, n_noise);
  Q_.fill(0.0);
  Q_ << var_a, 0, 0, var_yawdd;

  // dimensions and parameters
  n_x_ = 5; // for CTRV model (px, py, v, yaw, yaw rate)
  n_aug_ = n_x_ + n_noise; // augment with noise vector (longitudinal acceleration and yaw acceleration)
  n_sig_ = 2 * n_aug_ + 1; // 2 sigma points for each state variable, plus 1 sigma point for the mean
  lambda_ = 3 - n_aug_; // use typical lambda design parameter value

  // generate weights for predicted mean and covariance
  weights_ = VectorXd(n_sig_);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (size_t i = 1; i < n_sig_; ++i) {
      weights_(i) = 0.5 / (lambda_ + n_aug_);
  }

  // initialize predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, n_sig_);

  // initialize laser measurement noise matrix
  static constexpr int n_z_laser = 2; // laser can measure px and py
  R_laser_ = MatrixXd(n_z_laser, n_z_laser);
  double var_laspx = std_laspx_ * std_laspx_;
  double var_laspy = std_laspy_ * std_laspy_;
  R_laser_ <<  var_laspx,          0,
                       0,  var_laspy;

  // initialize radar measurement noise matrix
  static constexpr int n_z_radar = 3; // radar can measure radial distance , velocity yaw, and velocity
  R_radar_ = MatrixXd(n_z_radar, n_z_radar);
  double var_radr = std_radr_ * std_radr_;
  double var_radphi = std_radphi_ * std_radphi_;
  double var_radrd = std_radrd_ * std_radrd_;
  R_radar_ <<  var_radr,          0,         0,
                      0, var_radphi,         0,
                      0,          0, var_radrd;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * DONE: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */

  if (!is_initialized_) {
    switch(meas_package.sensor_type_) {
      case MeasurementPackage::SensorType::LASER: {
        double px = meas_package.raw_measurements_(0);
        double py = meas_package.raw_measurements_(1);

        double v = 0;
        double yaw = 0;
        double yawd = 0;

        x_ << px, py, v, yaw, yawd;
        break;
      }
      case MeasurementPackage::SensorType::RADAR: {
        double r = meas_package.raw_measurements_(0);
        double yaw = meas_package.raw_measurements_(1);
        double v = meas_package.raw_measurements_(2);

        double px = r * std::cos(yaw);
        double py = r * std::sin(yaw);
        double yawd = 0;

        x_ << px, py, v, yaw, 0;
        break;
      }
      default:
        throw std::runtime_error("Error! Received measurement from unknown SensorType");
    }

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

  // calculate dt in seconds
  static constexpr double US_TO_S = 1 / 1e6; // microseconds to seconds conversion factor
  double dt = (meas_package.timestamp_ - time_us_) * US_TO_S;
  // set last timestamp
  time_us_ = meas_package.timestamp_;

  // prediction
  this->Prediction(dt);

  // measurement update
  switch(meas_package.sensor_type_) {
    case MeasurementPackage::SensorType::LASER: {
      if (use_laser_) this->UpdateLidar(meas_package);
      break;
    }
    case MeasurementPackage::SensorType::RADAR: {
      if (use_radar_) this->UpdateRadar(meas_package);
      break;
    }
    default:
      throw std::runtime_error("Error! Received measurement from unknown SensorType");
  }
}

void UKF::Prediction(double delta_t) {
  /**
   * DONE: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

  /**
   * 1. Generate augmented sigma points.
   */

  // create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(5) = x_;
  x_aug(5) = 0; // noise mean is 0
  x_aug(6) = 0; // noise mean is 0

  // create augmented state covariance matrix
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_; // top left is P
  P_aug.bottomRightCorner(Q_.rows(), Q_.cols()) = Q_; // bottom right is Q

  // calculate square root matrix for generation of sigma points
  MatrixXd A = P_aug.llt().matrixL();
  // create augmented sigma points
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);
  Xsig_aug.fill(0.0);
  Xsig_aug.col(0) = x_aug; // first column is the mean vector
  for (size_t i = 1; i<= n_aug_; ++i) {
      VectorXd sig_offset = std::sqrt(lambda_ + n_aug_) * A.col(i - 1);
      // positive offset sigma point
      Xsig_aug.col(i) = x_aug + sig_offset;
      // negative offset sigma point
      Xsig_aug.col(i + n_aug_) = x_aug - sig_offset;
  }

  /**
   * 2. Predict sigma points.
   */

  for (size_t i = 0; i < n_sig_; ++i) {
    double px = Xsig_aug(0, i);
    double py = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double noise_a = Xsig_aug(5, i);
    double noise_yawdd = Xsig_aug(6, i);

    // avoid division by zero
    bool divByZero = std::fabs(yawd) < std::numeric_limits<double>::epsilon();
    double px_p = divByZero ? px + (v * delta_t * cos(yaw)) : px + (v / yawd * ( sin (yaw + yawd * delta_t) - sin(yaw)));
    double py_p = divByZero ? py + (v * delta_t * sin(yaw)) : py + (v / yawd * ( cos(yaw) - cos(yaw + yawd * delta_t)));
    double v_p = v + 0;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd + 0;

    // add noise
    double delta_t2 =  delta_t * delta_t;
    px_p +=  0.5 * delta_t2 * std::cos(yaw) * noise_a;
    py_p +=  0.5 * delta_t2 * std::sin(yaw) * noise_a;
    v_p += delta_t * noise_a;
    yaw_p += 0.5 * delta_t2 * noise_yawdd;
    yawd_p += delta_t * noise_yawdd;

    // set predicted sigma points into the corresponding column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }

  /**
   * 3. Predict mean and covariance.
   */

  // create vector for predicted state
  VectorXd x = VectorXd(n_x_);
  x.fill(0.0);

  // create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x_, n_x_);
  P.fill(0.0);

  // predict state mean
  for (size_t i = 0; i < n_sig_; ++i) {
    x += weights_(i) * Xsig_pred_.col(i);
  }
  // predict state covariance matrix
  for (size_t i = 0; i < n_sig_; ++i) {
    VectorXd x_delta = Xsig_pred_.col(i) - x;
    x_delta(3) = normalizeAngle(x_delta(3)); // normalize yaw
    P += weights_(i) * x_delta * x_delta.transpose();
  }

  // update state vector and covariance matrix
  x_ = x;
  P_ = P;
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * DONE: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  // set measurement dimension, laser can measure px and py
  static constexpr int n_z = 2;

  // transform sigma points into measurement space
  MatrixXd Z_sig = MatrixXd(n_z, n_sig_);
  Z_sig.fill(0.0);
  for (size_t i = 0; i < n_sig_; ++i) {
    Z_sig(0, i) = Xsig_pred_(0, i); // px
    Z_sig(1, i) = Xsig_pred_(1, i); // py
  }

  // calculate mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (size_t i = 0; i < n_sig_; ++i) {
    z_pred += weights_(i) * Z_sig.col(i);
  }

  // calculate innovation covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  // and cross correlatin matrix Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (size_t i = 0; i < n_sig_; ++i) {
    VectorXd z_delta = Z_sig.col(i) - z_pred;

    S += weights_(i) * z_delta * z_delta.transpose();

    VectorXd x_delta = Xsig_pred_.col(i) - x_;
    x_delta(3) = normalizeAngle(x_delta(3)); // normalize yaw

    Tc += weights_(i) * x_delta * z_delta.transpose();
  }
  S += R_laser_;

  // calculate Kalman gain K
  MatrixXd K = Tc * S.inverse();

  // update state mean and covariance matrix
  VectorXd z = meas_package.raw_measurements_;
  VectorXd z_delta = z - z_pred;
  x_ += K * z_delta;
  P_ -= K * S * K.transpose();
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * DONE: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  // set measurement dimension, radar can measure radial distance , velocity yaw, and velocity
  static constexpr int n_z = 3;

  // transform sigma points into measurement space
  MatrixXd Z_sig = MatrixXd(n_z, n_sig_);
  Z_sig.fill(0.0);
  for (size_t i = 0; i < n_sig_; ++i) {
    double px = Xsig_pred_(0,i);
    double py = Xsig_pred_(1,i);
    double v = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double vx = std::cos(yaw) * v;
    double vy = std::sin(yaw) * v;

    Z_sig(0, i) = std::sqrt(px * px + py * py); // radial distance
    Z_sig(1, i) = std::atan2(py, px); // yaw
    Z_sig(2, i) = (px * vx + py * vy) / Z_sig(0, i); // velocity
  }

  // calculate mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (size_t i = 0; i < n_sig_; ++i) {
    z_pred += weights_(i) * Z_sig.col(i);
  }

  // calculate innovation covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  // and cross correlatin matrix Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (size_t i = 0; i < n_sig_; ++i) {
    VectorXd z_delta = Z_sig.col(i) - z_pred;
    z_delta(1) = normalizeAngle(z_delta(1)); // normalize yaw

    S += weights_(i) * z_delta * z_delta.transpose();

    VectorXd x_delta = Xsig_pred_.col(i) - x_;
    x_delta(3) = normalizeAngle(x_delta(3)); // normalize yaw

    Tc += weights_(i) * x_delta * z_delta.transpose();
  }
  S += R_radar_;

  // calculate Kalman gain K
  MatrixXd K = Tc * S.inverse();

  // update state mean and covariance matrix
  VectorXd z = meas_package.raw_measurements_;
  VectorXd z_delta = z - z_pred;
  z_delta(1) = normalizeAngle(z_delta(1)); // normalize yaw
  x_ += K * z_delta;
  P_ -= K * S * K.transpose();
}
