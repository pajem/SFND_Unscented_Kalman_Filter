#include <stdexcept>

#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

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
  std_a_ = 30; // TODO: change to a more approriate value

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30; // TODO: change to a more approriate value
  
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
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

  // the UKF will be initialized on first sensor measurement
  is_initialized_ = false;

  // initialize state covariance matrix as identity matrix
  P_.setIdentity();

  // dimensions and parameters
  n_x_ = 5; // for CTRV model (px, py, v, yaw, yaw rate)
  n_aug_ = n_x_ + 2; // augment with noise vector (longitudinal acceleration and yaw acceleration)
  n_sig_ = 2 * n_aug_ + 1; // 2 sigma points for each state variable, plus 1 sigma point for the mean
  lambda_ = 3 - n_aug_; // use typical lambda design parameter value

  // generate weights for predicted mean and covariance
  weights_ = VectorXd(n_sig_);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (size_t i = 1; i < n_sig_; ++i) {
      weights_(i) = 0.5 / (lambda_ + n_aug_);
  }

  // initialize predicted sigma points matrix
  Xsig_pred_ = Eigen::MatrixXd(n_x_, n_aug_);
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
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
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
}