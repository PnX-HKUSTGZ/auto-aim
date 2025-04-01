#ifndef BCHEADER
#define BCHEADER

#include <auto_aim_interfaces/msg/detail/target__struct.hpp>
#include <geometry_msgs/msg/detail/vector3__struct.hpp>
#include <geometry_msgs/msg/detail/point__struct.hpp>
#include <ceres/ceres.h>
#include <ceres/jet.h>

//STD

#include <vector>
#include <utility>
#include <Eigen/Dense>



namespace rm_auto_aim
{
using target = auto_aim_interfaces::msg::Target;

class Ballistic
{
public:

Ballistic(double k = 0.1 , double K1 = 0.3 , double K2 = 0.3, double bulletV = 30, Eigen::Vector3d odom2gun = Eigen::Vector3d(0,0,0)); //构造函数

target target_msg;
geometry_msgs::msg::Point robotcenter;
geometry_msgs::msg::Vector3 velocity;
double last_yaw; 
double bulletV;
double K1;//第一次大迭代时的步长，需要parameter_declare来调整参数
double K2;//第一次大迭代时的步长，需要parameter_declare来调整参数
double k; //空气阻力系数，需要parameter_declare来调整参数
double theta;//计算t时变动的临时参数    
Eigen::Vector3d odom2gun; //机器人坐标系到枪口坐标系的变换

// 迭代中心,返回pitch和T，传入T和pitch的初始值
std::pair<double,double> iteration1(double &thres , double &init_pitch , double &initT );

// 找出步兵的最佳装甲板,返回该装甲板的预测中心坐标
struct armor_info{
    double x;
    double y;
    double z;
    std::vector<double>vec;
    std::vector<double>vecto_odom;
    double yaw;
    double r;
};
std::vector<double> predictInfantryBestArmor(double T, double min_v, double max_v, double v_yaw_PTZ); 
std::vector<double> stategy_1(double T); 
std::vector<double> stategy_2(double T, double v_yaw_PTZ);
double findYaw(double v_yaw, double v_yaw_PTZ, double distance, double radius); 

// 迭代装甲板,返回pitch和T，传入T和pitch的初始值
std::pair<double,double> iteration2(double &thres , double &init_pitch , double &initT , double& yaw , double& z , double& r); 

//ceres计算出一个T后进入该函数，去迭代出一个对准固定T预测的装甲板(固定dist)的theta和T
std::pair<double , double> fixTiteratPitch(double& horizon_dis , double& height);



double magnitude(const std::vector<double>& v);//计算向量的模

struct CostFunctor1 {
    
    Ballistic& ballistic_ref;
    
    explicit CostFunctor1( Ballistic& ballistic) : ballistic_ref(ballistic){}

    template <typename T>
    bool operator()(const T* const t, T* residual) const {
        T v0 = T(ballistic_ref.bulletV); // Set the initial value for v0
        T futurex = ballistic_ref.robotcenter.x + ballistic_ref.velocity.x * (*t);
        T futurey = ballistic_ref.robotcenter.y + ballistic_ref.velocity.y * (*t);
        
        residual[0] = T(1 / ballistic_ref.k)*ceres::log(T(ballistic_ref.k)*ceres::cos(ballistic_ref.theta)*v0 * (*t) + T(1.0)) - T(ceres::sqrt(futurex * futurex + futurey * futurey)) ;
        
        return true;
    }
};

struct CostFunctor2 {

    Ballistic& ballistic_ref;
    double x;
    double y;

    explicit CostFunctor2( Ballistic& ballistic , double& x , double& y) : ballistic_ref(ballistic) , x(x) , y(y){}

    template <typename T>
    bool operator()(const T* const t, T* residual) const {
        T v0 = T(ballistic_ref.bulletV);
        
        T x_aim = T(x);
        T y_aim = T(y);
        residual[0] =T(1 / ballistic_ref.k)*ceres::log(T(ballistic_ref.k)*ceres::cos(ballistic_ref.theta)*v0 * (*t) + T(1.0)) - T(ceres::sqrt(x_aim * x_aim + y_aim * y_aim)) ;
        return true;
        }
};

struct HitAim{

    double x;
    double y;
    double yaw;
};//描述最终击打目标

double optimizeTime1(double initial_guess);

double optimizeTime2(double initial_guess , double& x , double& y);

double dotProduct(const std::vector<double>& v1, const std::vector<double>& v2);

double angleBetweenVectors(const std::vector<double>& v1, const std::vector<double>& v2);


};

}//namespace rm_auto_aim


#endif // BCHEADER