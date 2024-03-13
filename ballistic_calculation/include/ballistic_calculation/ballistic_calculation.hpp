#ifndef BCHEADER
#define BCHEADER

#include <auto_aim_interfaces/msg/detail/target__struct.hpp>
#include <geometry_msgs/msg/detail/vector3__struct.hpp>
#include <geometry_msgs/msg/detail/point__struct.hpp>
#include <ceres/ceres.h>
#include <ceres/jet.h>

//STD
#include <memory>
#include <string>
#include <vector>
#include<cmath>

#include "auto_aim_interfaces/msg/target.hpp"

namespace rm_auto_aim
{
using target = auto_aim_interfaces::msg::Target;

class Ballistic
{
public:
Ballistic(double bulletV , target target_msg , double k = 0.1 , double K1 = 0.1 , double K2 = 0.05); //构造函数

target target_msg;
geometry_msgs::msg::Point robotcenter = target_msg.position;
geometry_msgs::msg::Vector3 velocity = target_msg.velocity;
double bulletV;
double K1;//第一次大迭代时的步长，需要parameter_declare来调整参数
double K2;//第一次大迭代时的步长，需要parameter_declare来调整参数
double k; //空气阻力系数，需要parameter_declare来调整参数
double theta;//计算t时变动的临时参数    

//iteration
std::pair<double,double> iteration1(double &thres , double &init_pitch , double &initT ); //迭代,返回pitch和T，传入T和pitch的初始值

std::pair<double,double> iteration2(double &thres , double &init_pitch , double &initT , double& x , double& y); //迭代,返回pitch和T，传入T和pitch的初始值
//figure out best armor,express the final distance's function of T,using the result to iteration
std::vector<double> predictBestArmor(double T); //找出最佳装甲板,返回该装甲板的预测中心坐标

double magnitude(const std::vector<double>& v);//计算向量的模

struct CostFunctor1 {
    
    Ballistic& ballistic_ref;
    
    explicit CostFunctor1( Ballistic& ballistic) : ballistic_ref(ballistic){}

    template <typename T>
    bool operator()(const T* const t, T* residual) const {
        T v0 = T(ballistic_ref.bulletV); // Set the initial value for v0
        T futurex = ballistic_ref.robotcenter.x + ballistic_ref.velocity.x * (*t);
        T futurey = ballistic_ref.robotcenter.y + ballistic_ref.velocity.y * (*t);
        
        residual[0] = T(1 / ballistic_ref.k)*ceres::log(T(ballistic_ref.k)*v0 * (*t) + T(1.0)) - T(ceres::sqrt(futurex * futurex + futurey * futurey)) ;
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
        residual[0] =T(1 / ballistic_ref.k)*ceres::log(T(ballistic_ref.k)*v0 * (*t) + T(1.0)) - T(ceres::sqrt(x_aim * x_aim + y_aim * y_aim)) ;
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

double sortFourdoubles(double a, double b, double c, double d);



};

}//namespace rm_auto_aim


#endif // BCHEADER