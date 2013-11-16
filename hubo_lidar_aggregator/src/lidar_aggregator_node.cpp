#include <ros/ros.h>
#include <math.h>
#include <string>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include <laser_geometry/laser_geometry.h>
#include <hubo_sensor_msgs/LidarAggregation.h>

// PCL specific includes
#include <pcl/ros/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/io.h>

#include <algorithm>
#include <vector>
#include <Eigen/Dense>

std::string g_fixed_frame;
ros::Publisher g_cloud_publisher;
laser_geometry::LaserProjection g_laser_projector;
tf::TransformListener* g_transformer;

bool TimeLess(const sensor_msgs::LaserScan& s1, const sensor_msgs::LaserScan& s2) {
	return s1.header.stamp.toSec() < s2.header.stamp.toSec();
}

bool TimeGreater(const sensor_msgs::LaserScan& s1, const sensor_msgs::LaserScan& s2) {
	return s1.header.stamp.toSec() > s2.header.stamp.toSec();
}

double ComputeTiltVelocity(const std::vector<dynamixel_msgs::JointState>& tilts) {
	if(tilts.size() == 0)
		return 0;
 	const double startup_delay = 0.1; // Time for motor to reach steady velocity
 	int ss = 0;                       // steady state index
 	double min_time = tilts[0].header.stamp.toSec();
 	for(int i=0; i < tilts.size(); i++) {
 		if(tilts[i].header.stamp.toSec() - min_time > startup_delay) {
 			ss = i;
 			break;
 		}
 	}
 	Eigen::MatrixXd A(tilts.size() - ss, 2);
	Eigen::VectorXd b(tilts.size() - ss);
	for(int i=0; i < tilts.size(); i++) {
		if(i >= ss) {
			A(i - ss,0) = tilts[i].header.stamp.toSec() - min_time;
			A(i - ss,1) = 1;
			b(i - ss) = tilts[i].current_pos;
		}
	}
	Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);
	ROS_INFO("Estimated lidar tilt velocity: %f", x(0));
	return x(0);
}

bool LaserAggregationServiceCB(hubo_sensor_msgs::LidarAggregation::Request& req, hubo_sensor_msgs::LidarAggregation::Response& res)
{
    ROS_INFO("Attempting to aggregate %ld laser scans into a pointcloud", req.Scans.size());
	g_transformer->waitForTransform("/head_base_link", ros::Time::now(),
									"/lidar_optical_frame", ros::Time::now(),
									"/head_base_link", ros::Duration(5.0));
    sensor_msgs::PointCloud2 full_cloud;
    if (req.Scans.size() > 0)
    {
	    if(ComputeTiltVelocity(req.Tilts) > 0)
		    std::sort(req.Scans.begin(), req.Scans.end(), TimeLess);
	    else
		    std::sort(req.Scans.begin(), req.Scans.end(), TimeGreater);
	    int points_per_scan = -1; // It's not ranges.size()! (due to min/max angle truncation in LaserProjector)
		g_laser_projector.transformLaserScanToPointCloud(g_fixed_frame, req.Scans[0], full_cloud, *g_transformer); // Setting the range cutoff doesn't work, I'm not going to bother figuring out why
		for (int index = 1; index < req.Scans.size(); index++)
        {
            sensor_msgs::PointCloud2 scan_cloud;
            g_laser_projector.transformLaserScanToPointCloud(g_fixed_frame, req.Scans[index], scan_cloud, *g_transformer); // Setting the range cutoff doesn't work, I'm not going to bother figuring out why
            bool succeded = pcl::concatenatePointCloud(full_cloud, scan_cloud, full_cloud);
            if (!succeded)
            {
                ROS_ERROR("PCL could not concatenate pointclouds");
            }
            // Check if all scanlines generate the same number of points
            if(points_per_scan == -1)
	            points_per_scan = scan_cloud.width;
            else if(points_per_scan != scan_cloud.width)
	            points_per_scan = -2;
        }
        if(points_per_scan > 0) {
	        full_cloud.height = req.Scans.size();
	        full_cloud.width = full_cloud.width/full_cloud.height; // Doing this right prevents malloc segfaults in rviz, otherwise leave it unorganized
        }
        ROS_INFO("Created cloud of width: %d, height: %d.", full_cloud.width, full_cloud.height);
    }
    else
    {
        ROS_WARN("No laser scans to aggregate");
    }
    full_cloud.header.frame_id = g_fixed_frame;
    full_cloud.header.stamp = ros::Time::now();
    res.header.frame_id = g_fixed_frame;
    res.header.stamp = full_cloud.header.stamp;
    res.Cloud = full_cloud;
    // g_cloud_publisher.publish(full_cloud);
    return true;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lidar_aggregator");
    ros::NodeHandle nh;
    ros::NodeHandle nhp("~");
    tf::TransformListener listener(nh, ros::Duration(20000.0));
    g_transformer = &listener;
    ROS_INFO("Starting LIDAR aggregator...");
    nhp.param(std::string("fixed_frame"), g_fixed_frame, std::string("/torso_lift_link"));
    g_cloud_publisher = nh.advertise<sensor_msgs::PointCloud2>("pointcloud", 1, true);
    ros::ServiceServer server = nh.advertiseService("aggregate_lidar", LaserAggregationServiceCB);
    ROS_INFO("LIDAR aggregator loaded");
    while (ros::ok())
    {
        ros::spinOnce();
    }
    ROS_INFO("Shutting down LIDAR aggregator");
    return 0;
}
