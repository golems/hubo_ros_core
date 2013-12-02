#include <ros/ros.h>
#include <std_msgs/Bool.h>
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
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/integral_image_normal.h>

#include <pcl/io/pcd_io.h>

#include <numeric>

#include <algorithm>
#include <vector>
#include <Eigen/Dense>

inline double max(const double a, const double b)
{
	return (a > b) ? a : b;
}

inline double min(const double a, const double b)
{
	return (a < b) ? a : b;
}

struct FilterSettings
{
	bool enableFilter;
	double intensityMin;
	double intensityMax;
	double intensityStdDevs;
	double distanceStdDevs;
	int numNeighbors;

	FilterSettings()
	{
		enableFilter = true;
		intensityMin = -std::numeric_limits<double>::max();
		intensityMax = std::numeric_limits<double>::max();
		intensityStdDevs = 2;
		distanceStdDevs = 2;
		numNeighbors = 20;
	}
};

std::ostream& operator <<(std::ostream& s, FilterSettings& i)
{
	s << "intensityMin: " << i.intensityMin << "\t";
	s << "intensityMax: " << i.intensityMax << "\t";
	s << "intensityStdDevs: " << i.intensityStdDevs << "\t";
	s << "distanceStdDevs: " << i.distanceStdDevs << "\t";

	return s;
}

struct IntensityStats
{
	double mean;
	double stdDev;
	double min;
	double max;
	//double median;

	IntensityStats()
	{
		max = -std::numeric_limits<double>::max();
		min = std::numeric_limits<double>::max();
		mean = 0.0;
		stdDev = 0.0;
		//median = 0.0;
	}

	typedef boost::shared_ptr<IntensityStats> Ptr;
};

std::ostream& operator <<(std::ostream& s, IntensityStats& i)
{
	s << "max: " << i.max << "\t";
	s << "min: " << i.min << "\t";
	s << "mean: " << i.mean << "\t";
	s << "stdDev: " << i.stdDev << "\t";

	return s;
}


IntensityStats::Ptr getStats(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud)
{
	IntensityStats::Ptr stats(new IntensityStats);

	/*
	for (int i = 0; i < cloud->points.size(); ++i)
	{
		//pcl::PointXYZI& p = cloud->points[i];
		float intensity = cloud->points[i].intensity;
		stats->mean += intensity;

		if (intensity > stats->max) { stats->max = intensity; }
		if (intensity < stats->min) { stats->min = intensity; }

	}

	stats->mean /= cloud->points.size();

	// TODO: more efficient implementation of this...
	for (int i = 0; i < cloud->points.size(); ++i)
	{
		float intensity = cloud->points[i].intensity;
		double delta = stats->mean - intensity;
		stats->stdDev += delta * delta;
	}

	stats->stdDev /= cloud->points.size();
	stats->stdDev = sqrt(stats->stdDev);
	*/

	for (int i = 0; i < cloud->points.size(); ++i)
	{
		float intensity = cloud->points[i].intensity;

		// Update Range Statistics
		if (intensity > stats->max) { stats->max = intensity; }
		if (intensity < stats->min) { stats->min = intensity; }

		// Update Distribution statistics
		float delta = intensity - stats->mean;
		stats->mean += (1.0/(float)i) * delta;
		stats->stdDev += delta * (intensity - stats->mean); // Variance
	}
	stats->stdDev = sqrt(stats->stdDev); // StdDev

	return stats;
}


FilterSettings g_settings;
bool doComputeNormals = true;

std::string g_fixed_frame;
//ros::Publisher g_cloud_publisher;
ros::Subscriber g_filter_enable_sub;
laser_geometry::LaserProjection g_laser_projector;
tf::TransformListener* g_transformer;

void cleanPointCloud(sensor_msgs::PointCloud2& cloudIn)
{
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::fromROSMsg(cloudIn, *cloud);

	IntensityStats::Ptr stats = getStats(cloud);

	pcl::PassThrough<pcl::PointXYZI> pass;
	pass.setInputCloud (cloud);
	pass.setKeepOrganized(true);
	pass.setFilterFieldName ("intensity");
	pass.setFilterLimits (max(g_settings.intensityMin, stats->mean - (g_settings.intensityStdDevs*stats->stdDev)),
						  min(g_settings.intensityMax, stats->mean + (g_settings.intensityStdDevs*stats->stdDev)));
	//pass.setFilterLimitsNegative (true);
	pass.filter (*cloud);

	pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
	sor.setInputCloud (cloud);
	sor.setKeepOrganized(true);
	sor.setMeanK (g_settings.numNeighbors);
	sor.setStddevMulThresh (g_settings.distanceStdDevs);
	sor.filter (*cloud);

	pcl::toROSMsg(*cloud, cloudIn);
}

void computeNormals(sensor_msgs::PointCloud2& cloudIn)
{
	std::cout << "Computing Normals." << std::endl;

	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::fromROSMsg(cloudIn, *cloud);

	pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);

	pcl::IntegralImageNormalEstimation<pcl::PointXYZI, pcl::Normal> ne;
	ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);
	ne.setMaxDepthChangeFactor(0.02f);
	ne.setNormalSmoothingSize(10.0f);
	ne.setKSearch(g_settings.numNeighbors/4);
	ne.setDepthDependentSmoothing(true);
	ne.setInputCloud(cloud);
	ne.compute(*normals);
	std::cout << "Computed Normals." << std::endl;
	pcl::PointCloud<pcl::PointXYZINormal>::Ptr fullCloud(new pcl::PointCloud<pcl::PointXYZINormal>);
	pcl::concatenateFields(*cloud, *normals, *fullCloud);

	pcl::io::savePCDFileASCII("/home/arprice/normals.pcd", *fullCloud);

	pcl::toROSMsg(*fullCloud, cloudIn);
}

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
	g_transformer->waitForTransform(g_fixed_frame, ros::Time::now(),
									"/lidar_optical_frame", ros::Time::now(),
									g_fixed_frame, ros::Duration(5.0));
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

		// Clean up messy lidar results
		if (g_settings.enableFilter)
		{
			cleanPointCloud(full_cloud);
		}

		// Estimate surface normals
		if (doComputeNormals && full_cloud.height > 1)
		{
			computeNormals(full_cloud);
		}
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

void filterEnableCB(const std_msgs::BoolConstPtr& on)
{
	g_settings.enableFilter = on->data;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lidar_aggregator");
    ros::NodeHandle nh;
    ros::NodeHandle nhp("~");

	nhp.getParam("intensity_floor", g_settings.intensityMin);
	nhp.getParam("intensity_ceiling", g_settings.intensityMax);
	nhp.getParam("intensity_stddevs", g_settings.intensityStdDevs);
	nhp.getParam("distance_stddevs", g_settings.distanceStdDevs);
	nhp.getParam("num_neighbors", g_settings.numNeighbors);
	nhp.getParam("compute_normals", doComputeNormals);

	g_filter_enable_sub = nh.subscribe("enable_filter", 1, filterEnableCB);

    tf::TransformListener listener(nh, ros::Duration(20000.0));
    g_transformer = &listener;
    ROS_INFO("Starting LIDAR aggregator...");
    nhp.param(std::string("fixed_frame"), g_fixed_frame, std::string("/torso_lift_link"));
	//g_cloud_publisher = nh.advertise<sensor_msgs::PointCloud2>("pointcloud", 1, true);
    ros::ServiceServer server = nh.advertiseService("aggregate_lidar", LaserAggregationServiceCB);
    ROS_INFO("LIDAR aggregator loaded");

	ros::spin();

    ROS_INFO("Shutting down LIDAR aggregator");
    return 0;
}
