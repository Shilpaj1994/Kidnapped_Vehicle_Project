/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static default_random_engine gen;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 10;

	normal_distribution<double> N_x_init(0.0, std[0]);
	normal_distribution<double> N_y_init(0.0, std[1]);
	normal_distribution<double> N_theta_init(0.0, std[2]);

	for (int i = 0; i < num_particles; i++)
	{
		Particle p;
		p.id = i;
		p.x = x + N_x_init(gen);
		p.y = y + N_y_init(gen);
		p.theta = theta + N_theta_init(gen);
		p.weight = 1.0;
		particles.push_back(p);
	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	//describing a normal distribution with std deviation of the control measurements
	normal_distribution<double> x_control(0, std_pos[0]);
	normal_distribution<double> y_control(0, std_pos[1]);
	normal_distribution<double> theta_control(0, std_pos[2]);

	if (yaw_rate == 0)
	{
	  for (int i = 0; i < num_particles; i++)
	  {
		//these equations are to be used when the yaw_rate = 0
		particles[i].x += velocity*delta_t*cos(particles[i].theta);
		particles[i].y += velocity*delta_t*sin(particles[i].theta);

		//Adding control noise to the position and diretion of the predicted particles
		particles[i].x += x_control(gen);
		particles[i].y += y_control(gen);
		particles[i].theta += theta_control(gen);
	  }
	}
	else
	{
	  double factor = velocity / yaw_rate;

	  for (int i = 0; i < num_particles; i++)
	  {
		//updating the position of the particles according to the given control measurements
		double theta_f = particles[i].theta + yaw_rate*delta_t;
		particles[i].x += factor*(sin(theta_f) - sin(particles[i].theta));
		particles[i].y += factor*(cos(particles[i].theta) - cos(theta_f));
		particles[i].theta = theta_f;

		//Adding control noise to the position and diretion of the predicted particles
		particles[i].x += x_control(gen);
		particles[i].y += y_control(gen);
		particles[i].theta += theta_control(gen);
	  }
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	//current_dist is given a random large number value
	double Large_Number = 10000;
	double prev_dist = Large_Number;
	double distance;

	for (int i = 0; i < observations.size(); i++)
	{
		prev_dist = Large_Number;
		LandmarkObs o = observations[i];
		int map_id = -1;
		for (int j = 0; j < predicted.size(); j++)
		{
			LandmarkObs p = predicted[j];
			distance = dist(o.x, o.y, p.x, p.y);
			if (distance < prev_dist)
			{
				prev_dist = distance;
				map_id = predicted[j].id;
			}
		}
		observations[i].id = map_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	
	//calculate normalization term
	int obs_len = observations.size();
	std::vector<LandmarkObs> observation_vector;

	//created a new vector observation_vector to try and avoid any garbage values
	for (int i = 0; i < obs_len; i++)
	{
		observation_vector.push_back(observations[i]);
	}

	double gauss_norm = (1 / (2 * M_PI * std_landmark[0] * std_landmark[1]));

	//clearing all the previously added weights
	weights.clear();


	for (int i = 0; i < num_particles; i++)
	{
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

		//Create a vector of all the landmarks within the sensor range
		vector<LandmarkObs> nearby_landmarks;
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++)
		{
			double lm_x = map_landmarks.landmark_list[j].x_f;
			double lm_y = map_landmarks.landmark_list[j].y_f;
			int lm_id = map_landmarks.landmark_list[j].id_i;
	
			if (dist(p_x, p_y, lm_x, lm_y)<sensor_range)
			{
				nearby_landmarks.push_back(LandmarkObs{ lm_id, lm_x, lm_y });

			}
		}

		//Create a vector of all the transformed observations for the particle
		vector<LandmarkObs> transformed_obs;
		transformed_obs.clear();
		for (int k = 0; k < observations.size(); k++)
		{

			double x_transformed = observations[k].x * cos(particles[i].theta)
				- observations[k].y * sin(particles[i].theta) + particles[i].x;

			double y_transformed = observations[k].x * sin(particles[i].theta)
				+ observations[k].y * cos(particles[i].theta) + particles[i].y;

			transformed_obs.push_back(LandmarkObs{ observations[k].id,
				x_transformed, y_transformed });
		}


		dataAssociation(nearby_landmarks, transformed_obs);


		particles[i].weight = 1.0;


		for (int j = 0; j < transformed_obs.size(); j++)
		{
			int obs_id = transformed_obs[j].id;
			for (int k = 0; k < nearby_landmarks.size(); k++)
			{
				if (nearby_landmarks[k].id == obs_id)
				{
					//calculate exponent
					double mu_x = nearby_landmarks[k].x;
					double mu_y = nearby_landmarks[k].y;
					double obs_x = transformed_obs[j].x;
					double obs_y = transformed_obs[j].y;

					double exponent = ((pow((obs_x - mu_x), 2)) / (2 * pow(std_landmark[0], 2))) + ((pow((obs_y - mu_y), 2)) / (2 * pow(std_landmark[1], 2)));
					double multivariate = gauss_norm*exp(-exponent);

					particles[i].weight *= multivariate;

				}
			}
		}

		//Now we have the weight of every particle
		weights.push_back(particles[i].weight);

	}

	//normalizing the weights
	//double sum_of_weights = 0.0;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution


	// Vector for new particles
	vector<Particle> new_particles(num_particles);
		
	
	// Use discrete distribution to return particles by weight
	discrete_distribution<int> distribution(weights.begin(), weights.end());
	for (int i = 0; i < num_particles; ++i) {
		int number = distribution(gen);
		new_particles[i] = particles[number];
	
	}

	//weights.clear();
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
