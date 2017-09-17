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
default_random_engine gen;
void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	// Following function samples GPS measurement
	std::default_random_engine gen;

	// Normally distribute each GPS measurement into N particle
	std::normal_distribution<double> N_x(x, std[0]);
	std::normal_distribution<double> N_y(y, std[1]);
	std::normal_distribution<double> N_theta(theta, std[2]);
	for(int i=0;i<num_particles;i++){
		// Generate id's for each particle
		Particle particle;
		particle.id = i;
		particle.x = N_x(gen);
		particle.y = N_y(gen);
		particle.theta = N_theta(gen);
		particle.weight =1;

		particles.push_back(particle);
		weights.push_back(1);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t,  double std_pos[],  double velocity,  double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine gen;


    // loop trough all particles
    for (int i = 0; i < num_particles; ++i) {


        // generating Gaussian noise for each particles
        normal_distribution<double> dist_x(0.0, std_pos[0]);
        normal_distribution<double> dist_y(0.0, std_pos[1]);
        normal_distribution<double> dist_theta(0.0, std_pos[2]);



        // If yaw rate differs from zero
        if (abs(yaw_rate) >= 1.0E-6) {

            particles[i].x += (velocity / yaw_rate)*(sin(particles[i].theta + (yaw_rate*delta_t)) - sin(particles[i].theta))+dist_x(gen);
            particles[i].y += (velocity / yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate*delta_t)))+ dist_y(gen);
            particles[i].theta += (yaw_rate * delta_t)+ dist_theta(gen);
        }


            // If yaw_rate is close to zero
        else {
            particles[i].x += velocity * delta_t * cos(particles[i].theta)+ dist_x(gen);
            particles[i].y += velocity * delta_t * sin(particles[i].theta)+ dist_y(gen);
            particles[i].theta += (yaw_rate * delta_t) + dist_theta(gen);
        }
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations)
{
    //   observed measurement to this particular landmark.
    //   implement this method and use it as a helper during the updateWeights phase.
    // Current Observation
    double min_distance, dist, dx, dy;
    int min_i;

    for(unsigned obs_i = 0; obs_i < observations.size(); obs_i++)
    {
        auto obs = observations[obs_i];

        min_distance = INFINITY;
        min_i = -1;
        for(unsigned i = 0; i < predicted.size(); i++)
        {
            auto pred_lm = predicted[i];
            dx = (pred_lm.x - obs.x);
            dy = (pred_lm.y - obs.y);
            dist = dx*dx + dy*dy;
            if(dist < min_distance)
            {
                min_distance = dist;
                min_i = i;
            }
        }
        observations[obs_i].id = min_i;
    }
}

const LandmarkObs local_to_global(const LandmarkObs& obs, const Particle& p)
{
    LandmarkObs out;

    // First rotate the local coordinates to the right orientation
    out.x = p.x + obs.x * cos(p.theta) - obs.y * sin(p.theta);
    out.y = p.y + obs.x * sin(p.theta) + obs.y * cos(p.theta);
    out.id = obs.id;
    return out;
}

inline const double gaussian_2d(const LandmarkObs& obs, const LandmarkObs &lm, const double sigma[])
{
    auto cov_x = sigma[0]*sigma[0];
    auto cov_y = sigma[1]*sigma[1];
    auto normalizer = 2.0*M_PI*sigma[0]*sigma[1];
    auto dx = (obs.x - lm.x);
    auto dy = (obs.y - lm.y);
    return exp(-(dx*dx/(2*cov_x) + dy*dy/(2*cov_y)))/normalizer;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    double sigma_landmark [2] = {0.3, 0.3}; // Landmark measurement uncertainty [x [m], y [m]]

    for(unsigned p_ctr=0; p_ctr < particles.size(); p_ctr++)
    {
        auto p = particles[p_ctr];

        std::vector<LandmarkObs> predicted_landmarks;

        for(auto lm : map_landmarks.landmark_list)
        {
            LandmarkObs lm_pred;
            lm_pred.x = lm.x_f;
            lm_pred.y = lm.y_f;
            lm_pred.id = lm.id_i;
            auto dx = lm_pred.x - p.x;
            auto dy = lm_pred.y - p.y;

            // Add only if in range
            if(dx*dx + dy*dy <= sensor_range*sensor_range)
                predicted_landmarks.push_back(lm_pred);
        }
        std::vector<LandmarkObs> transformed_obs;
        double total_prob = 1.0f;

        // Transform observations from vehicle cooridnate to map coordinate
        for(auto obs_lm : observations)
        {
            auto obs_global = local_to_global(obs_lm, p);
            transformed_obs.push_back(std::move(obs_global));
        }
        // Stores index of associated landmark in the observation
        dataAssociation(predicted_landmarks, transformed_obs);

        for(unsigned i=0; i < transformed_obs.size(); i++)
        {
            auto obs = transformed_obs[i];
            // Assume sorted by id and starting at 1
            auto assoc_lm = predicted_landmarks[obs.id];

            double pdf = gaussian_2d(obs, assoc_lm, sigma_landmark);
            total_prob *= pdf;
        }
        particles[p_ctr].weight = total_prob;
        weights[p_ctr] = total_prob;
    }
    std::cout<<std::endl;
}

void ParticleFilter::resample() {
    std::discrete_distribution<int> d(weights.begin(), weights.end());
    std::vector<Particle> new_particles;

    for(unsigned i = 0; i < num_particles; i++)
    {
        auto ind = d(gen);
        new_particles.push_back(std::move(particles[ind]));
    }
    particles = std::move(new_particles);
}


Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
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
