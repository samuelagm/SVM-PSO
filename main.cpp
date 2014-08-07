#include <QCoreApplication>
#include <dlib/svm.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <time.h>
#include <math.h>
#include <string>

#include "Particle.h"

using namespace dlib;

#define random      (std::rand()/(double)(RAND_MAX+1))      // random(0,1)
#define GAMMA_IDX   0                                       // Gamma index within position vector
#define NU_IDX      1                                       // Nu index within position vector


typedef matrix<double, 2, 1> sample_type;
typedef radial_basis_kernel<sample_type> kernel_type;



double eval_fitness(Particle* p, std::vector<sample_type> samples, std::vector<double> labels);
void print_particle_info(Particle* p);

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);


    // Now we make objects to contain our samples and their respective labels.
    std::vector<sample_type> samples;
    std::vector<double> labels;

    // Now let's put some data into our samples and labels objects.  We do this
    // by looping over a bunch of points and labeling them according to their
    // distance from the origin.
    for (double r = -20; r <= 20; r += 0.8)
    {
        for (double c = -20; c <= 20; c += 0.8)
        {
            sample_type samp;
            samp(0) = r;
            samp(1) = c;
            samples.push_back(samp);

            // if this point is less than 10 from the origin
            if (sqrt(r*r + c*c) <= 10)
                labels.push_back(+1);
            else
                labels.push_back(-1);

        }
    }

    std::cout << "Generated " << samples.size() << " points" << std::endl;

    vector_normalizer<sample_type> normalizer;
    // let the normalizer learn the mean and standard deviation of the samples
    normalizer.train(samples);
    // now normalize each sample
    for (unsigned long i = 0; i < samples.size(); ++i)
        samples[i] = normalizer(samples[i]);

    randomize_samples(samples, labels);




    srand(time(NULL));

    std::vector<Particle*> swarm;                       // Particle Swarm
    Particle *gbest     =   new Particle();             // Global Best gbest
    int num_Iterations  =   500;                        // Maximum number of iterations
    int swarm_size      =   10;                         // Number of particles
    int dim             =   2;                          // Number of dimensions
    double MaxV         =   100;                        // Maximum velocity
    double MinV         =   -100;                       // Minimum velocity
    double MinGamma     =   1e-5;                       // Minimum Gamma
    double MaxGamma     =   5;                          // Maximum Gamma
    double MinNu        =   1e-5;                       // Minimum Nu
    double MaxNu        =   0.999*maximum_nu(labels);   // Maximum Nu
    double w            =   0.729;                      // Inertia weight
    double c1           =   1.49445;                    // Cognitive weight
    double c2           =   1.49445;                    // Social weight


    // Initializing Particles

    for (int i = 0; i < swarm_size; i++){


        std::vector<double> init_velocity,
                            init_position;

        for(int d = 0; d < dim; d++)
            init_velocity.push_back(((MaxV/3)  - (MinV/3)) * random + (MinV/3) );

        init_position.push_back( (MaxGamma - MinGamma) * random + MinGamma );        // Generating random Gamma  [MinGamma, MaxGamma]
        init_position.push_back( (MaxNu - MinNu) * random + MinNu );                 // Generating random Gamma  [MinNu, MaxNu]


        Particle *p = new Particle(init_position, init_position, init_velocity);
        double fitness = eval_fitness(p, samples, labels);
        p->setBestFitness(fitness);                                                   //Best Fitness and Fitness are equal initialy
        swarm.push_back(p);


    }



    gbest = swarm.at(0);


    print_particle_info(gbest);
    std::cout << std::endl;

    //Particle Swarm Optimization

    std::cout << "DOING PSO" << std::endl;
    try{



        while( num_Iterations > 0){
            for (int i = 0; i < swarm_size; i++){
                Particle *p = swarm.at(i);
                std::vector<double> newVelocity, newPosition;

                //Computing new velocity
                for(int d = 0; d < dim; d++){
                    double v = ( w * p->getVelocity().at(d) )
                            + c1 * random * (p->getBestPosition().at(d) - p->getPosition().at(d))
                            + c2 * random * (gbest->getBestPosition().at(d) - p->getPosition().at(d));

                    newVelocity.push_back(std::max(std::min(v, MaxV), MinV)) ;
                }

                p->getVelocity();

                p->setVelocity(newVelocity);                                            //Updating Particle Velocity

                //Computing new Particle Position
                double Gamma = p->getPosition().at(0) + newVelocity.at(0);
                double Nu = p->getPosition().at(1) + newVelocity.at(1);
                newPosition.push_back(std::max(std::min(Gamma, MaxGamma), MinGamma) );
                newPosition.push_back(std::max(std::min(Nu, MaxNu), MinNu) );


                p->setPosition(newPosition);                                            //Updating Particle Position


            }

            for (int i = 0; i < swarm_size; i++){

                Particle *p = swarm.at(i);
                eval_fitness(p, samples, labels);

                if(p->getBestFitness() < p->getFitness()){
                    p->setBestPosition(p->getPosition());
                    p->setBestFitness(p->getFitness());

                }

                if(gbest->getBestFitness() < p->getBestFitness()){
                    gbest = p;
                }

                print_particle_info(p);

            }


            std::cout << std::endl;


            print_particle_info(gbest);
            std::cout << std::endl << time(NULL) <<std::endl;

            --num_Iterations;
        } // END OF ALGORITHM




    }catch(std::exception& e){
        std::cout << e.what() << std::endl;
    }

    return a.exec();
}


double eval_fitness(Particle* p, std::vector<sample_type> samples, std::vector<double> labels){

    double gamma = p->getPosition().at(0);
    double nu = p->getPosition().at(1);

    double fitness;


    svm_nu_trainer<kernel_type> trainer;
    trainer.set_kernel(kernel_type(gamma));
    trainer.set_nu(nu);

    // Finally, do 10 fold cross validation and then check if the results are the best we have seen so far.
    matrix<double> result = cross_validate_trainer(trainer, samples, labels, 2);
    //std::cout << "gamma: " << std::setw(11) << gamma << "  nu: " << std::setw(11) << nu <<  "  cross validation accuracy: " << result;
    fitness = sum(result);
    p->setFitness(fitness);

    return fitness;
}

void print_particle_info(Particle* p){

    std::cout <<p->getName() + " :: Gamma: "<< std::setw(6)<<p->getBestPosition().at(GAMMA_IDX)<< " Nu: "
             << std::setw(6)<< p->getBestPosition().at(NU_IDX)
             << " fitness:"
             <<p->getBestFitness()<< std::endl;
}
