#include <vector>
#include <string>


typedef std::vector<double> vd;

class Particle{
public:

    Particle():_name("Best Particle"), _position(0.0){}

    Particle(vd &position, vd &best_position, vd &velocity):_name("Particle")
    {
        _position = position;
        _best_position = best_position;
        _velocity = velocity;

    };

    vd &getPosition(){ return _position;}
    vd &getBestPosition(){return _best_position;}
    vd &getVelocity(){ return _velocity;}
    double getBestFitness(){ return _best_fitness;}
    double getFitness(){return _fitness;}
    std::string getName(){return _name;}

    void setBestPosition(vd &position){ _best_position = position;}
    void setFitness(double fitness){_fitness = fitness;}
    void setBestFitness(double fitness){_best_fitness = fitness;}
    void setVelocity(vd &velocity){ _velocity = velocity;}
    void setPosition(vd &position){_position = position;}

private:
    vd _position, _best_position, _velocity;
    std::string _name;

    double _fitness, _best_fitness;

};
