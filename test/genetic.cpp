#include "ml.hpp"

#include <ordeal/ordeal.hpp>

using namespace Slate;
using namespace Slate::Machine_Learning;
using namespace Slate::Ordeal;

int target_number = 125487;

class Number_Guesser
{
    int num;
    double fitness;
public:
    double score() const
    {
        return fitness;
    }
    int value() const
    {
        return num;
    }
    void randomize()
    {
        num = Random::number(0, 10000000);
    }
    void evaluate()
    {
        fitness = std::pow(target_number - num, 2);
    }
    auto breed_with(Number_Guesser const& o) 
    {
        Number_Guesser g;
        g.fitness = 0.0;
        g.num = (num + o.num) / 2;
        return g;
    }
    void mutate()
    {
        Random::choice
        (
            std::tuple{ 0.25, [this](){ num++; } },
            std::tuple{ 0.25, [this](){ num--; } }//,
            //std::tuple{ 0.01, [this](){ num = Random::number(0, 10000000); } }
        );
    }
    friend bool operator>(Number_Guesser const& left, Number_Guesser const& right)
    {
        return left.fitness < right.fitness;
    }
};


class Genetic_Test : public Unit_Test<Genetic_Test>
{
    Genetic::Specie<Number_Guesser> guessers{ 1000 };
public:
    Genetic_Test() : Unit_Test{ "genetic algo tests" }
    {}

    auto run(Test<0>)
    {
        guessers.populate();
        return "populate"_name = Value{ guessers.organisms() }.expects([](auto const& org){ return org.value() >= 0 && org.score() == 0; });
    }

    auto run(Test<1>)
    {
        guessers.evaluate();
        return "evaluating"_name = Value{ guessers.organisms() }.expects([](auto const& org){ return org.score() >= 0; }); 
    }

    auto run(Test<2>)
    {
        guessers.breed();
        return "breeding"_name = Value{ guessers.organisms() }.expects([](auto const& org){ return org.value() >= 0 && org.score() == 0; }); 
    }

    auto run(Test<3>)
    {
        guessers.mutate();
        return "mutating"_name = Value{ guessers.organisms() }.expects([](auto const& org){ return org.value() >= 0 && org.score() == 0; }); 
    }

    auto run(Test<4>)
    {
        auto x = guessers.evolve_for(150);
        return "trained"_name = Value{ x.value() } == Expected_Value{ target_number }.within(1000);
    }
};