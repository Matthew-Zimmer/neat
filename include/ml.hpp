#pragma once
#include <numeric>
#include <set>
#include <vector>
#include <unordered_map>

#include <math/vector.hpp>

#include <random/random.hpp>

#include <reflection/reflection.hpp>
#include <reflection/variables.hpp>

#include <iostream>

template <typename Type>
constexpr bool False = false;

template <typename Type>
void FAIL()
{
    static_assert(False<Type>);
}

template <typename Type>
void FAIL(Type)
{
    static_assert(False<Type>);
}

namespace Slate::Machine_Learning
{
    namespace Neural_Network 
    {
        namespace Variable
        {
            using ::Slate::Variable::Base;
            template <typename Type>
            class Nodes : public Base<std::vector<Type>>
            {
            public:
                auto& nodes() { return this->variable(); }
                auto const& nodes() const { return this->variable(); }
            };
        }
        namespace V = Variable;

        template <typename Type>
        class Dynamic 
        {
        public:
            auto& operator[](std::size_t index)
            {
                return Meta::cast<Type>(*this).nodes()[index];
            }
        };

        template <typename Type>
        class Evaluatable
        {
            template <typename Input>
            auto& operator()(std::size_t id, std::unordered_map<std::size_t, Input>& values)
            {
                if (values.find(id) != values.end())
                {
                    return values[id];
                }
                else
                {
                    auto inputs = Meta::cast<Type>(*this)[id].inputs();
                    values[id].value() = Meta::cast<Type>(*this).normalize(std::accumulate(inputs.begin(), inputs.end(), 0.0, [&](double x, auto const& y){ return x + Meta::cast<Type>(*this)(y.input_id(), values).value() * y.weight(); }));
                    return values[id];
                }
            }
        public:
            template <std::size_t Dim, typename Input>
            auto operator()(Math::Vector<Dim, Input> const& input) // requires(requires(Type x){ x[0]; })
            {
                std::unordered_map<std::size_t, Input> values;
                std::size_t c{ 0 };
                for (auto& x : input)
                    values[c++] = x;
                auto v = Meta::cast<Type>(*this).outputs();
                for (auto& x : v)
                    x.value() = Meta::cast<Type>(*this)(x.id(), values).value();
                return v;
            }
            // template <typename Input>
            // auto operator()(Input const& input) // requires(!requires(Type x){ x[0]; })
            // {
            //     //do fancy matrix multiplication
            // }
        };

        template <typename Type>
        class Sigmoid_Normalization
        {
        public:
            double normalize(double x)
            {
                return 1.0 / (1.0 + std::exp(-x));
            }
        };

        template <typename Type>
        class Randomizable
        {
        public:
            void randomize() 
            {

            }
            void randomly_adjust_weight() 
            {
                auto& inputs = Random::element(Meta::cast<Type>(*this).nodes()).inputs();
                auto& weight = Random::element(inputs).weight();
                weight = std::clamp(weight + Random::number(Meta::cast<Type>(*this).weight_adjustment_delta()), 0.0, 1.0);
            }
            void randomly_change_weight() 
            {
                auto& inputs = Random::element(Meta::cast<Type>(*this).nodes()).inputs();
                auto& weight = Random::element(inputs).weight();
                weight = Random::number(0.0, 1.0);
            }
            void randomly_add_connection() 
            {
                auto& input_node = Random::element(Meta::cast<Type>(*this).nodes());
                auto& output_node = Random::element(input_node.unconnected_nodes());
                input_node.connect_to(output_node);
            }
            void randomly_add_node() 
            {
                auto& node = Random::element(Meta::cast<Type>(*this).nodes());
                auto& connection = Random::element(node.inputs());
                connection.split();
            }
        };
    }

    namespace Genetic 
    {   
        namespace V 
        {
            using ::Slate::Variable::Base;

            template <typename Type>
            class Organisms : public Base<std::vector<Type>>
            {
            public:
                auto& organisms()
                {
                    return this->variable();
                }
                auto& organisms() const
                {
                    return this->variable();
                }
            };

            class Population_Size : public Base<std::size_t>
            {
            public:
                auto& population_size()
                {
                    return this->variable();
                }
                auto& population_size() const
                {
                    return this->variable();
                }
            };
        }

        template <typename Type>
        class Populator 
        {
        public:
            void populate()
            {
                Meta::cast<Type>(*this).organisms().clear();
                Meta::cast<Type>(*this).organisms().resize(Meta::cast<Type>(*this).population_size());
                for (auto& org : Meta::cast<Type>(*this).organisms())
                    org.randomize();
            }
        };

        template <typename Type>
        class Evaluator 
        {
        public:
            void evaluate()
            {
                auto& organisms = Meta::cast<Type>(*this).organisms();
                for (auto& org : organisms)
                    org.evaluate();
                std::sort(organisms.begin(), organisms.end(), std::greater{});
            }

            auto& best()
            {
                return Meta::cast<Type>(*this).organisms()[0];
            }

            auto const& best() const
            {
                return Meta::cast<Type>(*this).organisms()[0];
            }

            bool is_closeness_measure() const
            {
                auto const& organisms = Meta::cast<Type>(*this).organisms();
                return !(organisms.size() == 1 || organisms[0].score() > organisms[1].score() && organisms[0] > organisms[1]);
            }
        };

        template <typename Type>
        class Breeder 
        {
        public:
            void breed()
            {
                using Organism = typename Type::Organism;
                std::vector<Organism> next_generation(Meta::cast<Type>(*this).organisms().size());
                Organism* p1, *p2;
                auto weights = Meta::cast<Type>(*this).breeding_weights();
                // int k = 1;
                // for (auto& o : Meta::cast<Type>(*this).organisms())
                //     if (!(k++ % 10))
                //         k = 1, std::cout << o.score() << std::endl;
                // std::cout << "----------------------" << std::endl;
                // static int i = 0;
                // if (i++ == 5)
                //     std::exit(1);
                std::discrete_distribution<std::size_t> dis{ weights.begin(), weights.end() };
                for (std::size_t i = 0; i < next_generation.size();i++)
                {
                    p1 = &Random::element(Meta::cast<Type>(*this).organisms(), dis);
                    p2 = &Random::element(Meta::cast<Type>(*this).organisms(), dis);
                    next_generation[i] = p1->breed_with(*p2);
                }

                Meta::cast<Type>(*this).organisms() = std::move(next_generation);
            }

            std::vector<double> breeding_weights()
            {
                auto& organisms = Meta::cast<Type>(*this).organisms();
                std::vector<double> r;
                if (Meta::cast<Type>(*this).is_closeness_measure())
                {
                    double sum = std::accumulate(organisms.begin(), organisms.end(), 0.0, [](double d, auto const& x){ return d + x.score(); });
                    std::transform(organisms.begin(), organisms.end(), std::back_inserter(r), [&](auto const& x){ return 1 / x.score(); });
                }
                else
                {
                    std::transform(organisms.begin(), organisms.end(), std::back_inserter(r), [&](auto const& x){ return x.score(); });
                }
                return r;
            }
        };

        template <typename Type>
        class Mutator 
        {
        public:
            void mutate()
            {
                for (auto& org : Meta::cast<Type>(*this).organisms())
                    org.mutate();
            }
        };

        template <typename Type>
        class Evolvable 
        {
        public:
            template <typename Predicate>
            auto evolve_util(Predicate&& pred)
            {
                Meta::cast<Type>(*this).populate();
                Meta::cast<Type>(*this).evaluate();
                int g = 1;
                while (pred(Meta::cast<Type>(*this).best()))
                {
                    Meta::cast<Type>(*this).breed();
                    Meta::cast<Type>(*this).mutate();
                    Meta::cast<Type>(*this).evaluate();
                    //std::cout << g++ << ", " << Meta::cast<Type>(*this).best().value() << std::endl;
                }
                return Meta::cast<Type>(*this).best();
            }

            auto evolve_for(std::size_t generations)
            {
                std::size_t gen = 0;
                return Meta::cast<Type>(*this).evolve_util([&](auto const&){ return gen++ < generations; });
            }
        };

        template <typename Organism_Type>
        class Specie : public Is<Specie<Organism_Type>, Variables<V::Organisms<Organism_Type>, V::Population_Size>, Features<Populator, Evaluator, Breeder, Mutator, Evolvable>>
        {
        public: 
            using Organism = Organism_Type;
            Specie(std::size_t population_size) : Is<Specie<Organism_Type>, Variables<V::Organisms<Organism_Type>, V::Population_Size>, Features<Populator, Evaluator, Breeder, Mutator, Evolvable>>{ V::Population_Size{ population_size } }
            {}
        };
    }
    
    namespace Neat 
    {
        namespace V
        {
            using ::Slate::Variable::Base;
            class Weight : public Base<double> 
            {
            public:
                auto& weight() { return this->variable(); }
                auto const& weight() const { return this->variable(); }
            };

            class Value : public Base<double> 
            {
            public:
                auto& value() { return this->variable(); }
                auto const& value() const { return this->variable(); }
            };

            class Id : public Base<std::size_t> 
            {
            public:
                auto& id() { return this->variable(); }
                auto const& id() const { return this->variable(); }
            };
        }

        class Connection : public Is<Connection, Variables<V::Weight>>
        {
        public:
            std::size_t input_id() const
            {
                return 0;
            }

            void split()
            {

            }
        };

        class Node;

        namespace V 
        {
            class Inputs : public Base<std::vector<Connection>>
            {
            public:
                auto& inputs() { return this->variable(); }
                auto const& inputs() const { return this->variable(); }
            };

            class Unconnected_Nodes : public Base<std::vector<Node>>
            {
            public:
                auto& unconnected_nodes() { return this->variable(); }
                auto const& unconnected_nodes() const { return this->variable(); }
            };
        }

        class Node : public Is<Node, Variables<V::Value, V::Id, V::Inputs, V::Unconnected_Nodes>>
        {
            void remove_unconnectedness(Node const& n)
            {
                unconnected_nodes().erase(std::remove(unconnected_nodes().begin(), unconnected_nodes().end(), n), unconnected_nodes().end());
            }
        public:
            void connect_to(Node& n) 
            {
                inputs().push_back(Connection{});
                
                remove_unconnectedness(n);
                n.remove_unconnectedness(*this);
            }

            friend bool operator==(Node const& left, Node const& right)
            {
                return left.id() == right.id();
            }
        };

        template <typename Type>
        class Brain : public Is<Brain<Type>, Variables<Neural_Network::V::Nodes<Node>>, Features<Neural_Network::Dynamic, Neural_Network::Randomizable, Neural_Network::Sigmoid_Normalization, Neural_Network::Evaluatable>>
        {
        public:
            double weight_adjustment_delta()
            {
                return 0.1;
            }

            auto outputs()
            {
                return Meta::Unwrap<Meta::Args<decltype(&Type::score)>>{};
            }
        };
        namespace V
        {
            template <typename Type>
            class Brain : public Base<::Slate::Machine_Learning::Neat::Brain<Type>>
            {
            public:
                auto& brain(){ return this->variable(); }
                auto const& brain() const { return this->variable(); }
            };

            class Fitness : public Base<double>
            {
            public:
                auto& fitness() { return this->variable(); }
                auto const& fitness() const { return this->variable(); }
            };
        }

        template <typename Type>
        class Mutator
        {
        public:
            void mutate()
            {
                Random::choice
                (
                    std::tuple{ Meta::cast<Type>(*this).adjust_weight_chance(), [this](){ Meta::cast<Type>(*this).brain().randomly_adjust_weight(); } },
                    std::tuple{ Meta::cast<Type>(*this).change_weight_chance(), [this](){ Meta::cast<Type>(*this).brain().randomly_change_weight(); } },
                    std::tuple{ Meta::cast<Type>(*this).add_connection_chance(), [this](){ Meta::cast<Type>(*this).brain().randomly_add_connection(); } },
                    std::tuple{ Meta::cast<Type>(*this).add_node_chance(), [this](){ Meta::cast<Type>(*this).brain().randomly_add_node(); } }
                );
            }
            double adjust_weight_chance()
            {
                return 0.0;
            }
            double change_weight_chance()
            {
                return 0.0;
            }
            double add_connection_chance()
            {
                return 0.0;
            }
            double add_node_chance()
            {
                return 0.0;
            }
        };

        template <typename Type>
        class Evaluator
        {
        public:
            void evaluate()
            {
                Meta::cast<Type>(*this).fitness() = Meta::cast<Type>(*this).score(Meta::cast<Type>(*this).brain()(Meta::cast<Type>(*this).inputs()));                
            }
            friend bool operator<(Type const& left, Type const& right)
            {
                return left.fitness() < right.fitness();
            }
        };

        template <typename Type>
        class Breeder 
        {
        public:
            auto breed_with(Type& other)
            {
                return other;
            }
        };

        template <typename Type>
        class Randomizer
        {
        public:
            void randomize()
            {
                Meta::cast<Type>(*this).brain().randomize();
            }
        };

        template <typename Type>
        class Creature
        {
        public:
            using Required_Features = Features<Mutator, Evaluator, Breeder, Randomizer>;
            using Required_Variables = Variables<V::Brain<Type>, V::Fitness>;
        };

        template <typename Type>
        class Brainless_Creature
        {
        public:
            using Required_Features = Features<Mutator, Evaluator, Breeder, Randomizer>;
            using Required_Variables = Variables<V::Fitness>;
        };
    }
}