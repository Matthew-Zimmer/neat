#include "ml.hpp"

#include <ordeal/ordeal.hpp>

#include <iostream>

using namespace Slate;
using namespace Slate::Machine_Learning;
using namespace Slate::Ordeal;

class Connection
{
    double w;
    std::size_t iid;
public:
    Connection(double weight) : w{ weight }
    {}
    auto& input_id()
    {
        return iid;
    }
    auto const& input_id() const
    {
        return iid;
    }
    auto& weight()
    {
        return w;
    }
    auto const& weight() const
    {
        return w;
    }
};

class My_Node : public Is<My_Node,
    Variables<
        Node::V::Id,
        Node::V::Value>,
    Features<
        Node::Connectable>>
{
public:
    My_Node(std::size_t id) : Inherit{ Node::V::Id{ id },  Node::V::Value{ 0.0 } }
    {}

    auto inputs()
    {
        return std::vector<Connection>{};
    }
};

class Network : public Is<Network, 
    Variables<
        Neural_Network::V::Nodes<My_Node>  >,//, 
        //Neural_Network::Fixed_Size_Output<My_Node, 1>>, 
    Features<
        Neural_Network::Dynamic, 
        Neural_Network::Sigmoid_Normalization>>
{
public:
    auto outputs()
    {
        return Math::Vector<1, My_Node>{ My_Node{ 1 } };
    }
};

class Neural_Network_Test : public Unit_Test<Neural_Network_Test>
{
    Network network;
public:
    Neural_Network_Test() : Unit_Test{ "neural network tests" }
    {
        //using namespace Neural_Network::Literals;
        network[0] --> network[1] = 0.5;
        network[1] --> network[2] = 2.0;
    }

    auto run(Test<0>)
    {
        return "sigmoid normalization"_name = Value{ network.normalize(1.0) } == Expected_Value{ 0.73 }.within(0.01);
    }

    auto run(Test<1>)
    {
        double d = network(Math::Vector<1, double>{ 1 })[0].value();
        std::cout << d << std::endl;
        return "dynamic evaluate"_name = Value{ d } == Expected_Value{ 1.0 }.within(0.01);
    }
};