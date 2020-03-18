#include "ml.hpp"

#include <ordeal/ordeal.hpp>

using namespace Slate;
using namespace Slate::Machine_Learning;
using namespace Slate::Ordeal;

class Connection
{
    std::size_t iid;
    double w;
public:
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

class Node
{
    double v;
    std::size_t node_id;

public:
    Node() = default;
    Node(double v) : v{ v }
    {}

    auto& value()
    {
        return v;
    }
    auto const& value() const
    {
        return v;
    }
    auto& id()
    {
        return node_id;
    }
    auto const& id() const
    {
        return node_id;
    }
    auto inputs()
    {
        return std::vector<Connection>{};
    }
};

class Network : public Is<Network, Variables<Neural_Network::V::Nodes<Node>>, Features<Neural_Network::Dynamic, Neural_Network::Evaluatable, Neural_Network::Sigmoid_Normalization>>
{
public:
    auto outputs()
    {
        return Math::Vector<1, Node>{ 0.0 };
    }
};

class Neural_Network_Test : public Unit_Test<Neural_Network_Test>
{
    Network network;
public:
    Neural_Network_Test() : Unit_Test{ "neural network tests" }
    {
        network[0] = Node{ 0.0 };
    }

    auto run(Test<0>)
    {
        return "dynamic evaluate"_name = Value{ network(Math::Vector<1, Node>{ 1.0 })[0].value() } == Expected_Value{ 0.85 }.within(0.01);
    }
};