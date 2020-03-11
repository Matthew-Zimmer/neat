#include <vector>
#include <unordered_map>
#include <cmath>

#include <reflection/reflection.hpp>
#include <reflection/variables.hpp>


namespace Slate::Machine_Learning
{
    namespace Neural_Network 
    {
        namespace Variable
        {
            using ::Slate::Variable::Base;
        }
        namespace V = Variable;

        template <typename Type>
        class Dynamic 
        {
        public:

        };
    }

    namespace Genetic 
    {   
        namespace Variable 
        {
            using ::Slate::Variable::Base;
        }
        namespace V = Variable;
    }
    
    namespace Neat 
    {
        namespace Variable 
        {
            using ::Slate::Variable::Base;
        }
        namespace V = Variable;
    }
}


int main()
{

}

// namespace Slate::ML
// {
//     namespace Variable 
//     {
//         using namespace Slate::Variable;

//         class Id : public Base<std::size_t>
//         {
//         public:
//             Variable_Type& id()
//             {
//                 return Variable();
//             }
//             Variable_Type const& id() const
//             {
//                 return Variable();
//             }
//         };

//         class Weight : public Base<double>
//         {
//         public:
//             Variable_Type& weight()
//             {
//                 return Variable();
//             }
//             Variable_Type const& weight() const
//             {
//                 return Variable();
//             }
//         };

//         class Input_Node : public Base<std::size_t>
//         {
//         public:
//             Variable_Type& input_node_id()
//             {
//                 return Variable();
//             }
//             Variable_Type const& input_node_id() const
//             {
//                 return Variable();
//             }
//         };

//         class Output_Node : public Base<std::size_t>
//         {
//         public:
//             Variable_Type& output_node_id()
//             {
//                 return Variable();
//             }
//             Variable_Type const& output_node_id() const
//             {
//                 return Variable();
//             }
//         };

//         class Enabled : public Base<bool>
//         {
//         public:
//             Variable_Type& enabled()
//             {
//                 return Variable();
//             }
//             Variable_Type const& enabled() const
//             {
//                 return Variable();
//             }
//         };
//     }
//     namespace V = Variable;

//     class Connection : public Is<Connection, Variables<V::Id, V::Input_Node, V::Output_Node, V::Enabled, V::Weight>>
//     {};

//     enum class Node_Type : char
//     {
//         input, 
//         output,
//         hidden,
//     };

//     namespace Variable
//     {
//         class Node_Type : public Base<Slate::Neat::Node_Type>
//         {
//         public:
//             Variable_Type& type()
//             {
//                 return Variable();
//             }
//             Variable_Type const& type() const
//             {
//                 return Variable();
//             }
//         };

//         class Input_Connections : public Base<std::vector<std::size_t>>
//         {
//         public:
//             Variable_Type& input_connections()
//             {
//                 return Variable();
//             }
//             Variable_Type const& input_connections() const
//             {
//                 return Variable();
//             }
//         };

//         class Output_Connections : public Base<std::vector<std::size_t>>
//         {
//         public:
//             Variable_Type& output_connections()
//             {
//                 return Variable();
//             }
//             Variable_Type const& output_connections() const
//             {
//                 return Variable();
//             }
//         };
//     }

//     template <typename Type>
//     class Evaluatable
//     {
//     public:
//         using Required_Variables = Meta::Wrap<V::Id, V::Input_Connections>;

//         template <typename Neural_Network>
//         double evaluate(std::unordered_map<std::size_t, double>& inputs, Neural_Network const& nn)
//         {
//             if (inputs.find(Meta::Cast<Type>(*this).id()) != inputs.end())
//             {
//                 return inputs[Meta::Cast<Type>(*this).id()];
//             }
//             else
//             {
//                 double r{ 0.0 };
//                 for (auto& id : Meta::Cast<Type>(*this).input_connection_ids)
//                     r += nn.input_node(id).evaluate(inputs, nn) * nn.connection(id).weight;
//                 return inputs[Meta::Cast<Type>(*this).id()] = Meta::Cast<Type>(*this).activate(r);
//             }
//         }
//     };

//     template <typename Type>
//     class Sigmoid_Activation
//     {
//     public:
//         double activate(double x)
//         {
//             return 1.0 / (1.0 + std::exp(-x));
//         }
//     };

//     class Basic_Node : public Is<Basic_Node, Variables<>, Features<Evaluatable, Sigmoid_Activation>>
//     {};

//     template <typename Type>
//     class Basic_Breeder
//     {
//     public:
//         void breed()
//         {
            
//         }
//     };

//     template <typename Type>
//     class Basic_Mutator
//     {
//     public:
//         void mutate()
//         {
            
//         }
//     };

//     template <typename Type>
//     class Basic_Populator
//     {
//     public:
//         void populate()
//         {

//         }
//     };

//     template <typename Type>
//     class Evolvable
//     {
//     public:
//         auto evolve(std::size_t generations)
//         {
//             Meta::Cast<Type>(*this).populate();
//             Meta::Cast<Type>(*this).evaluate();
//             for (std::size_t i{0}; i < generations; i++)
//             {
//                 Meta::Cast<Type>(*this).breed();
//                 Meta::Cast<Type>(*this).mutate();
//                 Meta::Cast<Type>(*this).evalute();
//             }
//         }
//     };

//     template <typename Node, typename Connection>
//     class Neural_Network
//     {
//         std::vector<Node> nodes;
//         std::vector<Connection> connections;
//         std::vector<std::size_t> ouput_nodes;
//     public:
//         std::vector<double> preform(std::unordered_map<std::size_t, double> inputs)
//         {
//             std::vector<double> values(ouput_nodes.size());
//             std::size_t c{ 0 };
//             for (auto id : ouput_nodes)
//                 values[c++] = this->node(id).evaluate(inputs, *this);
//             return values;
//         }

//         Connection& connection(std::size_t id)
//         {
//             return connections[id];
//         }
//         Connection const& connection(std::size_t id) const
//         {
//             return connections[id];
//         }

//         Node& node(std::size_t id)
//         {
//             return nodes[id];
//         }
//         Node const& node(std::size_t id) const
//         {
//             return nodes[id];
//         }

//         Node& input_node(std::size_t id)
//         {
//             return this->node(this->connection(id).input_node);
//         }
//         Node const& input_node(std::size_t id) const
//         {
//             return this->node(this->connection(id).input_node);
//         }

//         Node& output_node(std::size_t id)
//         {
//             return this->node(this->connection(id).output_node);
//         }
//         Node const& output_node(std::size_t id) const
//         {
//             return this->node(this->connection(id).output_node);
//         }
//     };

//     template <typename Type>
//     class Neatable
//     {
//     public:
//         void 
//     };

//     template <typename Type>
//     class Basic_Evolvable
//     {
//     public:
//         using Required_Features = Features<Evolvable, Basic_Mutator, Basic_Populator, Basic_Breeder>;
//     };

//     template <typename Type>
//     class Neat_Evolvable
//     {
//     public:
//         using Required_Features = Features<Basic_Evolvable>;
//         using Required_Variables = Variables<Neural_Network<Basic_Node, Connection>>;
//     };
// }