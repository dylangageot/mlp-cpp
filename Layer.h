//
// Created by dgageot on 20/12/2019.
//

#ifndef MLP_LAYER_H
#define MLP_LAYER_H

#include <vector>

class Layer {

public:
    /**
     * Construct a layer by specifing its size
     * @param size size of the layer
     */
    explicit Layer(const int& size) : _size(size), _outputs(size) {};
    /**
     * Construct a layer by setting up output values
     * Used to construct input layer of a neural network
     * @param outputs desired output values
     */
    explicit Layer(const std::vector<double>& outputs) : _size(outputs.size()), _outputs(outputs) {};
    /**
     * Compute output values by feeding the layer
     * @param layer feeding layer
     */
    void feed(const Layer& input);
    /**
     * Return size of the layer
     * @return layer size
     */
    inline int getSize() const { return _size; };
    /**
     * Return computed values of the layer
     * @return output values
     */
    inline std::vector<double> getOutputs() const { return _outputs; }
    /**
     * Return ID of the layer
     * @return layer ID
     */
    inline int getID() const { return _id; }

protected:
    /**
     * Size of the layer
     */
    int _size;
    /**
     * Output of the layer
     */
    std::vector<double> _outputs;
    /**
      * Weights of the layer
      */
    std::vector<double> _weigths;
    /**
     * Biases of the layer
     */
    std::vector<double> _biases;

private:
    /**
     * ID counter mechanism
     */
    int _id = _id_counter++;
    static int _id_counter;

    /**
     * Set-up weights and biases to correspond with input layer
     * @param input
     */
    void setUp(const Layer& input);

};

#endif //MLP_LAYER_H