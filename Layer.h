//
// Created by dgageot on 20/12/2019.
//

#ifndef MLP_LAYER_H
#define MLP_LAYER_H

#include <vector>

#define LEARNING_RATE 0.01

class Layer {

public:
    /**
     * Construct a layer by specifing its size
     * @param size size of the layer
     */
    explicit Layer(const int& size) : _size(size), _outputs(size), _weights(size),
        _biases(size), _weightedInputs(size), _gradientErrors(size) {};
    /**
     * Compute output values by feeding the layer
     * @param input feeding layer
     */
    void feed(const Layer& input);
    /**
     * Compute output values by feeding the layer
     * @param input feeding vector
     */
    void feed(const std::vector<double>& input);
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
    /**
     * Return a weighted errors for backpropagating at the left layer
     * @return  weighted errors of the layer
     */
    std::vector<double> getWeightedErrors() const;
    /**
     * Backpropagate by computing weighted input error
     * @param weightedErrors
     */
    void backpropagate(const std::vector<double>& weightedErrors);
    /**
     * Backpropagate by computing weighted input error
     * @param layer right layer
    */
    void backpropagate(const Layer& layer);
    /**
     * Compute gradient descent with the specified inputs
     * @param inputs output activation of previous layer
     */
    void gradientDescent(const std::vector<double>& inputs, int miniBatchSize);
    /**
     * Compute gradient descent with the specified left layer
     * @param inputs left layer
     */
    void gradientDescent(const Layer& input, int miniBatchSize);

    void resetGradientErrors();

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
    std::vector<std::vector<double>> _weights;
    /**
     * Biases of the layer
     */
    std::vector<double> _biases;
    /**
     * Weighted input
     */
    std::vector<double> _weightedInputs;
    /**
     * Gradient error on inputs
     */
    std::vector<double> _gradientErrors;

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
    void setUp(const std::vector<double>& input);

};

#endif //MLP_LAYER_H