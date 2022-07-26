using System;
using System.Collections.Generic;

namespace XOR
{
    internal static class Program
    {
        public static void Main(string[] args)
        {
            NeuralNetwork neuralNetwork = new NeuralNetwork(2, 2, 1);
            neuralNetwork.Training(10000, 0.3, 0.7);
        }
    }

    class Synapse
    {
        public Synapse(double weight)
        {
            Weight = weight;
        }

        public double Weight;
    }

    class Neuron
    {
        public Neuron()
        {
            InputData = 0;
            OutputData = 0;
            
        }
        
        public double InputData;
        public double OutputData;
    }

    class NeuralNetwork
    {
        public NeuralNetwork(params int[] counterLayers)
        {
            for (int i = 0; i < counterLayers[0]; i++)
            {
                _inputLayer.Add(new Neuron());
            }

            for (int i = 0; i < counterLayers[1]; i++)
            {
                _hiddenLayer.Add(new Neuron());
            }

            for (int i = 0; i < counterLayers[2]; i++)
            {
                _outputLayer.Add(new Neuron());
            }

            for (int i = 0; i < _inputLayer.Count; i++)
            {
                for (int j = 0; j < _hiddenLayer.Count; j++)
                {
                    Synapse synapse = new Synapse(0.3 + (i + j) / 8);
                    _synapses.Add(synapse);
                }
            }
            
            for (int i = 0; i < _hiddenLayer.Count; i++)
            {
                for (int j = 0; j < _outputLayer.Count; j++)
                {
                    Synapse synapse = new Synapse(0.3 + (i + j) / 8);
                    _synapses.Add(synapse);
                }
            }

            double[] weights = new[] {0.45, 0.78, -0.12, 0.13, 1.5, -2.3};

            for (int i = 0; i < 6; i++)
            {
                _synapses[i].Weight = weights[i];
            }
        }
        
        private double Sigmoid(double value) {
            return (1.0 / (1.0 + Math.Pow(Math.E, -value)));
        }

        public void Training(int countEpoch, double moment, double speed)
        {
            double A = moment;
            double E = speed;

            double[] weight = new[] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            
            List<List<double>> trainingSet = new List<List<double>>{
                new List<double>{1, 0, 1},
                new List<double>{0, 1, 1}, 
                new List<double>{0, 0, 0}, 
                new List<double>{1, 1, 0}
            };

            for (int i = 0; i < countEpoch; i++)
            {
                Console.WriteLine($"Epoch:{i}");
                foreach (var set in trainingSet)
                {
                    _hiddenLayer[0].InputData = _synapses[0].Weight * set[0] + _synapses[2].Weight * set[1];
                    _hiddenLayer[0].OutputData = Sigmoid(_synapses[0].Weight * set[0] + _synapses[2].Weight * set[1]);

                    _hiddenLayer[0].InputData = _synapses[1].Weight * set[0] + _synapses[3].Weight * set[1];
                    _hiddenLayer[0].OutputData = Sigmoid(_synapses[1].Weight * set[0] + _synapses[3].Weight * set[1]);

                    _outputLayer[0].InputData = _synapses[4].Weight * _hiddenLayer[0].OutputData + _synapses[5].Weight * _hiddenLayer[1].OutputData;
                    _outputLayer[0].OutputData = Sigmoid(_outputLayer[0].InputData);

                    double error = (Math.Pow(set[2] - _outputLayer[0].OutputData, 2)) / 1;
                    
                    Console.WriteLine($"Result --- {_outputLayer[0].OutputData}; Error --- {error}");

                    double deltaO1 = (set[2] - _outputLayer[0].OutputData) *
                                     ((1 - _outputLayer[0].OutputData) * _outputLayer[0].OutputData);

                    double deltaH1 = ((1 - _hiddenLayer[0].OutputData) * _hiddenLayer[0].OutputData) *
                                     (_synapses[4].Weight * deltaO1);

                    double deltaH2 = ((1 - _hiddenLayer[1].OutputData) * _hiddenLayer[1].OutputData) *
                                     (_synapses[5].Weight * deltaO1);

                    double gradW6 = _hiddenLayer[1].OutputData * deltaO1;
                    double gradW5 = _hiddenLayer[0].OutputData * deltaO1;
                    double gradW4 = _inputLayer[1].OutputData * deltaH2;
                    double gradW3 = _inputLayer[1].OutputData * deltaH2;
                    double gradW2 = _inputLayer[0].OutputData * deltaH1;
                    double gradW1 = _inputLayer[0].OutputData * deltaH1;

                    _synapses[5].Weight += E * gradW6 + weight[5] * A;
                    _synapses[4].Weight += E * gradW5 + weight[4] * A;
                    _synapses[3].Weight += E * gradW4 + weight[3] * A;
                    _synapses[2].Weight += E * gradW3 + weight[2] * A;
                    _synapses[1].Weight += E * gradW2 + weight[1] * A;
                    _synapses[0].Weight += E * gradW1 + weight[0] * A;
                    
                    weight[5] = E * gradW6 + weight[5] * A;
                    weight[4] = E * gradW5 + weight[4] * A;
                    weight[3] = E * gradW4 + weight[3] * A;
                    weight[2] = E * gradW3 + weight[2] * A;
                    weight[1] = E * gradW2 + weight[1] * A;
                    weight[0] = E * gradW1 + weight[0] * A;
                }
                
                Console.WriteLine("");
            }
        }

        private List<Neuron> _inputLayer = new List<Neuron>();
        private List<Neuron> _hiddenLayer = new List<Neuron>();
        private List<Neuron> _outputLayer = new List<Neuron>();

        private List<Synapse> _synapses = new List<Synapse>();
    }
}