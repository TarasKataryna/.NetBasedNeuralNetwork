namespace NeuralNetwork.Common
{
    public delegate double ActivateFunction(double element);

    public enum LayerType
    {
        PerceptronLayer = 1,
        CovolutionalLayer = 2,
        PoolingLayer = 3,
        ReluLayer = 4,
        FlattenLayer = 5
    }

}
