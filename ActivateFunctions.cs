class ActivateFunctions
{
    // Scalar version
    public static double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));
    public static double Tanh(double x) => Math.Tanh(x);

    // Vector version (element-wise)
    public static double[] Sigmoid(double[] x) => Array.ConvertAll(x, Sigmoid);
    public static double[] Tanh(double[] x) => Array.ConvertAll(x, Tanh);
}
