using System;
public class LSTMCell{
    private int inputSize;
    private int hiddenSize;
    private double[,] Wf,Wi,Wo,Wc;
    private double[] bf,bi,bo,bc;
    private double[] h,c;
    public LSTMCell(int inputSize,int hiddenSize){
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        Random rand = new Random();
        Wf = RandomMatrix(hiddenSize, inputSize + hiddenSize, rand);
        Wi = RandomMatrix(hiddenSize, inputSize + hiddenSize, rand);
        Wc = RandomMatrix(hiddenSize, inputSize + hiddenSize, rand);
        Wo = RandomMatrix(hiddenSize, inputSize + hiddenSize, rand);

        bf = new double[hiddenSize];
        bi = new double[hiddenSize];
        bc = new double[hiddenSize];
        bo = new double[hiddenSize];

        h = new double[hiddenSize];
        c = new double[hiddenSize];
    }

    public double[,]? RandomMatrix(int rows, int cols, Random rand)
    {
        double[,] matrix = new double[rows, cols];
        for (int i = 0; i < rows; i++){
            for (int j = 0; j < cols; j++){
                matrix[i, j] = rand.NextDouble()*2 -1;
            }
        }
        return matrix;
    }
    public double[] Forward(double[] input){
        double[] concat = new double[input.Length + h.Length];
        Array.Copy(h, 0, concat, 0, h.Length);
        Array.Copy(input, 0, concat, h.Length, input.Length);

        double[] ft = ActivateFunctions.Sigmoid(MatrixVectorMultiply(Wf, concat, bf));
        double[] it = ActivateFunctions.Sigmoid(MatrixVectorMultiply(Wi, concat, bi));
        double[] cTilde = ActivateFunctions.Tanh(MatrixVectorMultiply(Wc, concat, bc));
        double[] ot = ActivateFunctions.Sigmoid(MatrixVectorMultiply(Wo, concat, bo));

        for (int i = 0; i < c.Length; i++)
        {
            c[i] = ft[i] * c[i] + it[i] * cTilde[i];
            h[i] = ot[i] * Math.Tanh(c[i]);
        }

        return h;
    }
    private double[] MatrixVectorMultiply(double[,] matrix, double[] vector, double[] bias)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        double[] result = new double[rows];

        for (int i = 0; i < rows; i++)
        {
            result[i] = bias[i];
            for (int j = 0; j < cols; j++)
            {
                result[i] += matrix[i, j] * vector[j];
            }
        }
        return result;
    }
}