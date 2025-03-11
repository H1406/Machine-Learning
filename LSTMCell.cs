using System;

public class LSTMCell
{
    private readonly int _inputSize;
    private int hiddenSize;
    private double[,] Wf, Wi, Wo, Wc; // Weight matrices
    private double[] bf, bi, bo, bc;   // Bias vectors
    private double[] h, c;             // Hidden state and cell state

    // Intermediate values for backpropagation
    private double[] ft, it, ot, cTilde; // Gate activations
    private double[] concat;             // Concatenated input and hidden state

    public LSTMCell(int inputSize, int hiddenSize)
    {
        _inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        Random rand = new Random();

        // Initialize weights and biases
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

    // Randomly initialize a matrix
    private double[,] RandomMatrix(int rows, int cols, Random rand)
    {
        double[,] matrix = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = rand.NextDouble() * 2 - 1; // Range: [-1, 1]
            }
        }
        return matrix;
    }

    // Forward pass
    public double[] Forward(double[] input)
    {
        // Concatenate input and hidden state
        concat = new double[input.Length + h.Length];
        Array.Copy(h, 0, concat, 0, h.Length);
        Array.Copy(input, 0, concat, h.Length, input.Length);

        // Compute gate activations
        ft = ActivateFunctions.Sigmoid(MatrixVectorMultiply(Wf, concat, bf)); // Forget gate
        it = ActivateFunctions.Sigmoid(MatrixVectorMultiply(Wi, concat, bi)); // Input gate
        cTilde = ActivateFunctions.Tanh(MatrixVectorMultiply(Wc, concat, bc)); // Candidate cell state
        ot = ActivateFunctions.Sigmoid(MatrixVectorMultiply(Wo, concat, bo)); // Output gate

        // Update cell state and hidden state
        for (int i = 0; i < c.Length; i++)
        {
            c[i] = ft[i] * c[i] + it[i] * cTilde[i]; // New cell state
            h[i] = ot[i] * Math.Tanh(c[i]);          // New hidden state
        }

        return h;
    }

    // Backward pass (gradient computation)
    public (double[,] dWf, double[,] dWi, double[,] dWc, double[,] dWo, double[] dbf, double[] dbi, double[] dbc, double[] dbo) Backward(double[] dLoss_dh, double[] dLoss_dc_next)
    {
        // Gradients for output gate
        double[] dot = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++)
        {
            dot[i] = dLoss_dh[i] * Math.Tanh(c[i]) * ActivateFunctions.Sigmoid(ot[i]);
        }

        // Gradients for cell state
        double[] dc = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++)
        {
            dc[i] = dLoss_dh[i] * ot[i] * (1 - Math.Pow(Math.Tanh(c[i]), 2)) + dLoss_dc_next[i];
        }

        // Gradients for input gate
        double[] dit = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++)
        {
            dit[i] = dc[i] * cTilde[i] * ActivateFunctions.Sigmoid(it[i]);
        }

        // Gradients for forget gate
        double[] dft = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++)
        {
            dft[i] = dc[i] * c[i] * ActivateFunctions.Sigmoid(ft[i]);
        }

        // Gradients for candidate cell state
        double[] dcTilde = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++)
        {
            dcTilde[i] = dc[i] * it[i] * (1 - Math.Pow(cTilde[i], 2));
        }

        // Gradients for weights and biases
        double[,] dWf = OuterProduct(dft, concat);
        double[,] dWi = OuterProduct(dit, concat);
        double[,] dWc = OuterProduct(dcTilde, concat);
        double[,] dWo = OuterProduct(dot, concat);

        double[] dbf = dft;
        double[] dbi = dit;
        double[] dbc = dcTilde;
        double[] dbo = dot;

        return (dWf, dWi, dWc, dWo, dbf, dbi, dbc, dbo);
    }

    // Update weights and biases using gradients
    public void UpdateWeights(double[,] dWf, double[,] dWi, double[,] dWc, double[,] dWo, double[] dbf, double[] dbi, double[] dbc, double[] dbo, double learningRate)
    {
        UpdateMatrix(ref Wf, dWf, learningRate);
        UpdateMatrix(ref Wi, dWi, learningRate);
        UpdateMatrix(ref Wc, dWc, learningRate);
        UpdateMatrix(ref Wo, dWo, learningRate);

        UpdateVector(ref bf, dbf, learningRate);
        UpdateVector(ref bi, dbi, learningRate);
        UpdateVector(ref bc, dbc, learningRate);
        UpdateVector(ref bo, dbo, learningRate);
    }

    // Helper method to update a matrix
    private void UpdateMatrix(ref double[,] matrix, double[,] gradients, double learningRate)
    {
        for (int i = 0; i < matrix.GetLength(0); i++)
        {
            for (int j = 0; j < matrix.GetLength(1); j++)
            {
                matrix[i, j] -= learningRate * gradients[i, j];
            }
        }
    }

    // Helper method to update a vector
    private void UpdateVector(ref double[] vector, double[] gradients, double learningRate)
    {
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] -= learningRate * gradients[i];
        }
    }

    // Compute outer product of two vectors
    private double[,] OuterProduct(double[] a, double[] b)
    {
        double[,] result = new double[a.Length, b.Length];
        for (int i = 0; i < a.Length; i++)
        {
            for (int j = 0; j < b.Length; j++)
            {
                result[i, j] = a[i] * b[j];
            }
        }
        return result;
    }

    // Matrix-vector multiplication
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